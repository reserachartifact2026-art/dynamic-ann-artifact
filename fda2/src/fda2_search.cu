#include "fda2_search.h"
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <vector>

extern int g_greedyTag;   // DEBUG: defined (global scope) in baseline_src/greedySearch.cu

namespace fda2 {

void allocateSearchBuffers(SearchBuffers* sb, unsigned qbs) {
    sb->maxBatch = qbs;
    gpuErrchk(cudaMalloc(&sb->d_topkIds,   (size_t)qbs * MAX_L * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&sb->d_topkDists, (size_t)qbs * MAX_L * sizeof(float)));
    gpuErrchk(cudaMallocHost(&sb->h_topkIds, (size_t)qbs * MAX_L * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&sb->d_queryVecs, (size_t)qbs * D * sizeof(GVEC)));
    gpuErrchk(cudaMalloc(&sb->d_visitedSets,
                          (size_t)qbs * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&sb->d_visitedCounts, (size_t)qbs * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&sb->d_visitedSetDists,
                          (size_t)qbs * MAX_PARENTS_PERQUERY * sizeof(float)));
    gpuErrchk(cudaMalloc(&sb->d_qIdx, (size_t)qbs * sizeof(unsigned)));
    gpuErrchk(cudaMallocHost(&sb->h_qIdx, (size_t)qbs * sizeof(unsigned)));
    allocateGreedySearchBuffers(&sb->gsBuffers, qbs);
    sb->allocated = true;
}

// One block per query; threads cooperate on the D-element copy.
__global__ void gatherQueryVecsKernel(const float* d_allQ, const unsigned* d_qIdx,
                                      unsigned bs, GVEC* d_out) {
    unsigned i = blockIdx.x;
    if (i >= bs) return;
    const float* src = d_allQ + (size_t)d_qIdx[i] * D;
    GVEC* dst = d_out + (size_t)i * D;
    for (unsigned d = threadIdx.x; d < D; d += blockDim.x) dst[d] = src[d];
}

void gatherQueries(const float* d_allQueryVecs, SearchBuffers* sb,
                   const unsigned* h_indices, unsigned bs, cudaStream_t stream) {
    for (unsigned i = 0; i < bs; i++) sb->h_qIdx[i] = h_indices[i];
    gpuErrchk(cudaMemcpyAsync(sb->d_qIdx, sb->h_qIdx, (size_t)bs * sizeof(unsigned),
                              cudaMemcpyHostToDevice, stream));
    gatherQueryVecsKernel<<<bs, BLOCK_D, 0, stream>>>(
        d_allQueryVecs, sb->d_qIdx, bs, sb->d_queryVecs);
}

void freeSearchBuffers(SearchBuffers* sb) {
    if (!sb->allocated) return;
    cudaFree(sb->d_topkIds);
    cudaFree(sb->d_topkDists);
    cudaFreeHost(sb->h_topkIds);
    cudaFree(sb->d_queryVecs);
    cudaFree(sb->d_visitedSets);
    cudaFree(sb->d_visitedCounts);
    cudaFree(sb->d_visitedSetDists);
    cudaFree(sb->d_qIdx);
    cudaFreeHost(sb->h_qIdx);
    freeGreedySearchBuffers(&sb->gsBuffers);
    sb->allocated = false;
}

// Extract top-k from the (ascending) worklist. Stride between per-query rows is
// searchL (the baseline greedy lays the worklist out with that stride).
__global__ void extractTopKKernel(
    const unsigned* d_worklist, const float* d_worklistDist,
    const unsigned* d_worklistCount,
    unsigned bs, unsigned k, unsigned searchL,
    unsigned* d_topkIds, float* d_topkDists)
{
    unsigned q = blockIdx.x;
    if (q >= bs) return;
    unsigned valid = d_worklistCount[q];
    if (valid > searchL) valid = searchL;
    for (unsigned i = threadIdx.x; i < k; i += blockDim.x) {
        if (i < valid) {
            d_topkIds[(size_t)q * k + i]   = d_worklist[(size_t)q * searchL + i];
            d_topkDists[(size_t)q * k + i] = d_worklistDist[(size_t)q * searchL + i];
        } else {
            d_topkIds[(size_t)q * k + i]   = 0xFFFFFFFFu;
            d_topkDists[(size_t)q * k + i] = FLT_MAX;
        }
    }
}

// NOTE: a top-k post-filter kernel (strip deleted ids from results) was removed —
// measured redundant. The greedy traversal already excludes deleted nodes from the
// beam (see searchBatch), so the top-k is naturally all-live with no holes; a
// post-filter changed nothing (99.60%/0 leak with or without it) and, used alone,
// only masked leaks while leaving holes (98.08%). Deleted-node handling in search is
// the traversal-time skip only.

void searchBatch(const UnifiedGraph* g,
                 const GVEC* d_queries,
                 unsigned bs,
                 unsigned k,
                 unsigned searchL,
                 SearchBuffers* sb,
                 cudaStream_t stream)
{
    if (bs == 0) return;

    setRuntimeMedoid(g->medoid, stream);

    gpuErrchk(cudaMemsetAsync(sb->d_visitedCounts, 0,
                              (size_t)bs * sizeof(unsigned), stream));
    // BANG greedy, lock-free, non-fused. For large query BATCHES this beats the
    // fused single-block-per-query kernel: the per-iteration kernels parallelize
    // each step across many threads (e.g. computeDists at R×8 threads) and
    // amortize launch overhead across the whole batch. (The fused kernel — 64
    // threads/query, L=150 beam — is better only for low-latency single queries;
    // measured ~60-70K vs ~110-120K q/s here.) d_visitedSetDists kept for the
    // fused path if ever re-enabled.
    g_greedyTag = 1;   // DEBUG: tag search greedy
    // Deleted nodes are excluded from the beam during greedy expansion (traversal
    // skip): a deleted neighbor is never added to the worklist, so the top-k is
    // naturally all-live. This is load-bearing while delete is approximate — measured
    // on deep sliding_window (500K del, searchL=100): on → 99.60%/0 leak; off →
    // 98.08%/1.59% leak (stale q→p edges keep deleted points reachable). Disable for
    // A/B with FDA2_NO_TRAVERSAL_SKIP=1 (or FDA2_NO_FILTER=1). FDA2_LEAK_DIAG=1 reports
    // deleted-in-topk. Once delete is made exact this skip becomes removable too.
    static const bool traversalSkip = getenv("FDA2_NO_TRAVERSAL_SKIP") == nullptr
                                   && getenv("FDA2_NO_FILTER") == nullptr;
    greedySearchNoSyncPrealloc(
        g->d_graph, const_cast<GVEC*>(d_queries),
        sb->d_visitedSets, sb->d_visitedCounts,
        /*batchStart*/ 0, bs, searchL,
        /*d_deleted=*/ traversalSkip ? g->d_deleted : nullptr,
        &sb->gsBuffers, stream);

    // Comment 3 diagnostic: the Visited Set must never expand children of its
    // own entries — its size is bounded by the greedy beam. For searchL=L the
    // number of expanded parents stays well under ~4·L (e.g. < 200 at L=50).
    // Enable with FDA2_VISITED_DIAG=1 (adds one D2H sync — off the hot path).
    if (getenv("FDA2_VISITED_DIAG")) {
        std::vector<unsigned> hv(bs);
        gpuErrchk(cudaMemcpyAsync(hv.data(), sb->d_visitedCounts,
                                  (size_t)bs * sizeof(unsigned),
                                  cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        unsigned mx = 0; double sum = 0.0;
        for (unsigned i = 0; i < bs; i++) { if (hv[i] > mx) mx = hv[i]; sum += hv[i]; }
        printf("[VISITED-DIAG] searchL=%u batch=%u  max_visited=%u  avg=%.1f  "
               "(bound ~4*searchL=%u; MAX_PARENTS_PERQUERY=%d)\n",
               searchL, bs, mx, bs ? sum / bs : 0.0, 4 * searchL, MAX_PARENTS_PERQUERY);
    }

    extractTopKKernel<<<bs, 32, 0, stream>>>(
        sb->gsBuffers.d_worklist, sb->gsBuffers.d_worklistDist,
        sb->gsBuffers.d_worklistCount,
        bs, k, searchL, sb->d_topkIds, sb->d_topkDists);

    gpuErrchk(cudaMemcpyAsync(sb->h_topkIds, sb->d_topkIds,
                              (size_t)bs * k * sizeof(unsigned),
                              cudaMemcpyDeviceToHost, stream));

    // Comment-4 validation (GT-independent): count how many returned top-k ids are
    // CURRENTLY deleted — a "leak". With the traversal skip on (default) this is 0;
    // with it off (FDA2_NO_TRAVERSAL_SKIP) the leak rate exposes how approximate the
    // delete is (stale q→p edges). Enable with FDA2_LEAK_DIAG=1 (adds a D2H sync +
    // a deleted-array copy — off the hot path).
    if (getenv("FDA2_LEAK_DIAG")) {
        static long g_leakTotal = 0, g_resultTotal = 0;
        std::vector<unsigned> hDel(g->capacity);
        gpuErrchk(cudaMemcpyAsync(hDel.data(), g->d_deleted,
                                  (size_t)g->capacity * sizeof(unsigned),
                                  cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        long leaks = 0, results = 0;
        for (unsigned q = 0; q < bs; q++)
            for (unsigned i = 0; i < k; i++) {
                unsigned id = sb->h_topkIds[(size_t)q * k + i];
                if (id == 0xFFFFFFFFu) continue;
                results++;
                if (id < g->capacity && hDel[id]) leaks++;
            }
        g_leakTotal += leaks; g_resultTotal += results;
        printf("[LEAK-DIAG] batch: %ld/%ld deleted-in-topk  |  cumulative: %ld/%ld (%.4f%%)\n",
               leaks, results, g_leakTotal, g_resultTotal,
               g_resultTotal ? 100.0 * g_leakTotal / g_resultTotal : 0.0);
    }
    gpuErrchk(cudaGetLastError());
}

} // namespace fda2
