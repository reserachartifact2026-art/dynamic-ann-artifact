#include "fda_search.h"
#include <cfloat>
#include <climits>
#include <cstdio>
#include <cstdlib>

namespace fda {

// ============================================================================
// Padding diagnostic: count how often the extract/filter kernels exit with
// fewer than k non-deleted entries and have to pad with sentinels. If this
// counter is > 0, the recall is bounded above by the (k - paddedSlots)/k
// fraction across affected queries.
//   d_paddingExtract — increments once per query in extractTopKFromWorklist
//                      when write < k at the end of the worklist walk.
//   d_paddingFilter  — increments once per query in filterDeletesKernel
//                      when write < k after compacting deletes.
// Per-query padded-slot totals (sum of (k-write) across queries) are also
// tracked so we can report avg padded-slots-per-query.
// ============================================================================
__device__ unsigned long long g_paddingExtract        = 0ULL;
__device__ unsigned long long g_paddingExtractSlots   = 0ULL;
__device__ unsigned long long g_paddingFilter         = 0ULL;
__device__ unsigned long long g_paddingFilterSlots    = 0ULL;

// ============================================================================
// SearchBuffers lifecycle
// ============================================================================
void allocateSearchBuffers(SearchBuffers* sb, unsigned qbs) {
    gpuErrchk(cudaMallocHost(&sb->h_topkIds,   (size_t)qbs * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMallocHost(&sb->h_topkDists, (size_t)qbs * MAX_PARENTS_PERQUERY * sizeof(float)));
    gpuErrchk(cudaMallocHost(&sb->h_idxBuf,    (size_t)qbs * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&sb->d_topkIds,        (size_t)qbs * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&sb->d_topkDists,      (size_t)qbs * MAX_PARENTS_PERQUERY * sizeof(float)));
    gpuErrchk(cudaMalloc(&sb->d_queryVecs,      (size_t)qbs * D * sizeof(GVEC)));
    gpuErrchk(cudaMalloc(&sb->d_idxBuf,         (size_t)qbs * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&sb->d_visitedSets,
                          (size_t)qbs * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&sb->d_visitedCounts,  (size_t)qbs * sizeof(unsigned)));
    allocateGreedySearchBuffers(&sb->gsBuffers, qbs);
    sb->allocated = true;
}

void freeSearchBuffers(SearchBuffers* sb) {
    if (!sb->allocated) return;
    cudaFreeHost(sb->h_topkIds);
    cudaFreeHost(sb->h_topkDists);
    cudaFreeHost(sb->h_idxBuf);
    cudaFree(sb->d_topkIds);
    cudaFree(sb->d_topkDists);
    cudaFree(sb->d_queryVecs);
    cudaFree(sb->d_idxBuf);
    cudaFree(sb->d_visitedSets);
    cudaFree(sb->d_visitedCounts);
    freeGreedySearchBuffers(&sb->gsBuffers);
    sb->allocated = false;
}

// ============================================================================
// Extract top-k from the worklist (sorted ascending). Uses searchL as the
// stride between per-query rows — baseline_src greedy lays out the worklist
// with stride = searchL (NOT MAX_L). Reading with the wrong stride scrambles
// queries across the batch.
// ============================================================================
__global__ void extractTopKFromWorklist(
    const unsigned* d_worklist, const float* d_worklistDist,
    const unsigned* d_worklistCount,
    unsigned bs, unsigned k, unsigned searchL,
    unsigned* d_topkIds, float* d_topkDists,
    const unsigned* d_deleted,           // bitvector; nullptr = no filtering
    unsigned        delCapBits)          // size of bitvector in bits
{
    // Note: "Finally, consult the DeletedList to drop the deleted nodes from
    // the final search results." The worklist holds up to searchL candidates
    // sorted ascending by distance. We pick the first k entries whose IDs
    // are NOT in the delete bitvector. This gives k real results instead of
    // (k - #deleted) real results + sentinels.
    //
    // One thread per query (thread 0). Other threads in the block are idle
    // — keeps the launch contract <<<bs, 32>>> identical to the previous
    // implementation while requiring sequential ordering.
    unsigned q = blockIdx.x;
    if (q >= bs) return;
    if (threadIdx.x != 0) return;

    unsigned valid = d_worklistCount[q];
    if (valid > searchL) valid = searchL;

    unsigned write = 0;
    for (unsigned i = 0; i < valid && write < k; i++) {
        unsigned id = d_worklist[(size_t)q * searchL + i];
        if (id == 0xFFFFFFFFu) continue;
        if (d_deleted != nullptr && id < delCapBits) {
            unsigned word = id >> 5;
            unsigned mask = 1u << (id & 31u);
            if ((d_deleted[word] & mask) != 0u) continue;   // deleted → skip
        }
        d_topkIds[(size_t)q * k + write]   = id;
        d_topkDists[(size_t)q * k + write] = d_worklistDist[(size_t)q * searchL + i];
        write++;
    }
    if (write < k) {
        atomicAdd(&g_paddingExtract,      (unsigned long long)1);
        atomicAdd(&g_paddingExtractSlots, (unsigned long long)(k - write));
    }
    for (unsigned i = write; i < k; i++) {
        d_topkIds[(size_t)q * k + i]   = 0xFFFFFFFFu;
        d_topkDists[(size_t)q * k + i] = FLT_MAX;
    }
}

// ============================================================================
// filterDeletesKernel — compact non-deleted to front, pad sentinels.
// One thread per query.
// ============================================================================
__global__ void filterDeletesKernel(
    unsigned* d_topkIds, float* d_topkDists,
    const unsigned* d_deleted,
    unsigned bs, unsigned k)
{
    unsigned q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= bs) return;

    unsigned* ids   = d_topkIds   + (size_t)q * k;
    float*    dists = d_topkDists + (size_t)q * k;

    unsigned write = 0;
    for (unsigned i = 0; i < k; i++) {
        unsigned id = ids[i];
        if (id == 0xFFFFFFFFu) continue;
        unsigned word = id / 32;
        unsigned mask = 1u << (id % 32);
        if (d_deleted[word] & mask) continue;
        if (write != i) {
            ids[write]   = id;
            dists[write] = dists[i];
        }
        write++;
    }
    if (write < k) {
        atomicAdd(&g_paddingFilter,      (unsigned long long)1);
        atomicAdd(&g_paddingFilterSlots, (unsigned long long)(k - write));
    }
    for (unsigned i = write; i < k; i++) {
        ids[i]   = 0xFFFFFFFFu;
        dists[i] = FLT_MAX;
    }
}


// ============================================================================
// Driver
// ============================================================================

// ============================================================================
// Per-index search (§"Search Kernel"):
//   Invoke Search on LTI, then each Temp Index, then RW — one index at a time.
//   Each index uses its own medoid as the greedy entry point.
//   Merge top-k from all indexes (concatenate, sort, take top k).
//   Apply DeleteList filter at the end.
//
// "BANG Exact Distance search" = baseline greedySearchVersionedPrealloc
// (filterNeighbors + computeDists + sortByDistance + mergeIntoWorklist).
//
// "No locks while reading adjacency list" — baseline greedy uses versioned
// reads on d_locks (treats lock as a seqlock-style version), which is the
// non-blocking equivalent of "no locks during search". For our sequential
// workload (no concurrent writes during a search batch), version is always
// even and reads are unblocked.
// ============================================================================

// Helper: run one index's greedy and extract top-k into the provided buffer.
// Caller must call setRuntimeMedoid(medoid, stream) before this and sync the
// medoid update (done in searchUnified between launches).
static void searchOneIndex(
    const UnifiedIndex*  idx,
    const GVEC*          d_queries,
    unsigned             bs,
    unsigned             k,
    unsigned             searchL,
    SearchBuffers*       sb,
    unsigned*            d_outTopkIds,    // [bs × k]
    float*               d_outTopkDists,
    cudaStream_t         stream,
    const unsigned*      d_deleted,
    unsigned             delCapBits)
{
    gpuErrchk(cudaMemsetAsync(sb->d_visitedCounts, 0,
                              (size_t)bs * sizeof(unsigned), stream));
    // SEARCH greedy: no locks, no version checks. greedy passes
    // d_deleted = nullptr because deleted nodes are still navigational
    // entries ("Search can still start from that node"). Filtering
    // happens at extract time.
    greedySearchNoSyncPrealloc(
        idx->d_graph,
        const_cast<GVEC*>(d_queries),
        sb->d_visitedSets,
        sb->d_visitedCounts,
        /*batchStart*/ 0,
        bs, searchL,
        /*d_deleted*/ nullptr,
        &sb->gsBuffers,
        stream);

    extractTopKFromWorklist<<<bs, 32, 0, stream>>>(
        sb->gsBuffers.d_worklist, sb->gsBuffers.d_worklistDist,
        sb->gsBuffers.d_worklistCount,
        bs, k, searchL, d_outTopkIds, d_outTopkDists,
        d_deleted, delCapBits);
}

// Merge (M+2) per-index top-k's into one final top-k.
// Layout: d_concatIds[(M+2) × bs × k], grouped per-query.
// Each query has (M+2) × k candidates; we sort and take the closest k (with dedup).
__global__ void mergeMultiIndexTopK(
    const unsigned* d_concatIds,    // [bs × numIndexes × k]
    const float*    d_concatDists,
    unsigned        numIndexes,
    unsigned        bs, unsigned k,
    unsigned*       d_outIds,
    float*          d_outDists)
{
    unsigned q = blockIdx.x;
    if (q >= bs) return;
    if (threadIdx.x != 0) return;

    constexpr unsigned MAX_TOTAL = 64 * 16;       // up to 16 indexes × 64 k
    unsigned ids[MAX_TOTAL];
    float    dists[MAX_TOTAL];
    unsigned total = numIndexes * k;
    if (total > MAX_TOTAL) total = MAX_TOTAL;

    // Gather all candidates for this query, in per-index stride.
    // Layout: d_concatIds[ (idxI * bs + q) * k + j ] for each idxI in [0, numIndexes), j in [0, k).
    unsigned w = 0;
    for (unsigned ii = 0; ii < numIndexes; ii++) {
        for (unsigned j = 0; j < k; j++) {
            unsigned id  = d_concatIds [(size_t)(ii * bs + q) * k + j];
            float    d   = d_concatDists[(size_t)(ii * bs + q) * k + j];
            if (id == 0xFFFFFFFFu) continue;
            // Dedup against already-gathered
            bool dup = false;
            for (unsigned x = 0; x < w; x++) if (ids[x] == id) { dup = true; break; }
            if (dup) continue;
            if (w < MAX_TOTAL) { ids[w] = id; dists[w] = d; w++; }
        }
    }
    // Insertion sort by dist ascending
    for (unsigned i = 1; i < w; i++) {
        unsigned ci = ids[i];
        float    di = dists[i];
        int j = (int)i - 1;
        while (j >= 0 && dists[j] > di) {
            ids[j+1]   = ids[j];
            dists[j+1] = dists[j];
            j--;
        }
        ids[j+1]   = ci;
        dists[j+1] = di;
    }
    // Write top k (pad with sentinels if fewer than k unique)
    for (unsigned i = 0; i < k; i++) {
        if (i < w) {
            d_outIds  [(size_t)q * k + i] = ids[i];
            d_outDists[(size_t)q * k + i] = dists[i];
        } else {
            d_outIds  [(size_t)q * k + i] = 0xFFFFFFFFu;
            d_outDists[(size_t)q * k + i] = FLT_MAX;
        }
    }
}

void searchUnified(
    const UnifiedIndex*       idx,
    const DeleteListWrapper*  deleteList,
    const GVEC*               d_queries,
    unsigned bs, unsigned k, unsigned searchL,
    SearchBuffers*            sb,
    cudaStream_t              stream)
{
    if (bs == 0) return;

    // Indexes to search: LTI + each active Temp + (current RW if non-empty).
    unsigned numIndexes = 1 + idx->numActiveTemps + (idx->rwCount > 0 ? 1 : 0);
    if (numIndexes > 16) numIndexes = 16;   // mergeMultiIndexTopK's MAX_TOTAL = 16 × k

    // Concat buffer for per-index top-k results.
    // Use the slack space in d_topkIds (allocated qbs × MAX_PARENTS_PERQUERY).
    unsigned* d_concatIds   = sb->d_topkIds   + (size_t)bs * k;
    float*    d_concatDists = sb->d_topkDists + (size_t)bs * k;

    // Cache delete bitvector ptr + bit-capacity once; pass to each per-index
    // extract so deleted IDs are skipped at result-extraction time (Note:
    // "consult the DeletedList to drop the deleted nodes from the final
    // search results").
    const unsigned* d_delBits  = (deleteList && deleteList->count > 0)
                                 ? deleteList->d_bits : nullptr;
    unsigned        delCapBits = (deleteList) ? deleteList->capacity : 0u;

    unsigned ii = 0;
    // LTI
    setRuntimeMedoid(idx->ltiMedoid, stream);
    searchOneIndex(idx, d_queries, bs, k, searchL, sb,
                   d_concatIds  + (size_t)ii * bs * k,
                   d_concatDists + (size_t)ii * bs * k,
                   stream, d_delBits, delCapBits);

    // Spec: "Invoke the kernel Search on LTI, Temp Indexes and finally the RO
    // index", with no extra delta-consultation step — each index is searched
    // independently and the per-index top-k's are merged below.
    ii++;

    // Each sealed Temp Index
    for (unsigned t = 0; t < idx->numActiveTemps && ii < numIndexes; t++) {
        if (idx->tempMedoids[t] == UINT_MAX) continue;
        setRuntimeMedoid(idx->tempMedoids[t], stream);
        searchOneIndex(idx, d_queries, bs, k, searchL, sb,
                       d_concatIds  + (size_t)ii * bs * k,
                       d_concatDists + (size_t)ii * bs * k,
                       stream, d_delBits, delCapBits);
        ii++;
    }

    // Current RW (if non-empty)
    if (idx->rwCount > 0 && ii < numIndexes && idx->rwMedoid != UINT_MAX) {
        setRuntimeMedoid(idx->rwMedoid, stream);
        searchOneIndex(idx, d_queries, bs, k, searchL, sb,
                       d_concatIds  + (size_t)ii * bs * k,
                       d_concatDists + (size_t)ii * bs * k,
                       stream, d_delBits, delCapBits);
        ii++;
    }

    unsigned actualNumIndexes = ii;

    // Merge per-index top-k's → final top-k
    mergeMultiIndexTopK<<<bs, 32, 0, stream>>>(
        d_concatIds, d_concatDists,
        actualNumIndexes, bs, k,
        sb->d_topkIds, sb->d_topkDists);

    // Don't pad. The per-index extractTopKFromWorklist already filters
    // deleted IDs from the searchL-sized worklist (much larger than k),
    // so the merged top-k from mergeMultiIndexTopK is already non-deleted
    // by construction. filterDeletesKernel was running over only the top-k
    // (k = 10 entries) and padding with sentinels when any were deleted.
    // Removed.

    // Restore runtime medoid to LTI for the next insert pipeline
    setRuntimeMedoid(idx->ltiMedoid, stream);
}

// Read padding-hit counters and print summary. Call this once after the
// workload is done (before exit). Padding diagnostic for whether queries
// run out of non-deleted candidates and trigger sentinel padding.
void reportPaddingDiagnostic(unsigned totalQueries)
{
    unsigned long long hExtract = 0, hExtractSlots = 0;
    unsigned long long hFilter  = 0, hFilterSlots  = 0;
    cudaMemcpyFromSymbol(&hExtract,      g_paddingExtract,      sizeof(hExtract));
    cudaMemcpyFromSymbol(&hExtractSlots, g_paddingExtractSlots, sizeof(hExtractSlots));
    cudaMemcpyFromSymbol(&hFilter,       g_paddingFilter,       sizeof(hFilter));
    cudaMemcpyFromSymbol(&hFilterSlots,  g_paddingFilterSlots,  sizeof(hFilterSlots));

    printf("\n=== Padding Diagnostic ===\n");
    printf("extractTopKFromWorklist: %llu padded calls / %u queries (%.2f%%), "
           "%llu padded slots (avg %.2f/call)\n",
           hExtract, totalQueries,
           totalQueries > 0 ? 100.0 * (double)hExtract / (double)totalQueries : 0.0,
           hExtractSlots,
           hExtract > 0 ? (double)hExtractSlots / (double)hExtract : 0.0);
    printf("filterDeletesKernel:     %llu padded calls / %u queries (%.2f%%), "
           "%llu padded slots (avg %.2f/call)\n",
           hFilter, totalQueries,
           totalQueries > 0 ? 100.0 * (double)hFilter / (double)totalQueries : 0.0,
           hFilterSlots,
           hFilter > 0 ? (double)hFilterSlots / (double)hFilter : 0.0);
    fflush(stdout);
}

} // namespace fda
