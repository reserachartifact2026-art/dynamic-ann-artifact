#include "fda2_insert.h"
#include "fda2_prune.cuh"
#include "fda2_locks.cuh"   // atomicCAS_u8 for the dirty-list dedup mark
#include <cstring>
#include <cstdio>

extern int g_greedyTag;   // DEBUG: defined (global scope) in baseline_src/greedySearch.cu

namespace fda2 {

// ============================================================================
// Buffers
// ============================================================================
void allocateInsertBuffers(InsertBuffers* b, unsigned maxBatch) {
    b->maxBatch = maxBatch;
    gpuErrchk(cudaMallocHost(&b->h_vectors,  (size_t)maxBatch * D * sizeof(GVEC)));
    gpuErrchk(cudaMallocHost(&b->h_pointIds, (size_t)maxBatch * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_vectors,  (size_t)maxBatch * D * sizeof(GVEC)));
    gpuErrchk(cudaMalloc(&b->d_pointIds, (size_t)maxBatch * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_visitedSets,
                          (size_t)maxBatch * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_visitedCounts, (size_t)maxBatch * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_revSrc, (size_t)maxBatch * R * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_revDst, (size_t)maxBatch * R * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_revCount, sizeof(unsigned)));
    allocateGreedySearchBuffers(&b->gsBuffers, maxBatch);
    b->allocated = true;
}

void freeInsertBuffers(InsertBuffers* b) {
    if (!b->allocated) return;
    cudaFreeHost(b->h_vectors);
    cudaFreeHost(b->h_pointIds);
    cudaFree(b->d_vectors);
    cudaFree(b->d_pointIds);
    cudaFree(b->d_visitedSets);
    cudaFree(b->d_visitedCounts);
    cudaFree(b->d_revSrc);
    cudaFree(b->d_revDst);
    cudaFree(b->d_revCount);
    freeGreedySearchBuffers(&b->gsBuffers);
    b->allocated = false;
}

// ============================================================================
// Kernel: write each new node's vector into its slot, zero its degree.
// ============================================================================
__global__ void writeNewVectorsKernel(uint8_t* d_graph,
                                       const GVEC*     d_vectors,
                                       const unsigned* d_pointIds,
                                       unsigned batchSize)
{
    unsigned i = blockIdx.x;
    if (i >= batchSize) return;
    unsigned slot = d_pointIds[i];
    GVEC* dst = (GVEC*)(d_graph + (size_t)slot * graphEntrySize);
    const GVEC* src = d_vectors + (size_t)i * D;
    for (unsigned d = threadIdx.x; d < D; d += blockDim.x) dst[d] = src[d];
    if (threadIdx.x == 0) {
        unsigned* deg = (unsigned*)(d_graph + (size_t)slot * graphEntrySize
                                     + D*sizeof(GVEC));
        *deg = 0;
    }
}

// ============================================================================
// Step 1+2: forward-edge selection. One block per new node.
// Gather the greedy visited set → sort by distance → single common RobustPrune
// (blockAlphaRNG) → write the new node's R forward neighbors.
// ============================================================================
__global__ void robustPruneForwardKernel(
    uint8_t*        d_graph,
    const unsigned* d_deleted,
    const GVEC*     d_newVectors,     // [B × D]
    const unsigned* d_visitedSets,    // [B × MAX_PARENTS_PERQUERY]
    const unsigned* d_visitedCounts,  // [B]
    const unsigned* d_pointIds,       // [B]
    unsigned B, float alpha)
{
    unsigned i = blockIdx.x;
    if (i >= B) return;

    unsigned slot = d_pointIds[i];
    const GVEC* myVec = d_newVectors + (size_t)i * D;
    unsigned visited = d_visitedCounts[i];
    if (visited > MAX_PARENTS_PERQUERY) visited = MAX_PARENTS_PERQUERY;

    __shared__ float    anchorVec[D];
    __shared__ unsigned cands[MAX_PARENTS_PERQUERY];
    __shared__ float    dists[MAX_PARENTS_PERQUERY];
    __shared__ unsigned nCands;
    __shared__ unsigned selected[R];
    __shared__ unsigned nSel;
    __shared__ bool     reject;

    if (threadIdx.x == 0) { nCands = 0; nSel = 0; }
    // Cache the anchor (new node) vector in shared mem for the BANG distances.
    for (unsigned d = threadIdx.x; d < D; d += blockDim.x) anchorVec[d] = myVec[d];
    __syncthreads();

    // Collect visited set (drop self, sentinels, deleted).
    for (unsigned ix = threadIdx.x; ix < visited; ix += blockDim.x) {
        unsigned c = d_visitedSets[(size_t)i * MAX_PARENTS_PERQUERY + ix];
        if (c == slot || c == 0xFFFFFFFFu) continue;
        if (d_deleted && d_deleted[c]) continue;
        unsigned s = atomicAdd(&nCands, 1u);
        if (s < MAX_PARENTS_PERQUERY) cands[s] = c;
    }
    __syncthreads();
    unsigned total = nCands;
    if (total > MAX_PARENTS_PERQUERY) total = MAX_PARENTS_PERQUERY;

    // Single common RobustPrune (BANG exact-distance style: dists + sort + α-RNG).
    robustPruneBANG(d_graph, anchorVec, cands, dists, total, alpha, R,
                    selected, &nSel, &reject);

    // Write forward adjacency.
    unsigned* degPtr = (unsigned*)(d_graph + (size_t)slot * graphEntrySize
                                    + D*sizeof(GVEC));
    unsigned* nbrPtr = degPtr + 1;
    if (threadIdx.x == 0) *degPtr = nSel;
    __syncthreads();
    for (unsigned ix = threadIdx.x; ix < nSel; ix += blockDim.x) nbrPtr[ix] = selected[ix];
}

// ============================================================================
// Step 3a: collect (target = NN, source = new node) reverse-edge pairs from
// each new node's freshly-written forward adjacency.
// ============================================================================
__global__ void collectReverseEdgesKernel(uint8_t* d_graph,
                                          const unsigned* d_pointIds,
                                          unsigned batchSize,
                                          unsigned* d_revSrc,
                                          unsigned* d_revDst,
                                          unsigned* d_revCount)
{
    unsigned i = blockIdx.x;
    if (i >= batchSize) return;
    unsigned newId = d_pointIds[i];
    unsigned* degPtr = (unsigned*)(d_graph + (size_t)newId * graphEntrySize
                                    + D*sizeof(GVEC));
    unsigned* nbrPtr = degPtr + 1;
    unsigned degree = *degPtr;
    if (threadIdx.x < degree) {
        unsigned target = nbrPtr[threadIdx.x];
        unsigned idx = atomicAdd(d_revCount, 1u);
        d_revSrc[idx] = target;     // NN (reverse-edge target)
        d_revDst[idx] = newId;      // new node (reverse-edge source)
    }
}

// ============================================================================
// Step 3b: atomic append of reverse-edge candidates into the P-scratch.
// One thread per (target, source) pair. "Multiple threads will append entries
// in a synchronized manner using atomic operations."
//   slot = atomicAdd(addScratchCount[target]); if slot < P_EXTRA write source.
// Each target that receives ≥1 candidate is added to the dirty list exactly
// once (atomicCAS on the dedup mark) so the prune pass only touches it once.
// ============================================================================
__global__ void appendReverseCandsKernel(
    unsigned* d_addScratch,        // [capacity × P_EXTRA]
    unsigned* d_addScratchCount,   // [capacity]
    uint8_t*  d_dirtyMark,      // [capacity]
    unsigned* d_dirtyList,      // [dirtyCap]
    unsigned* d_dirtyCount,
    unsigned  dirtyCap,
    const unsigned* d_revSrc,
    const unsigned* d_revDst,
    const unsigned* d_revCount)
{
    unsigned pairIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pairIdx >= *d_revCount) return;

    unsigned target = d_revSrc[pairIdx];
    unsigned source = d_revDst[pairIdx];
    if (target == source) return;

    unsigned slot = atomicAdd(&d_addScratchCount[target], 1u);
    if (slot < P_EXTRA) {
        d_addScratch[(size_t)target * P_EXTRA + slot] = source;
    } else {
        atomicSub(&d_addScratchCount[target], 1u);   // overflow: drop this candidate
    }

    // Register target in the dirty list once.
    uint8_t prev = atomicCAS_u8(&d_dirtyMark[target], 0u, 1u);
    if (prev == 0u) {
        unsigned di = atomicAdd(d_dirtyCount, 1u);
        if (di < dirtyCap) d_dirtyList[di] = target;
    }
}

// ============================================================================
// Step 4: prune affected adjacency lists back to R. One block per dirty node u.
//   candidate set = u's existing ≤R neighbors  ∪  u's ≤P scratch candidates
//   if |candidates| ≤ R → keep all (no prune)
//   else                → sort by dist(u, c) ascending, single common
//                         RobustPrune → R
// Resets addScratchCount[u] and dirtyMark[u].
// ============================================================================
__global__ void pruneAddScratchKernel(
    uint8_t*  d_graph,
    const unsigned* d_deleted,
    unsigned* d_addScratch,
    unsigned* d_addScratchCount,
    uint8_t*  d_dirtyMark,
    const unsigned* d_dirtyList,
    const unsigned* d_dirtyCount,
    float alpha)
{
    unsigned b = blockIdx.x;
    if (b >= *d_dirtyCount) return;
    unsigned u = d_dirtyList[b];

    // Defensive: a deleted node needs no maintenance. Just reset scratch.
    if (d_deleted && d_deleted[u]) {
        if (threadIdx.x == 0) { d_addScratchCount[u] = 0; d_dirtyMark[u] = 0; }
        return;
    }

    uint8_t* entry = d_graph + (size_t)u * graphEntrySize;
    const GVEC* uVec = (const GVEC*)entry;
    unsigned* degPtr = (unsigned*)(entry + D*sizeof(GVEC));
    unsigned* nbrPtr = degPtr + 1;
    unsigned degU = *degPtr;
    if (degU > R) degU = R;

    unsigned scratchN = d_addScratchCount[u];
    if (scratchN > P_EXTRA) scratchN = P_EXTRA;

    constexpr unsigned MAX_C = R + P_EXTRA;
    __shared__ unsigned cands[MAX_C];
    __shared__ float    dists[MAX_C];
    __shared__ unsigned nCands;
    __shared__ unsigned selected[R];
    __shared__ unsigned nSel;
    __shared__ float    anchorVec[D];
    __shared__ bool     reject;

    if (threadIdx.x == 0) nCands = 0;
    __syncthreads();

    // Existing neighbors first (preserve them; they are never deleted-targets
    // because the new node's greedy skipped deleted nodes).
    for (unsigned i = threadIdx.x; i < degU; i += blockDim.x) {
        unsigned s = atomicAdd(&nCands, 1u);
        if (s < MAX_C) cands[s] = nbrPtr[i];
    }
    __syncthreads();
    unsigned baseCount = nCands;
    if (baseCount > MAX_C) baseCount = MAX_C;

    // Append scratch candidates, deduped against existing + each other.
    // Single-thread dedup loop — small (≤ P_EXTRA).
    if (threadIdx.x == 0) {
        for (unsigned i = 0; i < scratchN; i++) {
            unsigned src = d_addScratch[(size_t)u * P_EXTRA + i];
            if (src == u) continue;
            bool dup = false;
            for (unsigned c = 0; c < nCands && c < MAX_C; c++) {
                if (cands[c] == src) { dup = true; break; }
            }
            if (!dup && nCands < MAX_C) { cands[nCands] = src; nCands++; }
        }
    }
    __syncthreads();
    unsigned total = nCands;
    if (total > MAX_C) total = MAX_C;

    if (total <= R) {
        // Keep all — no prune needed (adjacency did not grow beyond R).
        if (threadIdx.x == 0) *degPtr = total;
        __syncthreads();
        for (unsigned i = threadIdx.x; i < total; i += blockDim.x) nbrPtr[i] = cands[i];
        if (threadIdx.x == 0) { d_addScratchCount[u] = 0; d_dirtyMark[u] = 0; }
        return;
    }

    // Cache u's vector, then single common RobustPrune → R (BANG style).
    for (unsigned d = threadIdx.x; d < D; d += blockDim.x) anchorVec[d] = uVec[d];
    __syncthreads();
    robustPruneBANG(d_graph, anchorVec, cands, dists, total, alpha, R,
                    selected, &nSel, &reject);

    if (threadIdx.x == 0) *degPtr = nSel;
    __syncthreads();
    for (unsigned i = threadIdx.x; i < nSel; i += blockDim.x) nbrPtr[i] = selected[i];
    if (threadIdx.x == 0) { d_addScratchCount[u] = 0; d_dirtyMark[u] = 0; }
}

// ============================================================================
// Driver
// ============================================================================
unsigned insertBatch(UnifiedGraph* g,
                     const float*    h_newVectors,
                     const unsigned* h_newIds,
                     unsigned        batchSize,
                     unsigned        searchL,
                     float           alpha,
                     InsertBuffers*  bufs,
                     cudaStream_t    stream)
{
    if (batchSize == 0) return 0;

    // Greedy entry point: always the current (live) medoid.
    setRuntimeMedoid(g->medoid, stream);

    // Pack host buffers.
    for (size_t _i=0;_i<(size_t)batchSize*D;_i++) bufs->h_vectors[_i]=(GVEC)h_newVectors[_i];
    for (unsigned i = 0; i < batchSize; i++) bufs->h_pointIds[i] = h_newIds[i];

    gpuErrchk(cudaMemcpyAsync(bufs->d_vectors, bufs->h_vectors,
                              (size_t)batchSize * D * sizeof(GVEC),
                              cudaMemcpyHostToDevice, stream));
    gpuErrchk(cudaMemcpyAsync(bufs->d_pointIds, bufs->h_pointIds,
                              (size_t)batchSize * sizeof(unsigned),
                              cudaMemcpyHostToDevice, stream));

    // Step 0: materialize new nodes (vector + degree 0).
    writeNewVectorsKernel<<<batchSize, BLOCK_D, 0, stream>>>(
        g->d_graph, bufs->d_vectors, bufs->d_pointIds, batchSize);

    // Step 1: greedy search for each new node (lock-free; phased — no concurrent
    // writers during the insert greedy). Skips deleted nodes via g->d_deleted.
    gpuErrchk(cudaMemsetAsync(bufs->d_visitedCounts, 0,
                              (size_t)batchSize * sizeof(unsigned), stream));
    g_greedyTag = 0;   // DEBUG: tag insert greedy
    greedySearchNoSyncPrealloc(
        g->d_graph, bufs->d_vectors,
        bufs->d_visitedSets, bufs->d_visitedCounts,
        /*batchStart*/ 0, batchSize, searchL,
        g->d_deleted, &bufs->gsBuffers, stream);

    // Step 2: forward edges (RobustPrune the visited set → R).
    robustPruneForwardKernel<<<batchSize, BLOCK_D, 0, stream>>>(
        g->d_graph, g->d_deleted, bufs->d_vectors,
        bufs->d_visitedSets, bufs->d_visitedCounts, bufs->d_pointIds,
        batchSize, alpha);

    // Step 3a: collect reverse-edge pairs.
    gpuErrchk(cudaMemsetAsync(bufs->d_revCount, 0, sizeof(unsigned), stream));
    collectReverseEdgesKernel<<<batchSize, R, 0, stream>>>(
        g->d_graph, bufs->d_pointIds, batchSize,
        bufs->d_revSrc, bufs->d_revDst, bufs->d_revCount);

    // Step 3b: atomic append into the P-scratch + build the dirty list.
    gpuErrchk(cudaMemsetAsync(g->d_dirtyCount, 0, sizeof(unsigned), stream));
    {
        unsigned maxPairs = batchSize * R;
        unsigned threads = 256;
        unsigned blocks  = (maxPairs + threads - 1) / threads;
        appendReverseCandsKernel<<<blocks, threads, 0, stream>>>(
            g->d_addScratch, g->d_addScratchCount,
            g->d_dirtyMark, g->d_dirtyList, g->d_dirtyCount, g->dirtyCap,
            bufs->d_revSrc, bufs->d_revDst, bufs->d_revCount);
    }

    // Step 4: prune affected adjacency lists R+P → R (one block per dirty node).
    // dirtyCount ≤ batchSize × R; launch that many blocks (kernel self-guards
    // against b ≥ *d_dirtyCount, so an upper-bound launch is safe).
    {
        unsigned maxDirty = batchSize * R;
        if (maxDirty > g->dirtyCap) maxDirty = g->dirtyCap;
        pruneAddScratchKernel<<<maxDirty, BLOCK_D, 0, stream>>>(
            g->d_graph, g->d_deleted,
            g->d_addScratch, g->d_addScratchCount,
            g->d_dirtyMark, g->d_dirtyList, g->d_dirtyCount,
            alpha);
    }

    gpuErrchk(cudaGetLastError());
    return batchSize;
}

} // namespace fda2
