#include "fda_insert.h"
#include "fda_locks.cuh"
#include <cfloat>
#include <cstdio>
#include <cstring>

namespace fda {

// ============================================================================
// Per-stage insert-phase timing (accumulated across all insertBatchUnified
// calls; printed once by dumpInsertTimings).
// ============================================================================
struct InsertTimingAccumulator {
    double writeMs    = 0.0;   // writeNewVectorsAndZeroDegree
    double greedyMs   = 0.0;   // greedySearchLockedPrealloc
    double pruneMs    = 0.0;   // robustPruneBatchKernel
    double collectMs  = 0.0;   // collectReverseEdgesKernel
    double appendMs   = 0.0;   // appendReverseEdgesRoutedKernel
    unsigned long batches      = 0;
    unsigned long totalInserts = 0;
};
static InsertTimingAccumulator g_insertTimings;

void dumpInsertTimings() {
    if (g_insertTimings.batches == 0) return;
    double total = g_insertTimings.writeMs + g_insertTimings.greedyMs +
                   g_insertTimings.pruneMs + g_insertTimings.collectMs +
                   g_insertTimings.appendMs;
    if (total <= 0) total = 1.0;
    auto pct = [&](double x){ return 100.0 * x / total; };
    printf("\n=== Insert per-stage breakdown (over %lu batches, %lu vertices) ===\n",
           g_insertTimings.batches, g_insertTimings.totalInserts);
    printf("  writeNewVectors      %9.1f ms (%5.1f%%)\n",
           g_insertTimings.writeMs, pct(g_insertTimings.writeMs));
    printf("  greedySearchLocked   %9.1f ms (%5.1f%%)\n",
           g_insertTimings.greedyMs, pct(g_insertTimings.greedyMs));
    printf("  robustPruneBatch     %9.1f ms (%5.1f%%)\n",
           g_insertTimings.pruneMs, pct(g_insertTimings.pruneMs));
    printf("  collectReverseEdges  %9.1f ms (%5.1f%%)\n",
           g_insertTimings.collectMs, pct(g_insertTimings.collectMs));
    printf("  appendReverseEdges   %9.1f ms (%5.1f%%)\n",
           g_insertTimings.appendMs, pct(g_insertTimings.appendMs));
    printf("  ──────────────────────────────────────\n");
    printf("  Sum                  %9.1f ms\n", total);
    fflush(stdout);
}

// ============================================================================
// Buffer allocation — unchanged shape from previous impl
// ============================================================================
void allocateInsertBatchBuffers(InsertBatchBuffers* b, unsigned maxBatch) {
    b->maxBatch = maxBatch;
    gpuErrchk(cudaMallocHost(&b->h_vectors,  (size_t)maxBatch * D * sizeof(GVEC)));
    gpuErrchk(cudaMallocHost(&b->h_pointIds, (size_t)maxBatch * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_vectors,       (size_t)maxBatch * D * sizeof(GVEC)));
    gpuErrchk(cudaMalloc(&b->d_pointIds,      (size_t)maxBatch * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_visitedSets,
                          (size_t)maxBatch * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_visitedSetCounts,
                          (size_t)maxBatch * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_revEdgeSrc, (size_t)maxBatch * R * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_revEdgeDst, (size_t)maxBatch * R * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_revEdgeCount, sizeof(unsigned)));
    allocateGreedySearchBuffers(&b->gsBuffers,    maxBatch);
    allocateOutNeighborsBuffers(&b->outBuffers,   maxBatch);
    b->allocated = true;
}

void freeInsertBatchBuffers(InsertBatchBuffers* b) {
    if (!b->allocated) return;
    cudaFreeHost(b->h_vectors);
    cudaFreeHost(b->h_pointIds);
    cudaFree(b->d_vectors);
    cudaFree(b->d_pointIds);
    cudaFree(b->d_visitedSets);
    cudaFree(b->d_visitedSetCounts);
    cudaFree(b->d_revEdgeSrc);
    cudaFree(b->d_revEdgeDst);
    cudaFree(b->d_revEdgeCount);
    freeGreedySearchBuffers(&b->gsBuffers);
    freeOutNeighborsBuffers(&b->outBuffers);
    b->allocated = false;
}

// ============================================================================
// Helper kernel: write a new vertex's vector into its unified-graph slot and
// zero its degree. The neighbor list is overwritten later by
// computeOutNeighborsPrealloc.
// ============================================================================
__global__ void writeNewVectorsAndZeroDegree(uint8_t* d_graph,
                                              const GVEC*     d_vectors,
                                              const unsigned* d_pointIds,
                                              unsigned        batchSize)
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
// Collect (target, source) pairs from each new vertex's forward edges.
//
// Lock-free by design — relies on intra-stream ordering:
//   - The "own" new vertex's adj list was just written by robustPruneBatchKernel
//     on the SAME stream, so the prior kernel completes before we read here.
//   - We only READ from the own new vertex (never from another concurrently-
//     mutating slot), so there's no conflicting writer.
//   - The reverse-edge consumer (appendReverseEdgesRoutedKernel) writes to
//     OTHER vertices' adj lists and IS lock-protected; it runs after us on
//     the same stream.
//
// IMPORTANT: this kernel is safe ONLY because all insert pipeline work runs
// on a single stream (opStream). If insert ever fans out to multiple streams,
// readers of new-vertex adj lists must take the per-vertex lock too.
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
        unsigned idx    = atomicAdd(d_revCount, 1u);
        d_revSrc[idx]   = target;
        d_revDst[idx]   = newId;
    }
}

// ============================================================================
// Distance helpers (serial + block-parallel)
// ============================================================================
template<typename TA, typename TB>
__device__ __forceinline__
static float l2Dist(const TA* a, const TB* b) {
    float d = 0.0f;
    for (unsigned i = 0; i < D; i++) {
        float diff = a[i] - b[i];
        d += diff * diff;
    }
    return d;
}

// Block-parallel l2: all threads collaborate on one dist computation. Each
// thread sums its slice of D, warp-reduces, then a thread-0 cross-warp sum.
// Must be called by ALL threads in the block (uses __syncthreads).
template<typename TA, typename TB>
__device__ __forceinline__
static float l2DistBlock(const TA* a, const TB* b, float* smemAccum)
{
    float partial = 0.0f;
    for (unsigned i = threadIdx.x; i < D; i += blockDim.x) {
        float diff = a[i] - b[i];
        partial += diff * diff;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    if ((threadIdx.x & 31u) == 0) {
        smemAccum[threadIdx.x >> 5] = partial;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.0f;
        unsigned nWarps = (blockDim.x + 31) / 32;
        for (unsigned w = 0; w < nWarps; w++) total += smemAccum[w];
        smemAccum[0] = total;
    }
    __syncthreads();
    return smemAccum[0];
}

__device__ __forceinline__
static void inlineAlphaRNGOverflow(uint8_t* d_graph,
                                   unsigned target,
                                   unsigned newSrc,
                                   float    alpha)
{
    GVEC* tVec = (GVEC*)(d_graph + (size_t)target * graphEntrySize);
    unsigned* degPtr = (unsigned*)(d_graph + (size_t)target * graphEntrySize
                                    + D*sizeof(GVEC));
    unsigned* nbrPtr = degPtr + 1;
    unsigned currDeg = *degPtr;

    unsigned cands[R + 1];
    float    dists[R + 1];
    bool     dup = false;
    for (unsigned i = 0; i < currDeg && i < R; i++) {
        cands[i] = nbrPtr[i];
        if (cands[i] == newSrc) dup = true;
        GVEC* nv = (GVEC*)(d_graph + (size_t)cands[i] * graphEntrySize);
        dists[i] = l2Dist(tVec, nv);
    }
    unsigned total = currDeg;
    if (!dup && newSrc != target) {
        cands[total] = newSrc;
        GVEC* nv = (GVEC*)(d_graph + (size_t)newSrc * graphEntrySize);
        dists[total] = l2Dist(tVec, nv);
        total++;
    }

    for (unsigned i = 1; i < total; i++) {
        unsigned ci = cands[i];
        float    di = dists[i];
        int j = (int)i - 1;
        while (j >= 0 && dists[j] > di) {
            cands[j+1] = cands[j];
            dists[j+1] = dists[j];
            j--;
        }
        cands[j+1] = ci;
        dists[j+1] = di;
    }

    unsigned selected[R];
    unsigned numSel = 0;
    for (unsigned i = 0; i < total && numSel < R; i++) {
        unsigned c = cands[i];
        float    distTC = dists[i];
        bool reject = false;
        for (unsigned j = 0; j < numSel; j++) {
            unsigned s = selected[j];
            GVEC* sv = (GVEC*)(d_graph + (size_t)s * graphEntrySize);
            GVEC* cv = (GVEC*)(d_graph + (size_t)c * graphEntrySize);
            float dsc = l2Dist(sv, cv);
            if (alpha * dsc <= distTC) { reject = true; break; }
        }
        if (!reject) selected[numSel++] = c;
    }

    for (unsigned i = 0; i < numSel; i++) nbrPtr[i] = selected[i];
    *degPtr = numSel;
}

// ============================================================================
// REVERSE-EDGE KERNEL. One BLOCK per (target, source) pair.
// Thread 0 acquires the per-node lock; all threads cooperate on the α-RNG
// distance computations inside the locked region (block-cooperative l2 is
// ~30× faster per call than single-thread serial l2, reducing lock hold time
// on hub vertices). Every back-edge goes through the locked append + α-RNG on
// overflow (spec §"Before reading the adjacency list of a node, use the lock.").
// ============================================================================
__global__ void appendReverseEdgesRoutedKernel(
    uint8_t*  d_graph,
    uint8_t*  d_locks,
    unsigned* d_revSrc,
    unsigned* d_revDst,
    unsigned* d_revCount,
    float     alpha)
{
    unsigned pairIdx = blockIdx.x;
    unsigned count = *d_revCount;
    if (pairIdx >= count) return;

    unsigned target = d_revSrc[pairIdx];
    unsigned source = d_revDst[pairIdx];
    if (target == source) return;

    if (threadIdx.x == 0) acquireNodeLock(d_locks, target);
    __syncthreads();

    unsigned* degPtr = (unsigned*)(d_graph + (size_t)target * graphEntrySize
                                    + D*sizeof(GVEC));
    unsigned* nbrPtr = degPtr + 1;

    __shared__ unsigned currDeg;
    __shared__ bool     already;
    if (threadIdx.x == 0) {
        currDeg = *degPtr;
        already = false;
        for (unsigned i = 0; i < currDeg; i++) {
            if (nbrPtr[i] == source) { already = true; break; }
        }
    }
    __syncthreads();

    if (already) {
        if (threadIdx.x == 0) releaseNodeLock(d_locks, target);
        return;
    }

    if (currDeg < R) {
        // Simple append — thread 0 does it
        if (threadIdx.x == 0) {
            nbrPtr[currDeg] = source;
            *degPtr = currDeg + 1;
            releaseNodeLock(d_locks, target);
        }
        return;
    }

    // ── Overflow: block-cooperative α-RNG ──────────────────────────────────
    // Candidate set = existing R + new source = R+1 candidates.
    __shared__ unsigned cands[R + 1];
    __shared__ float    dists[R + 1];
    __shared__ unsigned selected[R];
    __shared__ unsigned nSel;
    __shared__ float    smemAccum[8];
    __shared__ bool     reject;
    __shared__ unsigned cCurr;

    if (threadIdx.x == 0) {
        nSel = 0;
        for (unsigned i = 0; i < R; i++) cands[i] = nbrPtr[i];
        cands[R] = source;
    }
    __syncthreads();
    unsigned total = R + 1;

    GVEC* tVec = (GVEC*)(d_graph + (size_t)target * graphEntrySize);

    // Compute dist(target, c) for each candidate using cooperative l2
    for (unsigned i = 0; i < total; i++) {
        unsigned c = cands[i];
        GVEC* cVec = (GVEC*)(d_graph + (size_t)c * graphEntrySize);
        float d = l2DistBlock(tVec, cVec, smemAccum);
        if (threadIdx.x == 0) dists[i] = d;
    }
    __syncthreads();

    // Insertion sort by dist ascending (thread 0, total = R+1 = 65)
    if (threadIdx.x == 0) {
        for (unsigned i = 1; i < total; i++) {
            unsigned cid = cands[i];
            float    cd  = dists[i];
            int j = (int)i - 1;
            while (j >= 0 && dists[j] > cd) {
                cands[j+1] = cands[j];
                dists[j+1] = dists[j];
                j--;
            }
            cands[j+1] = cid;
            dists[j+1] = cd;
        }
    }
    __syncthreads();

    // α-RNG with cooperative dist for each (s, c) pair
    for (unsigned i = 0; i < total && nSel < R; i++) {
        if (threadIdx.x == 0) { cCurr = cands[i]; reject = false; }
        __syncthreads();

        float distTC = dists[i];
        unsigned c = cCurr;
        GVEC* cVec = (GVEC*)(d_graph + (size_t)c * graphEntrySize);

        unsigned currNSel = nSel;
        for (unsigned j = 0; j < currNSel; j++) {
            if (reject) { __syncthreads(); break; }
            unsigned s = selected[j];
            GVEC* sVec = (GVEC*)(d_graph + (size_t)s * graphEntrySize);
            float dsc = l2DistBlock(sVec, cVec, smemAccum);
            if (threadIdx.x == 0) {
                if (alpha * dsc <= distTC) reject = true;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && !reject) {
            selected[nSel] = c;
            nSel++;
        }
        __syncthreads();
    }

    // Write back the new neighbor list + release lock (thread 0 finalizes)
    if (threadIdx.x == 0) *degPtr = nSel;
    __syncthreads();
    for (unsigned i = threadIdx.x; i < nSel; i += blockDim.x) nbrPtr[i] = selected[i];
    __syncthreads();

    if (threadIdx.x == 0) releaseNodeLock(d_locks, target);
}
// ============================================================================
// Batched RobustPrune: one block per new vertex. Each block runs the
// candidate-sort + α-RNG on its assigned vertex's visited set and writes the
// neighbor list to d_graph[d_pointIds[i]]. Replaces the previous loop of
// 256 single-vertex computeOutNeighborsPrealloc calls (1280 kernel launches
// per insert batch → 1 kernel launch per insert batch).
// ============================================================================
__global__ void robustPruneBatchKernel(
    uint8_t*        d_graph,
    const GVEC*     d_newVectors,        // [B x D]
    const unsigned* d_visitedSets,       // [B × MAX_PARENTS_PERQUERY]
    const unsigned* d_visitedCounts,     // [B]
    const unsigned* d_pointIds,          // [B] — target slots
    unsigned B, float alpha)
{
    unsigned i = blockIdx.x;
    if (i >= B) return;

    unsigned slot = d_pointIds[i];
    const GVEC* myVec = d_newVectors + (size_t)i * D;
    unsigned visitedCount = d_visitedCounts[i];
    if (visitedCount > MAX_PARENTS_PERQUERY) visitedCount = MAX_PARENTS_PERQUERY;

    __shared__ unsigned cands[MAX_PARENTS_PERQUERY];
    __shared__ float    cDists[MAX_PARENTS_PERQUERY];
    __shared__ unsigned nCands;
    __shared__ unsigned selected[R];
    __shared__ unsigned nSel;
    __shared__ float    smemAccum[8];
    __shared__ bool     reject;
    __shared__ unsigned cCurr;

    if (threadIdx.x == 0) { nCands = 0; nSel = 0; }
    __syncthreads();

    // Load visited set into cands (dedup against self, drop sentinels)
    for (unsigned ix = threadIdx.x; ix < visitedCount; ix += blockDim.x) {
        unsigned c = d_visitedSets[(size_t)i * MAX_PARENTS_PERQUERY + ix];
        if (c == slot || c == 0xFFFFFFFFu) continue;
        unsigned slot_local = atomicAdd(&nCands, 1u);
        if (slot_local < MAX_PARENTS_PERQUERY) cands[slot_local] = c;
    }
    __syncthreads();
    unsigned total = nCands;
    if (total > MAX_PARENTS_PERQUERY) total = MAX_PARENTS_PERQUERY;

    // Cooperative dist computation for each candidate
    for (unsigned ix = 0; ix < total; ix++) {
        unsigned c = cands[ix];
        const GVEC* cVec = (const GVEC*)(d_graph + (size_t)c * graphEntrySize);
        float d = l2DistBlock(myVec, cVec, smemAccum);
        if (threadIdx.x == 0) cDists[ix] = d;
    }
    __syncthreads();

    // Sort ascending by dist (thread 0; total ≤ 512)
    if (threadIdx.x == 0) {
        for (unsigned ix = 1; ix < total; ix++) {
            unsigned cid = cands[ix];
            float    cd  = cDists[ix];
            int j = (int)ix - 1;
            while (j >= 0 && cDists[j] > cd) {
                cands[j+1] = cands[j];
                cDists[j+1] = cDists[j];
                j--;
            }
            cands[j+1] = cid;
            cDists[j+1] = cd;
        }
    }
    __syncthreads();

    // α-RNG with cooperative dist for each (s, c) pair
    for (unsigned ix = 0; ix < total && nSel < R; ix++) {
        if (threadIdx.x == 0) { cCurr = cands[ix]; reject = false; }
        __syncthreads();

        float distVC = cDists[ix];
        unsigned c = cCurr;
        const GVEC* cVec = (const GVEC*)(d_graph + (size_t)c * graphEntrySize);

        unsigned currNSel = nSel;
        for (unsigned jx = 0; jx < currNSel; jx++) {
            if (reject) { __syncthreads(); break; }
            unsigned s = selected[jx];
            const GVEC* sVec = (const GVEC*)(d_graph + (size_t)s * graphEntrySize);
            float dsc = l2DistBlock(sVec, cVec, smemAccum);
            if (threadIdx.x == 0) {
                if (alpha * dsc <= distVC) reject = true;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && !reject) {
            selected[nSel] = c;
            nSel++;
        }
        __syncthreads();
    }

    // Write neighbor list back to d_graph[slot]
    unsigned* degPtr = (unsigned*)(d_graph + (size_t)slot * graphEntrySize
                                    + D*sizeof(GVEC));
    unsigned* nbrPtr = degPtr + 1;
    if (threadIdx.x == 0) *degPtr = nSel;
    __syncthreads();
    for (unsigned ix = threadIdx.x; ix < nSel; ix += blockDim.x) nbrPtr[ix] = selected[ix];
}

// ============================================================================
// Driver: full insert batch into the unified index.
// ============================================================================
unsigned insertBatchUnified(UnifiedIndex* idx,
                            const float*    h_newVectors,
                            const unsigned* h_newIds,
                            unsigned        batchSize,
                            unsigned        searchL,
                            float           alpha,
                            InsertBatchBuffers* bufs,
                            unsigned*       d_deleted,
                            cudaStream_t    stream,
                            bool            d_vectorsPreLoaded)
{
    if (batchSize == 0) return 0;

    // Slot range validation: regular workload inserts land in the current RW
    // range [rwBase, rwBase+T). Phase B re-insertion uses existing slots in the
    // immutable region (already covered by caller's bookkeeping), so we skip
    // the bounds check in that case.
    unsigned maxId = 0;
    if (!d_vectorsPreLoaded) {
        for (unsigned i = 0; i < batchSize; i++) {
            if (h_newIds[i] < idx->rwBase) {
                fprintf(stderr, "[insertBatchUnified] slot %u < rwBase %u — caller bug\n",
                        h_newIds[i], idx->rwBase);
                return 0;
            }
            if (h_newIds[i] >= idx->rwBase + idx->tempCapacity) {
                fprintf(stderr, "[insertBatchUnified] slot %u >= rwBase+T %u — RW overflow\n",
                        h_newIds[i], idx->rwBase + idx->tempCapacity);
                return 0;
            }
            if (h_newIds[i] > maxId) maxId = h_newIds[i];
        }
    } else {
        for (unsigned i = 0; i < batchSize; i++)
            if (h_newIds[i] > maxId) maxId = h_newIds[i];
    }

    // Step 1: pack pointIds into pinned host; pack vectors only when caller
    // hasn't pre-loaded bufs->d_vectors (Phase B path skips the vector copy
    // because vectors are read D→D from existing graph slots).
    for (unsigned i = 0; i < batchSize; i++)
        bufs->h_pointIds[i] = h_newIds[i];
    if (!d_vectorsPreLoaded) {
        for (size_t _i=0;_i<(size_t)batchSize*D;_i++) bufs->h_vectors[_i]=(GVEC)h_newVectors[_i];
    }

    // Step 2: H→D
    if (!d_vectorsPreLoaded) {
        gpuErrchk(cudaMemcpyAsync(bufs->d_vectors, bufs->h_vectors,
                                  (size_t)batchSize * D * sizeof(GVEC),
                                  cudaMemcpyHostToDevice, stream));
    }
    gpuErrchk(cudaMemcpyAsync(bufs->d_pointIds, bufs->h_pointIds,
                              (size_t)batchSize * sizeof(unsigned),
                              cudaMemcpyHostToDevice, stream));

    // Build a view of the live index and dispatch to the View-targeted impl.
    // (For Phase B, the caller invokes insertBatchToView directly with the
    // delta-memory view, bypassing this validation path.)
    GraphView view = makeViewFromIndex(idx);

    // Steps 3-7: kernel pipeline against the view (same code path as Phase B).
    // Per-stage CUDA event timing is commented out (forces a per-batch sync
    // that prevents host from queueing kernels ahead, ~2× slowdown on the
    // full workload). Re-enable to re-measure the per-stage breakdown.
    // static cudaEvent_t evt_a, evt_b, evt_c, evt_d, evt_e, evt_f;
    // static bool evt_init = false;
    // if (!evt_init) {
    //     cudaEventCreate(&evt_a); cudaEventCreate(&evt_b);
    //     cudaEventCreate(&evt_c); cudaEventCreate(&evt_d);
    //     cudaEventCreate(&evt_e); cudaEventCreate(&evt_f);
    //     evt_init = true;
    // }

    // cudaEventRecord(evt_a, stream);
    writeNewVectorsAndZeroDegree<<<batchSize, BLOCK_D, 0, stream>>>(
        view.d_graph, bufs->d_vectors, bufs->d_pointIds, batchSize);
    // cudaEventRecord(evt_b, stream);

    gpuErrchk(cudaMemsetAsync(bufs->d_visitedSetCounts, 0,
                              (size_t)batchSize * sizeof(unsigned), stream));
    greedySearchLockedPrealloc(
        view.d_graph,
        view.d_locks,
        bufs->d_vectors,
        bufs->d_visitedSets,
        bufs->d_visitedSetCounts,
        /*batchStart*/ 0,
        batchSize, searchL,
        d_deleted,
        &bufs->gsBuffers,
        stream);
    // cudaEventRecord(evt_c, stream);

    robustPruneBatchKernel<<<batchSize, BLOCK_D, 0, stream>>>(
        view.d_graph,
        bufs->d_vectors,
        bufs->d_visitedSets,
        bufs->d_visitedSetCounts,
        bufs->d_pointIds,
        batchSize, alpha);
    // cudaEventRecord(evt_d, stream);

    gpuErrchk(cudaMemsetAsync(bufs->d_revEdgeCount, 0, sizeof(unsigned), stream));
    collectReverseEdgesKernel<<<batchSize, R, 0, stream>>>(
        view.d_graph, bufs->d_pointIds, batchSize,
        bufs->d_revEdgeSrc, bufs->d_revEdgeDst, bufs->d_revEdgeCount);
    // cudaEventRecord(evt_e, stream);

    {
        unsigned maxPairs = batchSize * R;
        appendReverseEdgesRoutedKernel<<<maxPairs, BLOCK_D, 0, stream>>>(
            view.d_graph, view.d_locks,
            bufs->d_revEdgeSrc, bufs->d_revEdgeDst, bufs->d_revEdgeCount,
            alpha);
    }
    // cudaEventRecord(evt_f, stream);

    // cudaEventSynchronize(evt_f);
    // float ms_write = 0.f, ms_greedy = 0.f, ms_prune = 0.f,
    //       ms_collect = 0.f, ms_append = 0.f;
    // cudaEventElapsedTime(&ms_write,   evt_a, evt_b);
    // cudaEventElapsedTime(&ms_greedy,  evt_b, evt_c);
    // cudaEventElapsedTime(&ms_prune,   evt_c, evt_d);
    // cudaEventElapsedTime(&ms_collect, evt_d, evt_e);
    // cudaEventElapsedTime(&ms_append,  evt_e, evt_f);
    // g_insertTimings.writeMs   += ms_write;
    // g_insertTimings.greedyMs  += ms_greedy;
    // g_insertTimings.pruneMs   += ms_prune;
    // g_insertTimings.collectMs += ms_collect;
    // g_insertTimings.appendMs  += ms_append;
    // g_insertTimings.batches++;
    // g_insertTimings.totalInserts += batchSize;

    // Step 8: bump RW count for regular inserts. Phase B re-inserts use
    // existing slots so don't grow RW.
    if (!d_vectorsPreLoaded) idx->rwCount += batchSize;

    return batchSize;
}

// ============================================================================
// insertBatchToView — same kernel pipeline, but targets an EXPLICIT graph view
// (graph + locks + delta + routing threshold) instead of the live UnifiedIndex.
// Phase B of merge calls this with a view pointing at the "delta memory"
// (the new LTI being built) while reading vectors from the OLD graph.
// ============================================================================
unsigned insertBatchToView(UnifiedIndex* idx,
                           GraphView*    view,
                           const float*  h_newVectors,
                           const unsigned* h_newIds,
                           unsigned batchSize,
                           unsigned searchL,
                           float    alpha,
                           InsertBatchBuffers* bufs,
                           unsigned* d_deleted,
                           cudaStream_t stream,
                           bool d_vectorsPreLoaded)
{
    (void)idx;   // present for callers that want index-level bookkeeping later
    if (batchSize == 0) return 0;

    // Pack ids (and vectors if not pre-loaded) into pinned host
    for (unsigned i = 0; i < batchSize; i++)
        bufs->h_pointIds[i] = h_newIds[i];
    if (!d_vectorsPreLoaded) {
        for (size_t _i=0;_i<(size_t)batchSize*D;_i++) bufs->h_vectors[_i]=(GVEC)h_newVectors[_i];
        gpuErrchk(cudaMemcpyAsync(bufs->d_vectors, bufs->h_vectors,
                                  (size_t)batchSize * D * sizeof(GVEC),
                                  cudaMemcpyHostToDevice, stream));
    }
    gpuErrchk(cudaMemcpyAsync(bufs->d_pointIds, bufs->h_pointIds,
                              (size_t)batchSize * sizeof(unsigned),
                              cudaMemcpyHostToDevice, stream));

    writeNewVectorsAndZeroDegree<<<batchSize, BLOCK_D, 0, stream>>>(
        view->d_graph, bufs->d_vectors, bufs->d_pointIds, batchSize);

    gpuErrchk(cudaMemsetAsync(bufs->d_visitedSetCounts, 0,
                              (size_t)batchSize * sizeof(unsigned), stream));
    greedySearchLockedPrealloc(
        view->d_graph,
        view->d_locks,
        bufs->d_vectors,
        bufs->d_visitedSets,
        bufs->d_visitedSetCounts,
        /*batchStart*/ 0,
        batchSize, searchL,
        d_deleted,
        &bufs->gsBuffers,
        stream);

    robustPruneBatchKernel<<<batchSize, BLOCK_D, 0, stream>>>(
        view->d_graph,
        bufs->d_vectors,
        bufs->d_visitedSets,
        bufs->d_visitedSetCounts,
        bufs->d_pointIds,
        batchSize, alpha);

    gpuErrchk(cudaMemsetAsync(bufs->d_revEdgeCount, 0, sizeof(unsigned), stream));
    collectReverseEdgesKernel<<<batchSize, R, 0, stream>>>(
        view->d_graph, bufs->d_pointIds, batchSize,
        bufs->d_revEdgeSrc, bufs->d_revEdgeDst, bufs->d_revEdgeCount);
    {
        unsigned maxPairs = batchSize * R;
        appendReverseEdgesRoutedKernel<<<maxPairs, BLOCK_D, 0, stream>>>(
            view->d_graph, view->d_locks,
            bufs->d_revEdgeSrc, bufs->d_revEdgeDst, bufs->d_revEdgeCount,
            alpha);
    }

    return batchSize;
}

} // namespace fda
