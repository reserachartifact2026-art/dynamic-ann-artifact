#include "fda2_delete.h"
#include "fda2_prune.cuh"
#include "fda2_locks.cuh"   // atomicCAS_u8 for the dedup mark
#include <cstring>
#include <cstdio>

namespace fda2 {

// ============================================================================
// IP-DiskANN Algorithm 5 — exact in-place delete.
//
//   3:  [Visited, Candidates] ← GreedySearch(G, x_p, k, l_d)
//   4:  N'_in(p) ← { z ∈ Visited : p ∈ N_out(z) }
//   5:  for z ∈ N'_in(p):
//   6:      C_z ← closest-c points to x_z in Candidates
//   7:      N_out(z) ← N_out(z) ∪ C_z \ {p}
//   8:  for w ∈ N_out(p):
//   9:      C_w ← closest-c points to x_w in Candidates
//   10:     for y ∈ C_w:
//   11:         N_out(y) ← N_out(y) ∪ {w}
//   12: Remove p
//   13: For any updated v with deg > R:  N_out(v) ← RobustPrune(v, …, R, α)
//
// Constants (paper): l_d = 128, k = 50, c = 3, α = 1.2.
//
// GPU mapping (4 phases + tombstone), all lock-free:
//   greedy (l_d=128)      → Visited (d_visitedSets) + Candidates (top-k worklist)
//   inNeighborRepair      → block/p: emit (z ← C_z) additions to the scratch
//   outNeighborRepair     → block/p: emit (y ← w)  additions to the scratch
//   pruneUpdated          → block per UNIQUE dirty vertex v (grid-STRIDE so the
//                           grid is bounded, never B·R): N_out(v) = (N_out(v)\del)
//                           ∪ additions, RobustPrune if deg > R  [Algo-5 step 13]
//   tombstone             → p removed
//
// Additions are accumulated in the per-node P-scratch (d_addScratch) and the set of
// updated vertices is the deduped dirty list — both reused from insert (phased,
// so the scratch is free). Single owner per updated vertex ⇒ no locks.
// ============================================================================

#ifndef DEL_LD
#ifndef DEL_LD
#define DEL_LD 128      // l_d : deletion greedy beam width
#endif
#endif
#ifndef DEL_K
#define DEL_K  50       // k   : candidate-list size used for the closest-c search
#endif
#define DEL_C  2        // c   : edge copies (closest-c) -- matches DEL_C=2 sweep

// pruneUpdated grid: fixed and grid-strided so we never launch B·R blocks.
#ifndef DEL_MERGE_GRID
#define DEL_MERGE_GRID 16384u
#endif

// ============================================================================
// Buffers
// ============================================================================
void allocateDeleteBuffers(DeleteBuffers* b, unsigned maxBatch) {
    b->maxBatch = maxBatch;
    gpuErrchk(cudaMallocHost(&b->h_delIds, (size_t)maxBatch * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_delIds,   (size_t)maxBatch * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_visitedSets,
                          (size_t)maxBatch * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&b->d_visitedCounts, (size_t)maxBatch * sizeof(unsigned)));
    allocateGreedySearchBuffers(&b->gsBuffers, maxBatch);
    b->allocated = true;
}

void freeDeleteBuffers(DeleteBuffers* b) {
    if (!b->allocated) return;
    cudaFreeHost(b->h_delIds);
    cudaFree(b->d_delIds);
    cudaFree(b->d_visitedSets);
    cudaFree(b->d_visitedCounts);
    freeGreedySearchBuffers(&b->gsBuffers);
    b->allocated = false;
}

// ----------------------------------------------------------------------------
// Device helpers
// ----------------------------------------------------------------------------

// closest-c points to xv among cands[0..nCands), excluding deleted + excl1/excl2.
// Serial (called by one thread). Writes ≤ DEL_C ids to out, count to *outN.
__device__ __forceinline__
void closestCCandidates(const uint8_t* d_graph, const GVEC* xv,
                        const unsigned* cands, unsigned nCands,
                        const unsigned* d_deleted,
                        unsigned excl1, unsigned excl2,
                        unsigned* out, unsigned* outN)
{
    float    bD[DEL_C];
    unsigned bI[DEL_C];
    unsigned n = 0;
    for (unsigned j = 0; j < nCands; j++) {
        unsigned cid = cands[j];
        if (cid == 0xFFFFFFFFu || cid == excl1 || cid == excl2) continue;
        if (d_deleted && d_deleted[cid]) continue;
        const GVEC* cv = (const GVEC*)(d_graph + (size_t)cid * graphEntrySize);
        float d = 0.0f;
        for (unsigned t = 0; t < D; t++) { float df = xv[t] - cv[t]; d += df * df; }
        if (n < DEL_C) {
            unsigned pos = n;
            while (pos > 0 && bD[pos-1] > d) { bD[pos]=bD[pos-1]; bI[pos]=bI[pos-1]; pos--; }
            bD[pos] = d; bI[pos] = cid; n++;
        } else if (d < bD[DEL_C-1]) {
            unsigned pos = DEL_C-1;
            while (pos > 0 && bD[pos-1] > d) { bD[pos]=bD[pos-1]; bI[pos]=bI[pos-1]; pos--; }
            bD[pos] = d; bI[pos] = cid;
        }
    }
    for (unsigned i = 0; i < n; i++) out[i] = bI[i];
    *outN = n;
}

// Register `v` in the dirty list exactly once (idempotent).
// Invariant: markDirty increments the count at most once per vertex (the CAS
// dedup), and dirtyCap == capacity == #slots, so idx < dirtyCap always holds.
// The bound check + count rollback keep d_dirtyCount and d_dirtyList in lock-step
// even if a smaller cap were ever configured (consumers can trust count ≤ cap).
__device__ __forceinline__
void markDirty(unsigned v, uint8_t* d_dirtyMark, unsigned* d_dirtyList,
               unsigned* d_dirtyCount, unsigned dirtyCap)
{
    if (atomicCAS_u8(&d_dirtyMark[v], 0u, 1u) == 0u) {
        unsigned idx = atomicAdd(d_dirtyCount, 1u);
        if (idx < dirtyCap) d_dirtyList[idx] = v;
        else                atomicSub(d_dirtyCount, 1u);   // overflow: keep count == #entries
    }
}

// Append `addId` into v's scratch (drop on overflow) and mark v dirty.
__device__ __forceinline__
void appendAddition(unsigned v, unsigned addId,
                    unsigned* d_addScratch, unsigned* d_addScratchCount,
                    uint8_t* d_dirtyMark, unsigned* d_dirtyList,
                    unsigned* d_dirtyCount, unsigned dirtyCap)
{
    if (v == addId) return;
    unsigned slot = atomicAdd(&d_addScratchCount[v], 1u);
    if (slot < P_EXTRA) d_addScratch[(size_t)v * P_EXTRA + slot] = addId;
    else                atomicSub(&d_addScratchCount[v], 1u);   // overflow: drop
    markDirty(v, d_dirtyMark, d_dirtyList, d_dirtyCount, dirtyCap);
}

// Append a whole run of additions (adds[0..n)) targeting the SAME vertex v.
// One atomicAdd reserves the contiguous run instead of n separate atomics; the
// caller already holds the full list (e.g. the closest-c set), so there is no
// reason to bump the counter per element. Self-loops are skipped; on partial
// overflow the count is rolled back to the actual number written.
__device__ __forceinline__
void appendAdditions(unsigned v, const unsigned* adds, unsigned n,
                     unsigned* d_addScratch, unsigned* d_addScratchCount,
                     uint8_t* d_dirtyMark, unsigned* d_dirtyList,
                     unsigned* d_dirtyCount, unsigned dirtyCap)
{
    // Compact out self-loops up front so the reservation is exact.
    unsigned m = 0;
    unsigned tmp[DEL_C];
    for (unsigned i = 0; i < n; i++) if (adds[i] != v) tmp[m++] = adds[i];
    if (m == 0) return;

    unsigned base = atomicAdd(&d_addScratchCount[v], m);
    unsigned wrote = 0;
    for (unsigned i = 0; i < m && base + i < P_EXTRA; i++) {
        d_addScratch[(size_t)v * P_EXTRA + base + i] = tmp[i];
        wrote++;
    }
    if (wrote < m) atomicSub(&d_addScratchCount[v], m - wrote);   // overflow: drop tail
    markDirty(v, d_dirtyMark, d_dirtyList, d_dirtyCount, dirtyCap);
}

// ----------------------------------------------------------------------------
// Phase: in-neighbor repair (Algo 5 lines 4–7). One block per deleted node p.
//   for z ∈ Visited with p ∈ N_out(z):  add closest-c candidates of z; (drop p
//   is handled at merge by skipping deleted neighbors).
// ----------------------------------------------------------------------------
__global__ void inNeighborRepairKernel(
    uint8_t* d_graph, const unsigned* d_deleted,
    const unsigned* d_visitedSets, const unsigned* d_visitedCounts,
    const unsigned* d_worklist, const unsigned* d_worklistCount, unsigned worklistStride,
    const unsigned* d_delIds, unsigned batchSize,
    unsigned* d_addScratch, unsigned* d_addScratchCount,
    uint8_t* d_dirtyMark, unsigned* d_dirtyList, unsigned* d_dirtyCount, unsigned dirtyCap)
{
    unsigned bIdx = blockIdx.x;
    if (bIdx >= batchSize) return;
    unsigned p = d_delIds[bIdx];

    unsigned nC = d_worklistCount[bIdx]; if (nC > DEL_K) nC = DEL_K;
    const unsigned* cands = d_worklist + (size_t)bIdx * worklistStride;

    unsigned visited = d_visitedCounts[bIdx];
    if (visited > MAX_PARENTS_PERQUERY) visited = MAX_PARENTS_PERQUERY;

    // Each thread handles a slice of the visited set (fully parallel, lock-free).
    for (unsigned vi = threadIdx.x; vi < visited; vi += blockDim.x) {
        unsigned z = d_visitedSets[(size_t)bIdx * MAX_PARENTS_PERQUERY + vi];
        if (z == p) continue;
        if (d_deleted && d_deleted[z]) continue;

        // is p ∈ N_out(z)? (z is an in-neighbor of p)
        unsigned* degPtr = (unsigned*)(d_graph + (size_t)z * graphEntrySize + D*sizeof(GVEC));
        unsigned* nbrPtr = degPtr + 1;
        unsigned dz = *degPtr; if (dz > R) dz = R;
        bool isIn = false;
        for (unsigned i = 0; i < dz; i++) if (nbrPtr[i] == p) { isIn = true; break; }
        if (!isIn) continue;

        // C_z = closest-c candidates to x_z; add them. (p removed at merge time.)
        const GVEC* xz = (const GVEC*)(d_graph + (size_t)z * graphEntrySize);
        unsigned cz[DEL_C], ncz = 0;
        closestCCandidates(d_graph, xz, cands, nC, d_deleted, p, z, cz, &ncz);
        appendAdditions(z, cz, ncz, d_addScratch, d_addScratchCount,
                        d_dirtyMark, d_dirtyList, d_dirtyCount, dirtyCap);
        // z must be re-merged even if ncz==0, so p gets dropped from N_out(z).
        markDirty(z, d_dirtyMark, d_dirtyList, d_dirtyCount, dirtyCap);
    }
}

// ----------------------------------------------------------------------------
// Phase: out-neighbor repair (Algo 5 lines 8–11). One block per deleted node p.
//   for w ∈ N_out(p): C_w = closest-c candidates to x_w; for y ∈ C_w add w to y.
// ----------------------------------------------------------------------------
__global__ void outNeighborRepairKernel(
    uint8_t* d_graph, const unsigned* d_deleted,
    const unsigned* d_worklist, const unsigned* d_worklistCount, unsigned worklistStride,
    const unsigned* d_delIds, unsigned batchSize,
    unsigned* d_addScratch, unsigned* d_addScratchCount,
    uint8_t* d_dirtyMark, unsigned* d_dirtyList, unsigned* d_dirtyCount, unsigned dirtyCap)
{
    unsigned bIdx = blockIdx.x;
    if (bIdx >= batchSize) return;
    unsigned p = d_delIds[bIdx];

    unsigned nC = d_worklistCount[bIdx]; if (nC > DEL_K) nC = DEL_K;
    const unsigned* cands = d_worklist + (size_t)bIdx * worklistStride;

    unsigned* pDeg = (unsigned*)(d_graph + (size_t)p * graphEntrySize + D*sizeof(GVEC));
    unsigned* pNbr = pDeg + 1;
    unsigned dp = *pDeg; if (dp > R) dp = R;

    for (unsigned wi = threadIdx.x; wi < dp; wi += blockDim.x) {
        unsigned w = pNbr[wi];
        if (w == p) continue;
        if (d_deleted && d_deleted[w]) continue;   // don't reconnect a deleted out-neighbor

        const GVEC* xw = (const GVEC*)(d_graph + (size_t)w * graphEntrySize);
        unsigned cw[DEL_C], ncw = 0;
        closestCCandidates(d_graph, xw, cands, nC, d_deleted, p, w, cw, &ncw);
        for (unsigned i = 0; i < ncw; i++)
            appendAddition(cw[i], w, d_addScratch, d_addScratchCount,
                           d_dirtyMark, d_dirtyList, d_dirtyCount, dirtyCap);
    }
}

// ----------------------------------------------------------------------------
// Phase: merge + prune (Algo 5 lines 7,11,13). GRID-STRIDE over the dirty list
// so the grid is bounded (DEL_MERGE_GRID blocks), NOT B·R. One iteration = one
// UNIQUE updated vertex v → single owner → no lock.
//   N_out(v) ← (N_out(v) \ deleted) ∪ additions(v) ;  RobustPrune if > R.
// ----------------------------------------------------------------------------
__global__ void pruneUpdatedKernel(
    uint8_t* d_graph, const unsigned* d_deleted,
    unsigned* d_addScratch, unsigned* d_addScratchCount,
    uint8_t* d_dirtyMark, const unsigned* d_dirtyList, const unsigned* d_dirtyCount,
    float alpha)
{
    constexpr unsigned MAX_C = R + P_EXTRA;
    __shared__ unsigned cands[MAX_C];
    __shared__ float    dists[MAX_C];
    __shared__ unsigned nCands;
    __shared__ unsigned selected[R];
    __shared__ unsigned nSel;
    __shared__ float    anchorVec[D];
    __shared__ bool     reject;

    unsigned totalDirty = *d_dirtyCount;

    for (unsigned b = blockIdx.x; b < totalDirty; b += gridDim.x) {
        unsigned v = d_dirtyList[b];

        if (d_deleted && d_deleted[v]) {            // v itself got deleted — skip
            if (threadIdx.x == 0) { d_addScratchCount[v] = 0; d_dirtyMark[v] = 0; }
            __syncthreads();
            continue;
        }

        uint8_t* entry = d_graph + (size_t)v * graphEntrySize;
        const GVEC* vVec = (const GVEC*)entry;
        unsigned* degPtr = (unsigned*)(entry + D*sizeof(GVEC));
        unsigned* nbrPtr = degPtr + 1;
        unsigned dv = *degPtr; if (dv > R) dv = R;
        unsigned addN = d_addScratchCount[v]; if (addN > P_EXTRA) addN = P_EXTRA;

        // cache v's vector; thread 0 builds the deduped candidate set.
        for (unsigned i = threadIdx.x; i < D; i += blockDim.x) anchorVec[i] = vVec[i];
        if (threadIdx.x == 0) {
            nCands = 0;
            // existing neighbors, dropping deleted (this removes p — Algo 5 "\{p}")
            for (unsigned i = 0; i < dv; i++) {
                unsigned nb = nbrPtr[i];
                if (nb == v) continue;
                if (d_deleted && d_deleted[nb]) continue;
                bool dup = false;
                for (unsigned c = 0; c < nCands; c++) if (cands[c] == nb) { dup = true; break; }
                if (!dup && nCands < MAX_C) cands[nCands++] = nb;
            }
            // additions (the ∪ C_z / ∪ {w}), deduped, dropping deleted
            for (unsigned i = 0; i < addN; i++) {
                unsigned m = d_addScratch[(size_t)v * P_EXTRA + i];
                if (m == v) continue;
                if (d_deleted && d_deleted[m]) continue;
                bool dup = false;
                for (unsigned c = 0; c < nCands; c++) if (cands[c] == m) { dup = true; break; }
                if (!dup && nCands < MAX_C) cands[nCands++] = m;
            }
        }
        __syncthreads();

        unsigned total = nCands; if (total > MAX_C) total = MAX_C;

        if (total <= R) {                           // within degree — keep all
            if (threadIdx.x == 0) *degPtr = total;
            __syncthreads();
            for (unsigned i = threadIdx.x; i < total; i += blockDim.x) nbrPtr[i] = cands[i];
        } else {                                    // exceeded R — RobustPrune
            robustPruneBANG(d_graph, anchorVec, cands, dists, total, alpha, R,
                            selected, &nSel, &reject);
            if (threadIdx.x == 0) *degPtr = nSel;
            __syncthreads();
            for (unsigned i = threadIdx.x; i < nSel; i += blockDim.x) nbrPtr[i] = selected[i];
        }
        __syncthreads();
        if (threadIdx.x == 0) { d_addScratchCount[v] = 0; d_dirtyMark[v] = 0; }
        __syncthreads();
    }
}

// ----------------------------------------------------------------------------
// Tombstone deleted nodes (Algo 5 line 12), after merge has read N_out(p).
// ----------------------------------------------------------------------------
__global__ void tombstoneDeletedKernel(uint8_t* d_graph,
                                       const unsigned* d_delIds, unsigned batchSize)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batchSize) return;
    unsigned p = d_delIds[i];
    unsigned* degPtr = (unsigned*)(d_graph + (size_t)p * graphEntrySize + D*sizeof(GVEC));
    *degPtr = 0;
}

// ============================================================================
// Driver — exact Algorithm 5
// ============================================================================
unsigned deleteBatch(UnifiedGraph* g,
                     const unsigned* h_delIds,
                     unsigned        batchSize,
                     unsigned        searchL,   // ignored; Algo 5 uses l_d = DEL_LD
                     float           alpha,
                     DeleteBuffers*  bufs,
                     cudaStream_t    stream)
{
    if (batchSize == 0) return 0;
    (void)searchL;

    setRuntimeMedoid(g->medoid, stream);

    for (unsigned i = 0; i < batchSize; i++) bufs->h_delIds[i] = h_delIds[i];
    gpuErrchk(cudaMemcpyAsync(bufs->d_delIds, bufs->h_delIds,
                              (size_t)batchSize * sizeof(unsigned),
                              cudaMemcpyHostToDevice, stream));

    // Mark the batch deleted so neither Candidates nor splices include them.
    markDeletedBatch(g, bufs->d_delIds, batchSize, stream);

    // Step 3: GreedySearch(G, x_p, k, l_d) → Visited (d_visitedSets) +
    // Candidates (the sorted worklist; we take its top-DEL_K). Beam = l_d.
    // x_p (the query vector) is read in place from each deleted node's own graph
    // entry via d_querySlots = d_delIds — no gather/copy. (The query-vector read
    // is independent of d_deleted, which only filters neighbors during traversal,
    // so marking the batch deleted first is harmless.)
    gpuErrchk(cudaMemsetAsync(bufs->d_visitedCounts, 0,
                              (size_t)batchSize * sizeof(unsigned), stream));
    greedySearchNoSyncPrealloc(
        g->d_graph, /*d_queryVecs=*/nullptr,
        bufs->d_visitedSets, bufs->d_visitedCounts,
        /*batchStart*/ 0, batchSize, /*searchL=*/DEL_LD,
        g->d_deleted, &bufs->gsBuffers, stream,
        /*d_querySlots=*/bufs->d_delIds);

    // Reset the dirty-vertex set; the per-node scratch counts are already 0
    // (reset by the previous insert/delete; merge resets each touched node).
    gpuErrchk(cudaMemsetAsync(g->d_dirtyCount, 0, sizeof(unsigned), stream));

    // Steps 4–7: in-neighbor repair (one block per deleted node).
    inNeighborRepairKernel<<<batchSize, 128, 0, stream>>>(
        g->d_graph, g->d_deleted,
        bufs->d_visitedSets, bufs->d_visitedCounts,
        bufs->gsBuffers.d_worklist, bufs->gsBuffers.d_worklistCount, DEL_LD,
        bufs->d_delIds, batchSize,
        g->d_addScratch, g->d_addScratchCount,
        g->d_dirtyMark, g->d_dirtyList, g->d_dirtyCount, g->dirtyCap);

    // Steps 8–11: out-neighbor repair (one block per deleted node).
    // 64 threads = R: each thread owns exactly one out-neighbor (dp ≤ R = 64);
    // 128 would leave the upper 64 threads idle every block.
    outNeighborRepairKernel<<<batchSize, 64, 0, stream>>>(
        g->d_graph, g->d_deleted,
        bufs->gsBuffers.d_worklist, bufs->gsBuffers.d_worklistCount, DEL_LD,
        bufs->d_delIds, batchSize,
        g->d_addScratch, g->d_addScratchCount,
        g->d_dirtyMark, g->d_dirtyList, g->d_dirtyCount, g->dirtyCap);

    // Steps 7,11,13: merge additions into each updated vertex + RobustPrune.
    // Bounded grid + grid-stride (NOT B·R blocks).
    pruneUpdatedKernel<<<DEL_MERGE_GRID, 128, 0, stream>>>(
        g->d_graph, g->d_deleted,
        g->d_addScratch, g->d_addScratchCount,
        g->d_dirtyMark, g->d_dirtyList, g->d_dirtyCount, alpha);

    // Step 12: remove p.
    {
        unsigned threads = 256;
        unsigned blocks  = (batchSize + threads - 1) / threads;
        tombstoneDeletedKernel<<<blocks, threads, 0, stream>>>(
            g->d_graph, bufs->d_delIds, batchSize);
    }

    // Leave d_dirtyMark all-zero for the next op (merge cleared the touched ones,
    // but overflow-dropped marks past dirtyCap may remain — clear defensively).
    gpuErrchk(cudaMemsetAsync(g->d_dirtyMark, 0,
                              (size_t)g->capacity * sizeof(uint8_t), stream));

    gpuErrchk(cudaGetLastError());
    return batchSize;
}

} // namespace fda2
