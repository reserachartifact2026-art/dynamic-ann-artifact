#include "fda_merge.h"
#include "fda_locks.cuh"
#include "fda_robust_prune.cuh"
#include <cfloat>
#include <cstdio>

namespace fda {

// ============================================================================
// Helpers
// ============================================================================

__device__ __forceinline__
static bool isDeletedBit(const unsigned* d_deleted, unsigned id) {
    if (!d_deleted) return false;
    return (d_deleted[id / 32] >> (id % 32)) & 1u;
}

// Block-parallel l2 dist: every thread sums a slice of the D elements, then
// warp-reduces. Result is broadcast to all threads via shared memory.
// Caller must __syncthreads() before AND after.
template<typename TA, typename TB>
__device__ __forceinline__
static float l2Dist_blockParallel(const TA* a, const TB* b, float* smemAccum)
{
    float partial = 0.0f;
    for (unsigned i = threadIdx.x; i < D; i += blockDim.x) {
        float diff = a[i] - b[i];
        partial += diff * diff;
    }
    // Warp reduce
    for (int offset = 16; offset > 0; offset /= 2) {
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    }
    // First lane of each warp writes to smemAccum
    if ((threadIdx.x & 31u) == 0) {
        smemAccum[threadIdx.x >> 5] = partial;
    }
    __syncthreads();
    // Thread 0 sums across warps
    if (threadIdx.x == 0) {
        float total = 0.0f;
        unsigned nWarps = (blockDim.x + 31) / 32;
        for (unsigned w = 0; w < nWarps; w++) total += smemAccum[w];
        smemAccum[0] = total;
    }
    __syncthreads();
    return smemAccum[0];
}

// ============================================================================
// Phase A out-of-place: delete-consolidate that READS from the old graph and
// WRITES the consolidated neighbor list to the NEW ("delta memory") graph at
// the same slot index. Deleted vertices contribute nothing (new slot stays
// zero). One block per immutable slot in [0, immutableEnd).
//
// For each surviving vertex v:
//   copy v's vector to the new slot,
//   collect v's non-deleted neighbors, sort by dist(v, ·),
//   single common α-RNG (RobustPrune) → v's new neighbor list.
//
// Note: this used to also fold a per-vertex delta back-edge buffer. That buffer
// was removed — new↔LTI connectivity is rebuilt entirely by Phase B's
// GreedySearch+RobustPrune re-insert of the Temp/RW vertices into the new LTI.
// ============================================================================
__global__ void mergePhaseAOutOfPlaceKernel(
    const uint8_t*  d_graph_old,
    uint8_t*        d_graph_new,
    const unsigned* d_deleted,
    unsigned        immutableEnd, float alpha)
{
    unsigned v = blockIdx.x;
    if (v >= immutableEnd) return;
    if (isDeletedBit(d_deleted, v)) return;

    const uint8_t* oldEntry = d_graph_old + (size_t)v * graphEntrySize;
    const GVEC*    vVec     = (const GVEC*)oldEntry;
    const unsigned* oldDeg  = (const unsigned*)(oldEntry + D*sizeof(GVEC));
    const unsigned* oldNbr  = oldDeg + 1;
    unsigned        degV    = *oldDeg;

    uint8_t*        newEntry = d_graph_new + (size_t)v * graphEntrySize;
    GVEC*           newVec   = (GVEC*)newEntry;
    unsigned*       newDeg   = (unsigned*)(newEntry + D*sizeof(GVEC));
    unsigned*       newNbr   = newDeg + 1;

    // Copy the vector to the new slot
    for (unsigned i = threadIdx.x; i < D; i += blockDim.x) newVec[i] = vVec[i];

    // Candidate pool = v's surviving (non-deleted) neighbors, ≤ R.
    constexpr unsigned MAX_C = R;
    __shared__ unsigned cands[MAX_C];
    __shared__ float    dists[MAX_C];
    __shared__ unsigned nCands;
    __shared__ unsigned selected[R];
    __shared__ unsigned nSel;
    __shared__ float    smemAccum[8];
    __shared__ bool     reject;
    __shared__ unsigned cCurr;

    if (threadIdx.x == 0) { nCands = 0; nSel = 0; }
    __syncthreads();

    // Existing neighbors: drop deleted
    for (unsigned i = threadIdx.x; i < degV; i += blockDim.x) {
        unsigned n = oldNbr[i];
        if (!isDeletedBit(d_deleted, n)) {
            unsigned slot = atomicAdd(&nCands, 1u);
            if (slot < MAX_C) cands[slot] = n;
        }
    }
    __syncthreads();
    unsigned total = nCands;
    if (total > MAX_C) total = MAX_C;

    // Distances v↔candidates from the OLD graph (holds the authoritative
    // vectors for these candidate slots; Phase B fills any Temp/RW slots).
    for (unsigned i = 0; i < total; i++) {
        unsigned c = cands[i];
        const GVEC* cVec = (const GVEC*)(d_graph_old + (size_t)c * graphEntrySize);
        float d = l2Dist_blockParallel(vVec, cVec, smemAccum);
        if (threadIdx.x == 0) dists[i] = d;
    }
    __syncthreads();

    // Sort ascending by distance (insertion sort; total ≤ MAX_C is small)
    if (threadIdx.x == 0) {
        for (unsigned i = 1; i < total; i++) {
            unsigned cid = cands[i]; float d = dists[i];
            int j = (int)i - 1;
            while (j >= 0 && dists[j] > d) {
                cands[j+1] = cands[j]; dists[j+1] = dists[j]; j--;
            }
            cands[j+1] = cid; dists[j+1] = d;
        }
    }
    __syncthreads();

    // Single common α-RNG ("single common RobustPrune used by all operations").
    blockAlphaRNG(d_graph_old,
                  cands, dists,
                  total, alpha, R,
                  selected, &nSel,
                  smemAccum, &reject, &cCurr);

    if (threadIdx.x == 0) *newDeg = nSel;
    __syncthreads();
    for (unsigned i = threadIdx.x; i < nSel; i += blockDim.x) newNbr[i] = selected[i];
}

} // namespace fda
