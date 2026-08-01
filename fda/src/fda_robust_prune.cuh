#ifndef FDA_ROBUST_PRUNE_CUH
#define FDA_ROBUST_PRUNE_CUH
//
// Single common RobustPrune (α-RNG) used by Insert, locked back-edge
// overflow, and Merge.  Per the design:
//
//     "single common RobustPrune used by all operations
//      (such as Insert or Merge)."
//
// This file holds the two device-side primitives shared across all callers:
//
//   l2DistBlock       — block-cooperative L2 distance between two D-dim
//                       float vectors. All threads in the block participate;
//                       result is broadcast via shared memory.
//
//   blockAlphaRNG     — block-cooperative α-RNG (the diversification step
//                       of RobustPrune). Takes a candidate set already
//                       sorted by distance to the anchor vertex, returns up
//                       to `maxSelect` survivors.
//

#include <cuda_runtime.h>
#include "../baseline_src/vamana.h"

namespace fda {

// ============================================================================
// l2DistBlock — block-cooperative L2² between vectors `a` and `b` (each of
// length D). All threads in the block must call together. Result is broadcast
// via smemAccum[0]. Uses 8 floats of smemAccum (supports up to 8 warps).
// ============================================================================
template<typename TA, typename TB>
__device__ __forceinline__
float l2DistBlock(const TA* a, const TB* b, float* smemAccum)
{
    float partial = 0.0f;
    for (unsigned i = threadIdx.x; i < D; i += blockDim.x) {
        float diff = a[i] - b[i];
        partial += diff * diff;
    }
    // Warp-level reduce
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

// ============================================================================
// blockAlphaRNG — block-cooperative α-RNG.
//
// Inputs (all in shared memory, supplied by the caller):
//   cands[0..total)   — candidate IDs (must be pre-sorted by distance to anchor)
//   dists[0..total)   — distances anchor↔candidate, ascending
//   total             — # candidates
//   alpha             — pruning parameter
//   maxSelect         — pick at most this many survivors (typically R)
//
// Scratch (caller-provided shared memory):
//   selected[0..maxSelect)  — output slots for survivor IDs
//   p_nSel                  — single uint in smem: written with # selected
//   smemAccum[0..8)         — for l2DistBlock
//   p_reject                — single bool in smem: per-iteration reject flag
//   p_cCurr                 — single uint in smem: current candidate
//
// Postcondition: *p_nSel = number of selected, selected[0..*p_nSel) = survivors.
// ============================================================================
__device__ __forceinline__
void blockAlphaRNG(
    const uint8_t* d_graph,
    unsigned*      cands,
    float*         dists,
    unsigned       total,
    float          alpha,
    unsigned       maxSelect,
    unsigned*      selected,
    unsigned*      p_nSel,
    float*         smemAccum,
    bool*          p_reject,
    unsigned*      p_cCurr)
{
    if (threadIdx.x == 0) *p_nSel = 0;
    __syncthreads();

    for (unsigned i = 0; i < total && *p_nSel < maxSelect; i++) {
        if (threadIdx.x == 0) { *p_cCurr = cands[i]; *p_reject = false; }
        __syncthreads();

        float distVC = dists[i];
        unsigned c = *p_cCurr;
        const GVEC* cVec = (const GVEC*)(d_graph + (size_t)c * graphEntrySize);

        unsigned currNSel = *p_nSel;
        for (unsigned j = 0; j < currNSel; j++) {
            if (*p_reject) { __syncthreads(); break; }
            unsigned s = selected[j];
            const GVEC* sVec = (const GVEC*)(d_graph + (size_t)s * graphEntrySize);
            float dsc = l2DistBlock(sVec, cVec, smemAccum);
            if (threadIdx.x == 0) {
                if (alpha * dsc <= distVC) *p_reject = true;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0 && !*p_reject) {
            selected[*p_nSel] = c;
            (*p_nSel)++;
        }
        __syncthreads();
    }
}

} // namespace fda

#endif // FDA_ROBUST_PRUNE_CUH
