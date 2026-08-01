#ifndef FDA2_PRUNE_CUH
#define FDA2_PRUNE_CUH
//
// THE single common RobustPrune (α-RNG) for the whole sequential model.
//
// Spec (carried over from the parent FreshDiskANN spec): "There will be a
// single common RobustPrune used by all operations (such as Insert or Merge)."
// Every caller — insert forward-edge selection, reverse-edge R+P→R pruning, and
// delete consolidation — funnels through `blockAlphaRNG` below. There are NO
// other copies of the prune logic in this project.
//
// Two block-cooperative device primitives:
//   l2DistBlock    — block-cooperative squared-L2 between two D-dim vectors.
//   blockAlphaRNG  — block-cooperative α-RNG diversification of RobustPrune.
//

#include <cuda_runtime.h>
#include "../baseline_src/vamana.h"

namespace fda2 {

// ============================================================================
// l2DistBlock — block-cooperative L2² between vectors `a` and `b` (length D).
// ALL threads in the block must call together. Result broadcast via smemAccum[0].
// Uses up to 8 floats of smemAccum (≤ 8 warps / 256 threads).
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
    for (int offset = 16; offset > 0; offset /= 2)
        partial += __shfl_down_sync(0xffffffffu, partial, offset);
    if ((threadIdx.x & 31u) == 0)
        smemAccum[threadIdx.x >> 5] = partial;
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
// blockAlphaRNG — block-cooperative α-RNG (RobustPrune diversification).
//
// Inputs (shared memory, supplied by the caller):
//   anchorVec        — pointer to the anchor vertex's D-dim vector
//   cands[0..total)  — candidate IDs, MUST be pre-sorted ascending by distance
//                      to the anchor
//   dists[0..total)  — anchor↔candidate distances, ascending (matches cands)
//   total            — number of candidates
//   alpha            — pruning parameter (constant for the whole run)
//   maxSelect        — keep at most this many survivors (= R)
//
// Scratch (caller-provided shared memory):
//   selected[0..maxSelect)  — output survivor IDs
//   p_nSel                  — single uint: number selected (output)
//   smemAccum[0..8)         — for l2DistBlock
//   p_reject                — single bool: per-iteration reject flag
//   p_cCurr                 — single uint: current candidate
//
// Postcondition: *p_nSel = #survivors, selected[0..*p_nSel) = survivor IDs,
// in ascending-distance order.
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

        float distAC = dists[i];                 // dist(anchor, candidate c)
        unsigned c   = *p_cCurr;
        const GVEC* cVec = (const GVEC*)(d_graph + (size_t)c * graphEntrySize);

        unsigned currNSel = *p_nSel;
        for (unsigned j = 0; j < currNSel; j++) {
            if (*p_reject) { __syncthreads(); break; }
            unsigned s = selected[j];
            const GVEC* sVec = (const GVEC*)(d_graph + (size_t)s * graphEntrySize);
            float dsc = l2DistBlock(sVec, cVec, smemAccum);   // dist(selected_j, c)
            if (threadIdx.x == 0) {
                // α-RNG occlusion test: drop c if an already-selected neighbor s
                // is α-times closer to c than the anchor is.
                if (alpha * dsc <= distAC) *p_reject = true;
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

// ============================================================================
// robustPruneBANG — the single common RobustPrune, BANG exact-distance style.
//
// This is the fast replacement for the (l2DistBlock + blockAlphaRNG) path. The
// key difference is the BANG distance pattern: instead of the whole block
// cooperating on ONE distance via a barrier-heavy reduction (l2DistBlock),
// EACH THREAD computes a FULL D-dim distance independently (no reduction, no
// per-distance __syncthreads) — exactly like BANG's compute_L2Dist. That turns
// the α-RNG's ~R² serial block-reductions into a handful of parallel sweeps.
//
// Does the whole prune: (1) anchor↔candidate distances, (2) sort ascending,
// (3) α-RNG occlusion → up to `maxSelect` survivors.
//
//   anchorVec  — [D] anchor vector, readable by all threads (shared mem)
//   cands      — [total] candidate IDs (shared); reordered in place by the sort
//   dists      — [total] scratch for anchor distances (shared)
//   selected   — [maxSelect] survivor output (shared)
//   p_nSel     — survivor count (shared, output)
//   p_reject   — per-iteration reject flag (shared scratch)
//
// Postcondition: *p_nSel survivors in selected[0..*p_nSel), ascending distance.
// ============================================================================
__device__ __forceinline__
void robustPruneBANG(
    const uint8_t* d_graph,
    const float*   anchorVec,
    unsigned*      cands,
    float*         dists,
    unsigned       total,
    float          alpha,
    unsigned       maxSelect,
    unsigned*      selected,
    unsigned*      p_nSel,
    bool*          p_reject)
{
    // (1) anchor↔candidate distances — BANG style: one thread, one full distance.
    for (unsigned i = threadIdx.x; i < total; i += blockDim.x) {
        const GVEC* cVec = (const GVEC*)(d_graph + (size_t)cands[i] * graphEntrySize);
        float d = 0.0f;
        for (unsigned jj = 0; jj < D; jj++) { float df = anchorVec[jj] - cVec[jj]; d += df * df; }
        dists[i] = d;
    }
    __syncthreads();

    // (2) sort candidates ascending by distance (thread 0; total is small).
    if (threadIdx.x == 0) {
        for (unsigned a = 1; a < total; a++) {
            unsigned cid = cands[a]; float cd = dists[a];
            int j = (int)a - 1;
            while (j >= 0 && dists[j] > cd) { cands[j+1] = cands[j]; dists[j+1] = dists[j]; j--; }
            cands[j+1] = cid; dists[j+1] = cd;
        }
        *p_nSel = 0;
    }
    __syncthreads();

    // (3) α-RNG with BANG-style parallel occlusion — one thread per selected node
    // computes a full dist(selected[t], c); no block reduction, ~R× fewer barriers.
    for (unsigned i = 0; i < total && *p_nSel < maxSelect; i++) {
        unsigned c = cands[i];
        float distAC = dists[i];
        const GVEC* cVec = (const GVEC*)(d_graph + (size_t)c * graphEntrySize);
        unsigned nSel = *p_nSel;

        if (threadIdx.x == 0) *p_reject = false;
        __syncthreads();

        for (unsigned t = threadIdx.x; t < nSel; t += blockDim.x) {
            const GVEC* sVec = (const GVEC*)(d_graph + (size_t)selected[t] * graphEntrySize);
            float d = 0.0f;
            for (unsigned jj = 0; jj < D; jj++) { float df = sVec[jj] - cVec[jj]; d += df * df; }
            if (alpha * d <= distAC) *p_reject = true;     // benign same-value race
        }
        __syncthreads();

        if (threadIdx.x == 0 && !*p_reject) {
            selected[*p_nSel] = c;
            (*p_nSel)++;
        }
        __syncthreads();
    }
}

} // namespace fda2

#endif // FDA2_PRUNE_CUH
