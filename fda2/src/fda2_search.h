#ifndef FDA2_SEARCH_H
#define FDA2_SEARCH_H

#include <cuda_runtime.h>
#include "fda2_graph.h"
#include "../baseline_src/vamana.h"

namespace fda2 {

// Per-query scratch for search.
struct SearchBuffers {
    GreedySearchBuffers gsBuffers;
    unsigned* d_topkIds;       // [qbs × k]
    float*    d_topkDists;     // [qbs × k]
    unsigned* h_topkIds;       // pinned [qbs × k]
    GVEC*    d_queryVecs;     // [qbs × D]
    unsigned* d_visitedSets;   // [qbs × MAX_PARENTS_PERQUERY]
    unsigned* d_visitedCounts; // [qbs]
    float*    d_visitedSetDists;// [qbs × MAX_PARENTS_PERQUERY] — fused kernel output
    unsigned* d_qIdx;          // [qbs] query indices (for the gather kernel)
    unsigned* h_qIdx;          // [qbs] pinned host mirror
    unsigned  maxBatch;
    bool      allocated;
};

// Gather query vectors d_allQueryVecs[h_indices[i]] → sb->d_queryVecs[i] using a
// SINGLE index copy + one kernel launch (replaces a per-query cudaMemcpyAsync
// loop that was tens of ms of launch overhead per batch).
void gatherQueries(const float* d_allQueryVecs, SearchBuffers* sb,
                   const unsigned* h_indices, unsigned bs, cudaStream_t stream);

void allocateSearchBuffers(SearchBuffers* sb, unsigned qbs);
void freeSearchBuffers(SearchBuffers* sb);

// ============================================================================
// BANG search on the single unified graph.
//   1. greedy (lock-free) from the medoid
//   2. extract top-k from the worklist
//   3. drop any deleted slot from the result (compact + pad sentinels)
// Result top-k slot IDs land in sb->d_topkIds / sb->h_topkIds.
// ============================================================================
void searchBatch(const UnifiedGraph* g,
                 const GVEC* d_queries,
                 unsigned bs,
                 unsigned k,
                 unsigned searchL,
                 SearchBuffers* sb,
                 cudaStream_t stream);

} // namespace fda2

#endif // FDA2_SEARCH_H
