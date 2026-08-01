#ifndef FDA2_DELETE_H
#define FDA2_DELETE_H

#include <cuda_runtime.h>
#include "fda2_graph.h"
#include "../baseline_src/vamana.h"

namespace fda2 {

// ============================================================================
// Pre-allocated buffers reused by every delete batch.
// ============================================================================
struct DeleteBuffers {
    unsigned* h_delIds;          // pinned [maxBatch]
    unsigned* d_delIds;          // [maxBatch] — slot IDs being deleted (also greedy query slots)
    unsigned* d_visitedSets;     // [maxBatch × MAX_PARENTS_PERQUERY]
    unsigned* d_visitedCounts;   // [maxBatch]
    GreedySearchBuffers gsBuffers;
    unsigned  maxBatch;
    bool      allocated;
};

void allocateDeleteBuffers(DeleteBuffers* b, unsigned maxBatch);
void freeDeleteBuffers(DeleteBuffers* b);

// ============================================================================
// Delete a batch of nodes in place (IP-DiskANN Algorithm 5).
//
// Spec "DeleteKernel":
//   Each thread-block searches for the node to be deleted; the CandidateSet /
//   VisitedSet from GreedySearch is preserved and used to locate in-neighbors.
//   Algorithm 5 in IP-DiskANN deletes the node in place — no deferred / lazy /
//   background consolidation.
//
// Sequence:
//   1. Mark all batch nodes deleted (so splice candidates exclude them and
//      search/greedy stop expanding them).
//   2. Greedy-search each deleted node's own vector → visited set V_p (the
//      in-neighbor candidates).
//   3. For each p, snapshot N_out(p); for each u ∈ V_p with p ∈ N_out(u):
//        C ← (N_out(u) \ {p}) ∪ N_out(p)   (drop deleted, u, p)
//        N_out(u) ← C            if |C| ≤ R
//        N_out(u) ← RobustPrune(u, C, α, R)   otherwise
//      All u-rewrites take u's per-node lock (concurrent deletes may share u).
// ============================================================================
unsigned deleteBatch(UnifiedGraph* g,
                     const unsigned* h_delIds,
                     unsigned        batchSize,
                     unsigned        searchL,
                     float           alpha,
                     DeleteBuffers*  bufs,
                     cudaStream_t    stream);

} // namespace fda2

#endif // FDA2_DELETE_H
