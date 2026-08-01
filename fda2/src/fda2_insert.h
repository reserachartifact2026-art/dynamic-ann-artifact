#ifndef FDA2_INSERT_H
#define FDA2_INSERT_H

#include <cuda_runtime.h>
#include "fda2_graph.h"
#include "../baseline_src/vamana.h"

namespace fda2 {

// ============================================================================
// Pre-allocated buffers reused by every insert batch.
// ============================================================================
struct InsertBuffers {
    GVEC*     h_vectors;          // pinned [maxBatch × D]
    unsigned* h_pointIds;         // pinned [maxBatch]
    GVEC*     d_vectors;          // [maxBatch × D]
    unsigned* d_pointIds;         // [maxBatch] — global slot IDs of new nodes
    unsigned* d_visitedSets;      // [maxBatch × MAX_PARENTS_PERQUERY]
    unsigned* d_visitedCounts;    // [maxBatch]
    unsigned* d_revSrc;           // [maxBatch × R] reverse-edge target (the NN)
    unsigned* d_revDst;           // [maxBatch × R] reverse-edge source (new node)
    unsigned* d_revCount;         // single uint
    GreedySearchBuffers gsBuffers;
    unsigned  maxBatch;
    bool      allocated;
};

void allocateInsertBuffers(InsertBuffers* b, unsigned maxBatch);
void freeInsertBuffers(InsertBuffers* b);

// ============================================================================
// Insert a batch of new nodes into the unified graph.
//
// Spec "Insert Kernel":
//   1. Each thread-block searches for R NNs for its new node (greedy search).
//   2. Establish forward edges new → NNs (RobustPrune the visited set to R).
//   3. For each reverse edge NN → new, append the new node as a candidate to
//      the NN's adjacency list (atomic append into the P-scratch).
//   4. If an adjacency list grows beyond R, RobustPrune it back to R.
//
// `h_newIds` are the global slot IDs assigned to the new nodes (must be in
// [g->numPoints_before, capacity)). The caller advances g->numPoints.
// ============================================================================
unsigned insertBatch(UnifiedGraph* g,
                     const float*    h_newVectors,   // [batchSize × D]
                     const unsigned* h_newIds,       // [batchSize]
                     unsigned        batchSize,
                     unsigned        searchL,
                     float           alpha,
                     InsertBuffers*  bufs,
                     cudaStream_t    stream);

} // namespace fda2

#endif // FDA2_INSERT_H
