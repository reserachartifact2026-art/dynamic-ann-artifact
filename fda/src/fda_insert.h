#ifndef FDA_INSERT_H
#define FDA_INSERT_H

#include <cuda_runtime.h>
#include "fda_index.h"
#include "../baseline_src/vamana.h"

namespace fda {

// ============================================================================
// Pre-allocated buffers reused by every insert batch into the unified index.
// ============================================================================
struct InsertBatchBuffers {
    GVEC*     h_vectors;            // pinned host [maxBatch × D]
    unsigned* h_pointIds;           // pinned host [maxBatch]
    GVEC*     d_vectors;            // device [maxBatch × D]
    unsigned* d_pointIds;           // device [maxBatch] — = global slot IDs
    unsigned* d_visitedSets;        // device [maxBatch × MAX_PARENTS_PERQUERY]
    unsigned* d_visitedSetCounts;   // device [maxBatch]
    unsigned* d_revEdgeSrc;         // device [maxBatch × R] target IDs (chosen neighbor)
    unsigned* d_revEdgeDst;         // device [maxBatch × R] source IDs (new vertex)
    unsigned* d_revEdgeCount;       // device single uint
    GreedySearchBuffers  gsBuffers;
    OutNeighborsBuffers  outBuffers;
    unsigned  maxBatch;
    bool      allocated;
};

void allocateInsertBatchBuffers(InsertBatchBuffers* b, unsigned maxBatch);
void freeInsertBatchBuffers(InsertBatchBuffers* b);

// Prints accumulated per-stage timing for the insert pipeline.
void dumpInsertTimings();

// ============================================================================
// Reverse-edge kernel — for each (target T, source P) pair: acquire d_locks[T],
// append P to T's neighbor list (or α-RNG on overflow), release.
// ============================================================================
__global__ void appendReverseEdgesRoutedKernel(
    uint8_t*  d_graph,
    uint8_t*  d_locks,
    unsigned* d_revSrc,
    unsigned* d_revDst,
    unsigned* d_revCount,
    float     alpha);

// ============================================================================
// Driver: full insert batch into the unified index.
//
// Greedy runs from the current RW medoid. RobustPrune picks R out-neighbors
// from the visited set. New vertex p (slot = workload pointId) is written into
// d_graph[p] with those forward edges. Reverse edges are appended into the
// target's neighbor list under the per-node lock (α-RNG on overflow).
// ============================================================================
unsigned insertBatchUnified(UnifiedIndex* idx,
                            const float*  h_newVectors,    // [batchSize × D]; ignored if d_vectorsPreLoaded
                            const unsigned* h_newIds,       // [batchSize] — global slot IDs
                            unsigned batchSize,
                            unsigned searchL,
                            float    alpha,
                            InsertBatchBuffers* bufs,
                            unsigned* d_deleted,
                            cudaStream_t stream,
                            // Phase B helper: when true, caller has already filled
                            // bufs->d_vectors with device-side vectors (e.g. via D→D
                            // copy from existing graph slots) and we skip the
                            // H→D copy step. Set false for regular workload inserts.
                            bool d_vectorsPreLoaded = false);

// GraphView-targeted variant. Same semantics as insertBatchUnified but writes
// to the explicit (d_graph, d_locks, immutableEnd) view instead of dereferencing
// through idx. Used by Phase B of merge to target the new-LTI scratch ("delta
// memory") instead of the live graph.
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
                           bool d_vectorsPreLoaded = false);

} // namespace fda

#endif // FDA_INSERT_H
