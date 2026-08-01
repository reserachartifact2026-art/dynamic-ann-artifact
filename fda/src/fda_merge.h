#ifndef FDA_MERGE_H
#define FDA_MERGE_H

#include <cuda_runtime.h>
#include "fda_index.h"
#include "fda_insert.h"

namespace fda {

// ============================================================================
// Merge — out-of-place StreamingMerge (driven by the executor, fda_executor.cu).
//
//   Phase A (mergePhaseAOutOfPlaceKernel): read the OLD LTI ∪ sealed Temps, drop
//           deleted neighbors + RobustPrune, write the consolidated lists into a
//           fresh graph at the same slot indices.
//   Phase B (executor::phaseBReinsertAllTempRWToView): GreedySearch+RobustPrune
//           re-insert of every Temp/RW vertex into the new LTI being built.
//   Then the current RW is copied verbatim and the graph pointers are swapped
//   under the global write lock.
// ============================================================================

// Phase A out-of-place: read OLD graph, write the consolidated neighbor list to
// the NEW graph at the same slot index. Drops deleted vertices (leaves new slot
// zero). One block per slot in [0, immutableEnd).
__global__ void mergePhaseAOutOfPlaceKernel(
    const uint8_t*  d_graph_old,
    uint8_t*        d_graph_new,
    const unsigned* d_deleted,
    unsigned immutableEnd, float alpha);

} // namespace fda

#endif // FDA_MERGE_H
