#ifndef FDA_SEARCH_H
#define FDA_SEARCH_H

#include <cuda_runtime.h>
#include "fda_index.h"
#include "fda_executor.h"
#include "../baseline_src/vamana.h"

namespace fda {

// ============================================================================
// Unified search (spec §"Search Kernel"): search LTI, each sealed Temp, and the
// RW index independently (each from its own medoid), then merge the per-index
// top-k's and filter the result against the DeleteList.
// ============================================================================
void searchUnified(
    const UnifiedIndex*        idx,
    const DeleteListWrapper*   deleteList,
    const GVEC*                d_queries,     // [bs × D]
    unsigned                   bs,
    unsigned                   k,
    unsigned                   searchL,
    SearchBuffers*             sb,
    cudaStream_t               stream);

// ============================================================================
// Helper kernels
// ============================================================================

// Filter the final top-k by the delete bitvector — compact non-deleted entries
// to the front, pad sentinels.
__global__ void filterDeletesKernel(
    unsigned*       d_topkIds,
    float*          d_topkDists,
    const unsigned* d_deleted,
    unsigned        bs, unsigned k);

// Padding diagnostic: read padding-hit counters and print summary. Call once
// after the workload is finished.
void reportPaddingDiagnostic(unsigned totalQueries);

} // namespace fda

#endif // FDA_SEARCH_H
