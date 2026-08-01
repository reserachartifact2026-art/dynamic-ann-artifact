#ifndef FDA2_GRAPH_H
#define FDA2_GRAPH_H

#include <cuda_runtime.h>
#include <stdint.h>
#include "../baseline_src/vamana.h"

// ============================================================================
// FreshDiskANN — Sequential Model
//
// A SINGLE unified Vamana graph (no LTI / Temp / RW tiers, no merge, no
// delete-list consolidation). Operations are phased: INSERT, DELETE, and
// SEARCH never run concurrently with one another (the executor batches
// consecutive same-type events). Same-type operations DO run in parallel.
//
// Graph layout = the baseline BANG layout, UNCHANGED, so a pre-built vamana.out
// loads verbatim:
//     | Vector (D × 4B) | Degree (4B) | Neighbors (R × 4B) |   (= graphEntrySize)
//
// Spec "R+P adjacency": rather than widen the on-disk entry, the P extra
// candidate slots live in a dedicated per-node scratch region (d_addScratch /
// d_addScratchCount). During an insert batch, reverse edges are atomically
// appended into this P-scratch; a prune pass then forms the union of the node's
// existing ≤R neighbors with its ≤P scratch candidates (≤ R+P total) and runs
// the single common RobustPrune down to R, writing back into the R-slot entry.
// Algorithmically identical to an inline R+P list; keeps baseline_src pristine
// and avoids re-laying-out the graph on load.
// ============================================================================

// Per-node reverse-edge scratch capacity (the spec's "P" extra slots).
#ifndef P_EXTRA
#define P_EXTRA 64
#endif
// Total candidates considered during a reverse-edge prune (R existing neighbors
// + up to P scratch candidates), pruned back down to R.
#define ADJ_CAP (R + P_EXTRA)

namespace fda2 {

struct UnifiedGraph {
    uint8_t*  d_graph;        // [capacity × graphEntrySize] — R-slot entries
    uint8_t*  d_locks;        // [capacity] — uint8 binary lock per node
    unsigned* d_deleted;      // [capacity] — per-slot flag (0 = live, 1 = deleted)

    // Per-node "P extra slots" additions-scratch, shared across phased ops:
    //   INSERT: holds REVERSE edges (back-edges nbr→new) before the R+P→R prune.
    //   DELETE: holds FORWARD additions (z→C_z, y→w) from Algo-5 repair.
    // Named generically (not "rev") because the additions are reverse edges only
    // in the insert phase; in delete they are forward edges.
    unsigned* d_addScratch;      // [capacity × P_EXTRA] — appended neighbor candidates
    unsigned* d_addScratchCount; // [capacity] — per-node append counter (atomic target)

    // Compacted list of nodes that received reverse candidates this batch, so
    // the prune pass touches only affected nodes (not all N).
    unsigned* d_dirtyList;    // [dirtyCap] — unique affected node IDs
    unsigned* d_dirtyCount;   // single uint
    uint8_t*  d_dirtyMark;    // [capacity] — dedup flag for d_dirtyList
    unsigned  dirtyCap;       // capacity of d_dirtyList

    unsigned  numPoints;      // occupied slots (base points + inserted); host mirror
    unsigned  capacity;       // total slot ceiling
    unsigned  medoid;         // greedy entry point
};

// Lifecycle.
void allocateUnifiedGraph(UnifiedGraph* g, unsigned capacity, unsigned dirtyCap);
void freeUnifiedGraph(UnifiedGraph* g);

// Load a pre-built Vamana graph (baseline R-slot layout) into slots [0, numPoints).
void loadGraphFromHost(UnifiedGraph* g, const uint8_t* h_graph,
                       unsigned numPoints, unsigned medoidId);

// Mark a batch of slots as deleted (d_deleted[id] = 1). Used by DELETE.
void markDeletedBatch(UnifiedGraph* g, const unsigned* d_ids, unsigned n,
                      cudaStream_t stream);

} // namespace fda2

#endif // FDA2_GRAPH_H
