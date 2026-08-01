#ifndef FDA_INDEX_H
#define FDA_INDEX_H

#include <cuda_runtime.h>
#include <stdint.h>
#include "../baseline_src/vamana.h"

// ============================================================================
// FreshDiskANN — Naive Implementation (paper-faithful, unified-graph design)
//
// The paper's "multi-index" pattern (LTI + multiple Temp Indexes + RW) is a
// LOGICAL partitioning of vertices in a SHARED global ID space. The forward-
// edge graph is global: a vertex's neighbor IDs may reference any region.
// During greedy traversal, an edge from an RW vertex to an LTI vertex is
// followed exactly the way an LTI→LTI edge would be.
//
// We collapse this into a single allocation:
//   d_graph[0  .. ltiCount)            ← LTI (logically read-only)
//   d_graph[ltiCount .. capacity)      ← RW (mutable, locked)
//
// Per-vertex locks (d_locks) cover the whole range for uniform indexing, but
// only RW slots [ltiCount..] are ever acquired by writers — LTI never takes
// the lock.
// ============================================================================

namespace fda {

// ============================================================================
// Multi-index layout (paper-faithful, single unified allocation):
//   [0,         L)              ← LTI (immutable)
//   [L,         L+T)            ← Temp[0]  (immutable once sealed)
//   [L+T,       L+2T)           ← Temp[1]
//   ...
//   [L+(M-1)T,  L+MT)           ← Temp[M-1]
//   [L+MT,      L+(M+1)T)       ← active RW (mutable; locks live here)
//
// rwBase = L + numActiveTemps × T (= start of the current mutable RW).
// rwCount = number of vertices written into current RW (0..T).
//
// "Sealing" an RW = setting numActiveTemps++ and advancing rwBase by T.
// No memory is moved; the slots stay where they are, the routing logic just
// stops accepting locked writes there and starts accepting them in the new
// RW range.
// ============================================================================

struct UnifiedIndex {
    uint8_t*   d_graph;          // [capacity × graphEntrySize] — unified graph
    uint8_t*   d_locks;          // [capacity] — uint8 binary lock per node

    unsigned   ltiCount;         // L — LTI count
    unsigned   tempCapacity;     // T — per-Temp/RW capacity
    unsigned   maxTemps;         // M — max sealed Temps before merge
    unsigned   numActiveTemps;   // current count of sealed Temps (0..M)
    unsigned   rwBase;           // L + numActiveTemps × T — start of current RW
    unsigned   rwCount;          // vertices written into current RW (0..T)

    unsigned   ltiMedoid;        // greedy entry point for LTI region
    unsigned   tempMedoids[64];  // medoid for Temp[i] (= first slot written into it)
    unsigned   rwMedoid;         // medoid for current RW (first slot written)

    unsigned   capacity;         // = L + (M+1) × T
};

// ============================================================================
// GraphView — a (pointer, view) triple for one logical index allocation.
//
// The insert/RobustPrune pipelines work against ONE graph at a time. During
// normal operation the GraphView points at the live UnifiedIndex buffers.
// During Phase B of merge, we build a NEW graph allocation ("delta memory")
// and the pipeline is called with a GraphView pointing at it — same code,
// different target. After Phase B completes we swap the live pointers.
// ============================================================================
struct GraphView {
    uint8_t*  d_graph;
    uint8_t*  d_locks;
    unsigned  immutableEnd;     // = rwBase: start of the mutable RW window
};

inline GraphView makeViewFromIndex(const UnifiedIndex* idx) {
    GraphView v;
    v.d_graph      = idx->d_graph;
    v.d_locks      = idx->d_locks;
    v.immutableEnd = idx->rwBase;
    return v;
}

// Lifecycle.
// `totalCapacity` is the maximum slot count this index can ever reach,
// including post-merge ltiCount growth. Must be ≥ L + (M+1)·T (the minimum
// for one merge cycle) AND ≥ L + total_workload_inserts + T (for many
// merge cycles where ltiCount absorbs each merged Temp).
void allocateUnifiedIndex(UnifiedIndex* idx,
                          unsigned ltiCount,
                          unsigned tempCapacity,
                          unsigned maxTemps,
                          unsigned totalCapacity);
void freeUnifiedIndex(UnifiedIndex* idx);

// Allocate a parallel set of buffers (graph + locks) of the same shape as
// `primary`. Used as the "delta memory" (new-LTI scratch) during merge. The
// caller fills these via Phase A (LTI copy with delete-consolidate) and Phase B
// (Temp/RW re-insert), then atomically swaps them into the primary UnifiedIndex
// pointers and frees the old buffers.
void allocateDeltaMemory(uint8_t**  d_graph_out,
                         uint8_t**  d_locks_out,
                         unsigned   capacity);

// Free buffers previously allocated by allocateDeltaMemory.
void freeDeltaMemory(uint8_t*  d_graph,
                     uint8_t*  d_locks);

// Load a pre-built Vamana graph into the LTI region [0, numPoints).
void loadLTIFromHost(UnifiedIndex* idx, const uint8_t* h_graph,
                     unsigned numPoints, unsigned medoidId);

// Seal the current RW as a Temp Index. Advances rwBase by T, resets rwCount.
// Caller must ensure rwCount == T before calling (or it'll seal a partial RW).
// Returns the index of the new Temp (0..M-1).
unsigned sealCurrentRW(UnifiedIndex* idx);

// Total number of immutable slots = L + numActiveTemps × T.
inline unsigned immutableCount(const UnifiedIndex* idx) {
    return idx->ltiCount + idx->numActiveTemps * idx->tempCapacity;
}

// Total occupied slots (immutable + current RW count).
inline unsigned occupiedCount(const UnifiedIndex* idx) {
    return immutableCount(idx) + idx->rwCount;
}

// ============================================================================
// DeleteList — bitvector covering the whole unified ID space
// ============================================================================

struct DeleteListWrapper {
    unsigned* d_bits;        // bitvector of d_capacity bits
    unsigned  capacity;      // number of bits
    unsigned  count;         // host-side mirror of total deletes
};

void allocateDeleteList(DeleteListWrapper* dl, unsigned capacity);
void freeDeleteList(DeleteListWrapper* dl);
void clearDeleteList(DeleteListWrapper* dl, cudaStream_t stream);
void markDeleted(DeleteListWrapper* dl, unsigned id);
void batchMarkDeleted(DeleteListWrapper* dl,
                      const unsigned* d_ids, unsigned numIds,
                      cudaStream_t stream);

} // namespace fda

#endif // FDA_INDEX_H
