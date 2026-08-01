#ifndef FDA2_EXECUTOR_H
#define FDA2_EXECUTOR_H

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include "fda2_graph.h"
#include "fda2_insert.h"
#include "fda2_delete.h"
#include "fda2_search.h"
#include "../baseline_src/dynamic/workload.h"

namespace fda2 {

struct Stats {
    unsigned long inserts = 0, deletes = 0, queries = 0;
    unsigned long insertBatches = 0, deleteBatches = 0, queryBatches = 0;
    double insertMs = 0, deleteMs = 0, searchMs = 0, wallMs = 0;
    double recall10 = 0, recallComputeMs = 0;
    void print() const;
};

// ============================================================================
// Sequential-model executor.
//
// Phasing rule (spec): INSERT / DELETE / SEARCH never run concurrently with one
// another. The executor accumulates consecutive same-type events into a batch
// and, on any type transition, flushes + stream-syncs the pending work before
// the new type starts — so two different op-types never have kernels in flight
// at the same time. Same-type ops within a batch run in parallel (many blocks).
// ============================================================================
class Executor {
public:
    Executor(uint8_t* h_initialGraph, unsigned numBasePoints, unsigned medoidId,
             unsigned capacity,
             unsigned searchL, unsigned k, float alpha,
             unsigned insertBatchSize, unsigned queryBatchSize, unsigned deleteBatchSize,
             float* h_queries, unsigned numQueries,
             unsigned* h_groundtruth, unsigned gtK);
    ~Executor();

    void runWorkload(const std::vector<WorkloadEvent>& events);
    const Stats& getStats() const { return stats; }

    // Post-run adjacency diagnostic: for every active (non-deleted) node, report
    // degree, live vs stale (deleted-target) edges, and base-region vs inserted-
    // region neighbours. Used to diagnose graph erosion under delete-heavy
    // workloads (e.g. the wiki sliding_window recall collapse). Gated by the
    // FDA2_ADJ_DIAG env var in main.
    void dumpAdjacencyDiagnostic() const;

private:
    UnifiedGraph  g;
    InsertBuffers insertBufs;
    DeleteBuffers deleteBufs;
    SearchBuffers searchBufs;

    unsigned numBasePoints;     // original L (for recall GT classification)
    unsigned searchL, k;
    float    alpha;
    unsigned insertBatchSize, queryBatchSize, deleteBatchSize;

    // external pointId ↔ internal slot
    std::unordered_map<unsigned, unsigned> pointIdToSlot;
    std::vector<unsigned>                  slotToPointId;   // indexed by slot

    // Host mirror of the device deleted flags + current entry point. If a delete
    // tombstones the medoid, greedy would start from a degree-0 node and find
    // nothing; we re-select a live entry point. (The spec's IP-DiskANN delete
    // doesn't cover entry-point maintenance — this keeps search navigable.)
    std::vector<char> hostDeleted;          // indexed by slot (0 = live)
    unsigned findLiveMedoid() const;        // first live slot in [0, numPoints)

    float*    d_allQueryVecs;
    unsigned  numAllQueries;
    unsigned* h_groundtruth;
    unsigned  gtK;

    cudaStream_t stream;

    // pending batches (only one type non-empty at a time)
    std::vector<float>    pendInsVecs;
    std::vector<unsigned> pendInsIds;
    std::vector<unsigned> pendQryIdx;
    std::vector<std::vector<unsigned>> pendQryGT;
    std::vector<unsigned> pendDelIds;

    struct RecallBatch {
        std::vector<unsigned> queryIdxs;
        std::vector<unsigned> predicted;      // [n × k] slot IDs
        std::vector<std::vector<unsigned>> perQueryGT;
    };
    std::vector<RecallBatch> deferredRecall;

    Stats stats;

    void flushInserts();
    void flushDeletes();
    void flushQueries();
    void computeAllRecall();
};

} // namespace fda2

#endif // FDA2_EXECUTOR_H
