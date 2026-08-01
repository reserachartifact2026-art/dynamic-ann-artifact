#ifndef FDA_EXECUTOR_H
#define FDA_EXECUTOR_H

#include <cuda_runtime.h>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include "fda_index.h"
#include "../baseline_src/vamana.h"
#include "../baseline_src/dynamic/workload.h"

namespace fda {

struct FdaStats {
    unsigned long inserts        = 0;
    unsigned long deletes        = 0;
    unsigned long queries        = 0;
    unsigned long insertBatches  = 0;
    unsigned long queryBatches   = 0;
    unsigned long deleteBatches  = 0;
    unsigned long mergesFired    = 0;
    double totalInsertMs    = 0.0;
    double totalSearchMs    = 0.0;
    double totalDeleteMs    = 0.0;
    double totalMergeMs     = 0.0;
    double totalWallMs      = 0.0;
    double totalRecall10    = 0.0;
    double recallComputeMs  = 0.0;

    void print() const;
};

// Per-query scratch buffers
struct SearchBuffers {
    GreedySearchBuffers gsBuffers;
    unsigned* d_topkIds;
    float*    d_topkDists;
    unsigned* h_topkIds;
    float*    h_topkDists;
    GVEC*    d_queryVecs;
    unsigned* d_idxBuf;
    unsigned* h_idxBuf;
    // Separate visited-set buffer: the filterNeighbors* kernels write here
    // with stride MAX_PARENTS_PERQUERY. Distinct from gsBuffers.d_worklist
    // (which has stride MAX_L); aliasing them OOBs across queries.
    unsigned* d_visitedSets;     // [qbs × MAX_PARENTS_PERQUERY]
    unsigned* d_visitedCounts;   // [qbs]
    bool      allocated = false;
};

void allocateSearchBuffers(SearchBuffers* sb, unsigned qbs);
void freeSearchBuffers(SearchBuffers* sb);

class FdaExecutor {
public:
    FdaExecutor(uint8_t* h_initialGraph,         // base graph for LTI
                unsigned numBasePoints,
                unsigned medoidId,
                unsigned ltiCount,                // L
                unsigned tempCapacity,            // T (per Temp/RW)
                unsigned maxTemps,                // M (max sealed Temps before merge)
                unsigned totalCapacity,           // unified-graph slot ceiling
                unsigned deleteListCapacity,
                unsigned searchL, unsigned k,
                float alpha,
                unsigned insertBatchSize,
                unsigned queryBatchSize,
                unsigned deleteBatchSize,
                unsigned insertSearchL,
                float*    h_queries,
                unsigned  numQueries,
                unsigned* h_groundtruth,
                unsigned  gtK);
    ~FdaExecutor();

    void runWorkload(const std::vector<WorkloadEvent>& events);
    const FdaStats& getStats() const { return stats; }

    // Enable eager-merge mode (force merge at every insert→query / delete→
    // query transition). Off by default — the standard merge cadence is
    // M/T-driven. This is for recall-vs-cadence diagnostic experiments.
    void setEagerMergeOnTransition(bool on) { eagerMergeOnTransition = on; }

private:
    UnifiedIndex        idx;
    DeleteListWrapper   deleteList;

    unsigned ltiCount, tempCapacity, maxTemps, deleteListCapacity;
    unsigned searchL, k;
    float    alpha;
    unsigned insertBatchSize, queryBatchSize, deleteBatchSize;
    // Insert greedy uses a smaller worklist than query: inserts only need R=64
    // neighbor candidates, so searchL=50 is plenty (vs 100 for queries).
    // Each greedy step is O(searchL²) for the merge → halving searchL is ~4×
    // less work in the worklist merge, ~2× fewer iterations to converge.
    unsigned insertSearchL;

    // Eager-merge mode: when set, force a merge at every insert→query
    // and delete→query transition, regardless of M/T thresholds. The
    // current RW is sealed even if rwCount < T. Diagnostic / recall-
    // experiment knob.
    bool     eagerMergeOnTransition = false;

    // External-pointId ↔ internal-slot maps.
    // pointId comes from the workload's "point_id" field (caller's label).
    // slot is our system-assigned ID (the unified-graph slot index).
    // Predicted top-k results are slots → translated to pointIds for recall.
    std::unordered_map<unsigned, unsigned> pointIdToSlot;
    std::vector<unsigned>                  slotToPointId;   // indexed by slot

    GVEC*     d_allQueryVecs;
    unsigned  numAllQueries;
    unsigned* h_groundtruth;
    unsigned  gtK;

    // Insert / delete / search all share opStream — they touch the same graph
    // and the same delete bitvector, and per-event sync gives natural
    // workload-order serialization on a single stream. Merge runs on its own
    // mergeStream so it can overlap with the next insert burst while the
    // background mergeThread executes Phase A / B against the old buffers.
    cudaStream_t opStream;
    cudaStream_t mergeStream;

    // Note: "There is a single global write lock held while the LTI is swapped
    // out with the new LTI." Held by mergeNow during the atomic pointer swap.
    // Concurrent insert/search/delete must take it as a shared/read lock; the
    // public flush* / launchSearchBatch entry points below acquire it exclusively
    // to serialize against the swap. (Mutex chosen over shared_mutex because the
    // current executor is single-threaded on the driver side; upgrade to
    // shared_mutex when a multi-thread driver is added.)
    std::mutex globalIndexLock;

    // Background merge: when triggerMergeAsync() fires, mergeNow() runs on
    // this thread while the main thread continues. Joined before launching
    // the next merge (only one merge at a time) and at executor shutdown.
    std::thread mergeThread;

    SearchBuffers searchBuffers;

    std::vector<float>    pendingInsertVecs;
    std::vector<unsigned> pendingInsertIds;
    std::vector<unsigned> pendingQueryIdx;
    // Per-pending-query embedded GT vector (empty if event had no embedded GT).
    // Used by computeAllRecall to measure against per-query-time GT instead
    // of the static bin-file GT.
    std::vector<std::vector<unsigned>> pendingQueryGT;
    std::vector<unsigned> pendingDeleteIds;

    FdaStats stats;

    struct PendingRecallBatch {
        std::vector<unsigned> queryIdxs;
        std::vector<unsigned> predictedTopK;
        // Per-query embedded GT from the workload event (populated at query
        // time by the runner). When non-empty, recall is computed against
        // this instead of the static bin-file GT — matches how MVCC, BANG,
        // and DiskANN measure recall on these workloads.
        std::vector<std::vector<unsigned>> perQueryGT;
    };
    std::vector<PendingRecallBatch> deferredRecall;

    void flushInsertBatch();
    void flushDeleteBatch();
    void launchSearchBatch();
    void triggerMergeIfNeeded();
    void mergeNow();
    // Background merge: spawns mergeNow() on a separate std::thread so that
    // regular insert/search continue against the (still-valid) OLD index
    // pointers while Phase A+B run on mergeStream. The atomic pointer swap
    // at the end takes globalIndexLock and syncs all streams, so any
    // in-flight kernels finish before the OLD buffers are freed. RW index
    // operations continue undisturbed during the merge phase.
    void triggerMergeAsync();
    // Phase B per FreshDiskANN paper §5.3:
    // Re-insert every Temp/RW vertex (that isn't deleted) by running a fresh
    // GreedySearch + RobustPrune on the now-Phase-A-cleaned graph and
    // overwriting that vertex's neighbor list. Back-edges into LTI/Temp
    // accumulate in the delta buffer (folded in next merge); back-edges into
    // current RW (none at merge time) use locked append.
    void phaseBReinsertAllTempRW(struct InsertBatchBuffers* insertBufs);
    // Out-of-place variant: re-inserts every Temp/RW vertex into the new
    // graph pointed to by `view`. Reads vectors D→D from the OLD graph
    // (idx->d_graph) but greedy + RobustPrune target the new view. Back-edges
    // route through view's locks / delta buffers. Used by the out-of-place
    // merge path.
    void phaseBReinsertAllTempRWToView(struct InsertBatchBuffers* insertBufs,
                                       struct GraphView* view);
    void computeAllRecall();
};

} // namespace fda

#endif // FDA_EXECUTOR_H
