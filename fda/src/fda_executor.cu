#include "fda_executor.h"
#include "fda_insert.h"
#include "fda_search.h"
#include "fda_merge.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <climits>
#include <algorithm>

namespace fda {

// ============================================================================
// Stats
// ============================================================================
void FdaStats::print() const {
    printf("\n=== Naive FreshDiskANN (Unified) Statistics ===\n");
    printf("Inserts:           %lu (in %lu batches)\n", inserts, insertBatches);
    printf("Deletes:           %lu (in %lu batches)\n", deletes, deleteBatches);
    printf("Queries:           %lu (in %lu batches)\n", queries, queryBatches);
    printf("Merges fired:      %lu\n", mergesFired);
    printf("Wall time:         %.3f ms\n", totalWallMs);
    if (inserts > 0)
        printf("  Insert phase:    %.3f ms (%.1f ins/sec)\n",
               totalInsertMs, inserts / (totalInsertMs / 1000.0));
    if (queries > 0)
        printf("  Search phase:    %.3f ms (%.1f q/sec)\n",
               totalSearchMs, queries / (totalSearchMs / 1000.0));
    if (deletes > 0)
        printf("  Delete phase:    %.3f ms (%.1f d/sec)\n",
               totalDeleteMs, deletes / (totalDeleteMs / 1000.0));
    if (mergesFired > 0)
        printf("  Merge phase:     %.3f ms (avg %.1f ms/merge)\n",
               totalMergeMs, totalMergeMs / mergesFired);
    if (queries > 0)
        printf("  10-recall@10:    %.2f%% (compute: %.1f ms)\n",
               totalRecall10 / queries, recallComputeMs);
}

// ============================================================================
// Constructor / destructor
// ============================================================================
FdaExecutor::FdaExecutor(uint8_t* h_initialGraph, unsigned numBasePoints,
                         unsigned medoidId,
                         unsigned ltiCount_, unsigned tempCapacity_,
                         unsigned maxTemps_,
                         unsigned totalCapacity_,
                         unsigned deleteListCapacity_,
                         unsigned searchL_, unsigned k_, float alpha_,
                         unsigned insertBatchSize_, unsigned queryBatchSize_,
                         unsigned deleteBatchSize_,
                         unsigned insertSearchL_,
                         float* h_queries, unsigned numQueries,
                         unsigned* h_groundtruth_, unsigned gtK_)
    : ltiCount(ltiCount_), tempCapacity(tempCapacity_), maxTemps(maxTemps_),
      deleteListCapacity(deleteListCapacity_),
      searchL(searchL_), k(k_), alpha(alpha_),
      insertBatchSize(insertBatchSize_), queryBatchSize(queryBatchSize_),
      deleteBatchSize(deleteBatchSize_),
      insertSearchL(insertSearchL_ ? insertSearchL_ : searchL_),
      h_groundtruth(h_groundtruth_), gtK(gtK_),
      numAllQueries(numQueries)
{
    allocateUnifiedIndex(&idx, ltiCount, tempCapacity, maxTemps, totalCapacity_);
    loadLTIFromHost(&idx, h_initialGraph, numBasePoints, medoidId);

    // Override baseline_src's compile-time MEDOID with our runtime medoid so
    // initializeWorklist/initializeParents start greedy at the right vertex.
    setRuntimeMedoid(medoidId, 0);
    cudaDeviceSynchronize();

    // Initialize the slot→pointId map for LTI vertices (identity for slots [0, L))
    slotToPointId.resize(idx.capacity, UINT_MAX);
    for (unsigned i = 0; i < ltiCount; i++) {
        slotToPointId[i] = i;
        pointIdToSlot[i] = i;
    }

    allocateDeleteList(&deleteList, deleteListCapacity);

    if (h_queries && numQueries > 0) {
        gpuErrchk(cudaMalloc(&d_allQueryVecs, (size_t)numQueries * D * sizeof(GVEC)));
        std::vector<GVEC> _hq((size_t)numQueries * D);
        for (size_t _i=0;_i<(size_t)numQueries*D;_i++) _hq[_i]=(GVEC)h_queries[_i];
        gpuErrchk(cudaMemcpy(d_allQueryVecs, _hq.data(),
                             (size_t)numQueries * D * sizeof(GVEC),
                             cudaMemcpyHostToDevice));
    } else {
        d_allQueryVecs = nullptr;
    }

    gpuErrchk(cudaStreamCreate(&opStream));
    gpuErrchk(cudaStreamCreate(&mergeStream));

    allocateSearchBuffers(&searchBuffers, queryBatchSize);
}

FdaExecutor::~FdaExecutor() {
    // Wait for any in-flight background merge before tearing down state.
    if (mergeThread.joinable()) mergeThread.join();
    freeUnifiedIndex(&idx);
    freeDeleteList(&deleteList);
    if (d_allQueryVecs) cudaFree(d_allQueryVecs);
    freeSearchBuffers(&searchBuffers);
    cudaStreamDestroy(opStream);
    cudaStreamDestroy(mergeStream);
}

// ============================================================================
// Pipeline phases
// ============================================================================
void FdaExecutor::flushInsertBatch() {
    if (pendingInsertVecs.empty()) return;
    unsigned bs = pendingInsertVecs.size() / D;
    auto t0 = std::chrono::high_resolution_clock::now();

    // Serialize against merge swap. Released at end of scope; cudaStreamSynchronize
    // on opStream below ensures all GPU work finishes before we unlock and a
    // pending merge can swap idx.d_graph out from under us.
    std::lock_guard<std::mutex> indexLk(globalIndexLock);

    static InsertBatchBuffers insertBufs = {};
    if (!insertBufs.allocated) {
        allocateInsertBatchBuffers(&insertBufs, insertBatchSize);
    }

    static unsigned dbgFlushCount = 0;
    if (dbgFlushCount % 500 == 0) {
        printf("[flushInsert #%u] bs=%u rwCount=%u rwBase=%u numActiveTemps=%u\n",
               dbgFlushCount, bs, idx.rwCount, idx.rwBase, idx.numActiveTemps);
        fflush(stdout);
    }
    dbgFlushCount++;

    // Walk through pending inserts in workload order. Each insert event has a
    // pointId (caller's label). We assign internal slots sequentially in the
    // current RW range. When RW fills (rwCount==T), seal it and start the next
    // RW range. When numActiveTemps==M, trigger merge.
    unsigned offset = 0;
    while (offset < bs) {
        // Compute slots for as many pending inserts as fit in the current RW
        // (without crossing the seal boundary).
        unsigned slotsLeftInRW = tempCapacity - idx.rwCount;
        unsigned room = std::min({bs - offset, insertBatchSize, slotsLeftInRW});
        if (room == 0) {
            // RW full — seal and continue
            sealCurrentRW(&idx);
            if (idx.numActiveTemps >= maxTemps) triggerMergeAsync();
            continue;
        }

        // Assign slots [rwBase + rwCount, rwBase + rwCount + room) to these inserts.
        // Update both maps.
        std::vector<unsigned> chunkSlots(room);
        unsigned firstSlot = idx.rwBase + idx.rwCount;
        for (unsigned i = 0; i < room; i++) {
            unsigned slot = firstSlot + i;
            unsigned pid  = pendingInsertIds[offset + i];
            chunkSlots[i] = slot;
            pointIdToSlot[pid] = slot;
            slotToPointId[slot] = pid;
            // First insert into this RW range = the RW medoid
            if (idx.rwMedoid == UINT_MAX) idx.rwMedoid = slot;
        }

        const float* vp = pendingInsertVecs.data() + (size_t)offset * D;
        // LTI, RO Temp, and RW are independent graphs with their own
        // medoids. Insert goes into RW only — so the greedy + RobustPrune
        // step of insert must run over RW only, not LTI. Set the runtime
        // medoid to rwMedoid (= first slot of current RW, set above when
        // rwCount went 0→1). RW vertices' neighbor lists only reference
        // other RW slots (by construction of locked back-edges within RW),
        // so greedy from rwMedoid naturally stays in the RW region.
        // Major throughput win: greedy traverses at most T vertices instead
        // of all 500K of LTI per query.
        setRuntimeMedoid(idx.rwMedoid, opStream);

        unsigned inserted = insertBatchUnified(
            &idx, vp, chunkSlots.data(), room, insertSearchL, alpha,
            &insertBufs, /*d_deleted=*/nullptr, opStream);

        if (inserted == 0) {
            fprintf(stderr, "[flushInsertBatch] insertBatchUnified returned 0; bailing\n");
            break;
        }
        offset += inserted;

        // If this chunk filled the RW, seal it and possibly merge
        if (idx.rwCount >= tempCapacity) {
            sealCurrentRW(&idx);
            if (idx.numActiveTemps >= maxTemps) triggerMergeAsync();
        }
    }
    gpuErrchk(cudaStreamSynchronize(opStream));

    auto t1 = std::chrono::high_resolution_clock::now();
    stats.totalInsertMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    { double _ms = std::chrono::duration<double, std::milli>(t1 - t0).count();  /* MARKER:insertBatches */
      printf("[PSTEP] op=%s idx=%lu n=%u ms=%.4f thr=%.2f\n", "insert", (unsigned long)stats.insertBatches, (unsigned)bs, _ms, _ms>0?(unsigned)bs/(_ms/1000.0):0.0); fflush(stdout); }
    stats.inserts += bs;
    stats.insertBatches++;

    pendingInsertVecs.clear();
    pendingInsertIds.clear();
}

void FdaExecutor::flushDeleteBatch() {
    if (pendingDeleteIds.empty()) return;
    auto t0 = std::chrono::high_resolution_clock::now();

    // Serialize against merge swap; released with opStream-sync below.
    std::lock_guard<std::mutex> indexLk(globalIndexLock);

    // Translate workload pointIds → internal slots before marking deletes
    std::vector<unsigned> slots;
    slots.reserve(pendingDeleteIds.size());
    for (unsigned pid : pendingDeleteIds) {
        auto it = pointIdToSlot.find(pid);
        if (it != pointIdToSlot.end()) slots.push_back(it->second);
        // else: unknown pointId (workload bug), silently drop
    }
    unsigned bs = slots.size();
    if (bs == 0) { pendingDeleteIds.clear(); return; }

    unsigned* d_ids;
    gpuErrchk(cudaMalloc(&d_ids, bs * sizeof(unsigned)));
    gpuErrchk(cudaMemcpyAsync(d_ids, slots.data(),
                              bs * sizeof(unsigned),
                              cudaMemcpyHostToDevice, opStream));
    batchMarkDeleted(&deleteList, d_ids, bs, opStream);
    gpuErrchk(cudaStreamSynchronize(opStream));
    cudaFree(d_ids);

    auto t1 = std::chrono::high_resolution_clock::now();
    stats.totalDeleteMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    { double _ms = std::chrono::duration<double, std::milli>(t1 - t0).count();  /* MARKER:deleteBatches */
      printf("[PSTEP] op=%s idx=%lu n=%u ms=%.4f thr=%.2f\n", "delete", (unsigned long)stats.deleteBatches, (unsigned)bs, _ms, _ms>0?(unsigned)bs/(_ms/1000.0):0.0); fflush(stdout); }
    stats.deletes += bs;
    stats.deleteBatches++;
    pendingDeleteIds.clear();

    // Deletes contribute to S = L + M·T + D, so re-check the merge trigger.
    triggerMergeIfNeeded();
}

void FdaExecutor::launchSearchBatch() {
    if (pendingQueryIdx.empty()) return;
    unsigned bs = pendingQueryIdx.size();
    auto t0 = std::chrono::high_resolution_clock::now();

    // Serialize against merge swap; released with opStream-sync below.
    std::lock_guard<std::mutex> indexLk(globalIndexLock);

    // Gather query vectors
    for (unsigned i = 0; i < bs; i++) {
        unsigned qid = pendingQueryIdx[i];
        gpuErrchk(cudaMemcpyAsync(
            searchBuffers.d_queryVecs + (size_t)i * D,
            d_allQueryVecs + (size_t)qid * D,
            D * sizeof(GVEC),
            cudaMemcpyDeviceToDevice, opStream));
    }

    searchUnified(&idx, &deleteList,
                  searchBuffers.d_queryVecs, bs, k, searchL,
                  &searchBuffers, opStream);

    gpuErrchk(cudaMemcpyAsync(searchBuffers.h_topkIds, searchBuffers.d_topkIds,
                              (size_t)bs * k * sizeof(unsigned),
                              cudaMemcpyDeviceToHost, opStream));
    gpuErrchk(cudaStreamSynchronize(opStream));

    // DEBUG: first batch — print first query's predicted (translated to
    // workload pointIds) vs GT
    if (stats.queryBatches == 0 && h_groundtruth) {
        unsigned qIdx0 = pendingQueryIdx[0];
        printf("[DEBUG] First query @ pool_idx=%u\n", qIdx0);
        printf("  predicted top-%u (pointIds): ", k);
        for (unsigned i = 0; i < k; i++) {
            unsigned slot = searchBuffers.h_topkIds[i];
            if (slot == 0xFFFFFFFFu) { printf("- "); continue; }
            unsigned pid = (slot < slotToPointId.size()) ? slotToPointId[slot] : 0xFFFFFFFFu;
            if (pid == 0xFFFFFFFFu) printf("?slot%u ", slot);
            else                    printf("%u ", pid);
        }
        printf("\n  groundtruth top-%u: ", k);
        const unsigned* gt = h_groundtruth + (size_t)qIdx0 * gtK;
        for (unsigned i = 0; i < k; i++) printf("%u ", gt[i]);
        printf("\n");
        fflush(stdout);
    }

    PendingRecallBatch pb;
    pb.queryIdxs = pendingQueryIdx;
    pb.predictedTopK.assign(searchBuffers.h_topkIds,
                             searchBuffers.h_topkIds + (size_t)bs * k);
    pb.perQueryGT = pendingQueryGT;   // ← per-query-time embedded GT for recall
    deferredRecall.push_back(std::move(pb));

    auto t1 = std::chrono::high_resolution_clock::now();
    stats.totalSearchMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    { double _ms = std::chrono::duration<double, std::milli>(t1 - t0).count();  /* MARKER:queryBatches */
      printf("[PSTEP] op=%s idx=%lu n=%u ms=%.4f thr=%.2f\n", "search", (unsigned long)stats.queryBatches, (unsigned)bs, _ms, _ms>0?(unsigned)bs/(_ms/1000.0):0.0); fflush(stdout); }
    stats.queries += bs;
    stats.queryBatches++;
    pendingQueryIdx.clear();
    pendingQueryGT.clear();
}

void FdaExecutor::triggerMergeIfNeeded() {
    // Parameter S = L + M·T + D — the maximum state the system can hold
    // (original LTI + max Temp storage + max delete list). Merge fires when
    // we reach S OR when any of its components saturates individually:
    //   - numActiveTemps == M   (Temp slots full; can't seal another RW)
    //   - deleteList.count == D (DeleteList full; can't tombstone more)
    //
    // Merge fires on either of the saturation conditions OR when the
    // combined live state reaches the threshold S = L + M*T + D.

    unsigned long _L  = ltiCount;
    unsigned long _MT = (unsigned long)maxTemps * tempCapacity;
    unsigned long _D  = deleteListCapacity;
    unsigned long S   = _L + _MT + _D;

    unsigned long currLive = idx.ltiCount
                           + (unsigned long)idx.numActiveTemps * idx.tempCapacity
                           + idx.rwCount
                           + deleteList.count;

    bool sReached     = (currLive >= S);
    bool tempsFull    = (idx.numActiveTemps >= maxTemps);
    bool deletesFull  = (deleteList.count   >= deleteListCapacity);

    if (sReached || tempsFull || deletesFull) triggerMergeAsync();
}

void FdaExecutor::triggerMergeAsync() {
    // FDA_SKIP_MERGE: OOM-mode for >GPU-memory datasets (e.g. 100M). The out-of-place
    // merge below allocates a 2nd unified graph (~2x), which OOMs at 100M. Skipping it keeps
    // a single graph: temps accumulate in-place and deletes stay tombstone-only (no
    // consolidation, like svf repair-off). Size --m/--t so M*T >= numInserts so temps never
    // overflow. Recall may drift on delete-heavy workloads; the run completes.
    static const bool _skipMerge = (getenv("FDA_SKIP_MERGE") != nullptr);
    if (_skipMerge) { printf("[Merge] SKIPPED (FDA_SKIP_MERGE) sealedTemps=%u/%u\n", idx.numActiveTemps, maxTemps); fflush(stdout); return; }
    // Wait for prior merge to finish if one is still running. CRITICAL: this is
    // called from inside flushInsertBatch / flushDeleteBatch / launchSearchBatch
    // which hold globalIndexLock. A previous merge thread may be parked at its
    // swap step waiting on globalIndexLock, so we MUST release the lock during
    // the join to avoid deadlock. Re-acquire after join so the caller's
    // lock_guard can release at scope exit as usual.
    if (mergeThread.joinable()) {
        globalIndexLock.unlock();
        mergeThread.join();
        globalIndexLock.lock();
    }
    // Spawn the merge in a background std::thread. mergeNow() itself takes
    // the global lock around the pointer swap (the only state mutation
    // visible to other threads), and uses cudaStreamSynchronize on the
    // insert/search streams inside that lock so any in-flight kernels
    // finish before old buffers are freed. Regular searches and RW
    // operations continue undisturbed during the merge phase.
    mergeThread = std::thread([this]{ this->mergeNow(); });
}

void FdaExecutor::mergeNow() {
    auto t0 = std::chrono::high_resolution_clock::now();
    printf("[Merge] firing (out-of-place) — sealedTemps=%u/%u deletes=%u rwCount=%u\n",
           idx.numActiveTemps, maxTemps, deleteList.count, idx.rwCount);
    fflush(stdout);

    // Race-prevention: opStream (insert/search/delete) may still hold queued
    // work that reads/writes idx.d_graph. Drain it before staging the merge.
    gpuErrchk(cudaStreamSynchronize(opStream));

    // ── Allocate new-LTI scratch ("delta memory"): parallel graph + locks ──
    // Spec: "Merge Kernel(LTI and RO Indexes as input, New LTI index as
    // output)" — only LTI + RO Temps are absorbed. The current RW index
    // continues undisturbed across the merge.
    unsigned mergedSlots  = idx.numActiveTemps * idx.tempCapacity;   // Temps only
    unsigned newLtiCount  = idx.ltiCount + mergedSlots;              // = old idx.rwBase
    unsigned newCapacity  = idx.capacity;        // keep the same total ceiling
    if (newCapacity < newLtiCount) newCapacity = newLtiCount;
    // Snapshot the RW window BEFORE Phase A/B so we can copy it verbatim and
    // restore it in the swap.
    unsigned savedRwBase    = idx.rwBase;
    unsigned savedRwCount   = idx.rwCount;
    unsigned savedRwMedoid  = idx.rwMedoid;

    uint8_t*  new_d_graph        = nullptr;
    uint8_t*  new_d_locks        = nullptr;
    allocateDeltaMemory(&new_d_graph, &new_d_locks, newCapacity);

    GraphView newView;
    newView.d_graph        = new_d_graph;
    newView.d_locks        = new_d_locks;
    newView.immutableEnd   = newLtiCount;   // after Phase A+B, everything is LTI

    // ── Phase A out-of-place ───────────────────────────────────────────────
    // Read OLD graph + OLD deltas; write the consolidated neighbor lists into
    // the NEW graph at the SAME slot index. Deleted vertices contribute
    // nothing to the new graph.
    {
        unsigned immutableEnd = idx.rwBase;   // [0, rwBase) = LTI ∪ sealed Temps
        if (immutableEnd > 0) {
            mergePhaseAOutOfPlaceKernel<<<immutableEnd, BLOCK_D, 0, mergeStream>>>(
                idx.d_graph,
                new_d_graph,
                deleteList.d_bits,
                immutableEnd,
                alpha);
            gpuErrchk(cudaGetLastError());
        }
    }

    // ── Phase B ────────────────────────────────────────────────────────────
    // Fresh re-insert of every Temp/RW vertex into the NEW graph. Reads
    // vectors D→D from the OLD graph but all greedy / RobustPrune work
    // happens against the new view.
    static InsertBatchBuffers insertBufs = {};
    if (!insertBufs.allocated)
        allocateInsertBatchBuffers(&insertBufs, insertBatchSize);
    phaseBReinsertAllTempRWToView(&insertBufs, &newView);

    // ── Preserve current RW across the swap ────────────────────────────────
    // Spec: "RW index operations continue as before (and also remain
    // undisturbed during merge phase)." We copy the RW vertex entries (vector
    // + degree + adjacency) verbatim from the old graph into the new graph at
    // the SAME slot positions. By construction, RW insert greedy stays inside
    // RW, so RW adjacency lists only reference other RW slots — those slot
    // IDs remain valid in the new graph because old idx.rwBase equals
    // newLtiCount, so the RW window sits exactly where the new RW region
    // begins.
    if (savedRwCount > 0) {
        const size_t startByte = (size_t)savedRwBase * graphEntrySize;
        const size_t nBytes    = (size_t)savedRwCount * graphEntrySize;
        gpuErrchk(cudaMemcpyAsync(new_d_graph + startByte,
                                   idx.d_graph + startByte,
                                   nBytes,
                                   cudaMemcpyDeviceToDevice,
                                   mergeStream));
    }

    gpuErrchk(cudaStreamSynchronize(mergeStream));

    // ── Atomic swap under the global write lock ────────────────────────────
    // Note: "There is a single global write lock held while the LTI is
    // swapped out with the new LTI."
    uint8_t*  old_d_graph        = nullptr;
    uint8_t*  old_d_locks        = nullptr;
    {
        std::lock_guard<std::mutex> lk(globalIndexLock);
        old_d_graph        = idx.d_graph;
        old_d_locks        = idx.d_locks;

        idx.d_graph        = new_d_graph;
        idx.d_locks        = new_d_locks;

        idx.ltiCount       = newLtiCount;        // = old idx.rwBase
        idx.numActiveTemps = 0;                  // Temps absorbed into new LTI
        // Preserve RW continuity per spec — RW operations resume on the same
        // window. newLtiCount == savedRwBase so the RW range is unchanged.
        idx.rwBase         = newLtiCount;
        idx.rwCount        = savedRwCount;
        idx.rwMedoid       = savedRwMedoid;
        for (unsigned i = 0; i < 64; i++) idx.tempMedoids[i] = UINT_MAX;
    }

    // Clear the delete list now that consolidation is final.
    clearDeleteList(&deleteList, mergeStream);
    gpuErrchk(cudaStreamSynchronize(mergeStream));

    // ── Free the old buffers ───────────────────────────────────────────────
    freeDeltaMemory(old_d_graph, old_d_locks);

    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "[mergeNow] post-merge cudaDeviceSynchronize: %s\n",
                cudaGetErrorString(syncErr));
        fflush(stderr);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    stats.totalMergeMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    stats.mergesFired++;
    printf("[Merge] done in %.1f ms\n",
           std::chrono::duration<double, std::milli>(t1 - t0).count());
    fflush(stdout);
}

// ============================================================================
// Event loop
// ============================================================================
void FdaExecutor::runWorkload(const std::vector<WorkloadEvent>& events) {
    auto wallStart = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < events.size(); i++) {
        const WorkloadEvent& e = events[i];
        switch (e.type) {
            case EVENT_INSERT: {
                if (e.vector.size() < D) break;
                if (!pendingQueryIdx.empty()) launchSearchBatch();
                size_t off = pendingInsertVecs.size();
                pendingInsertVecs.resize(off + D);
                std::memcpy(pendingInsertVecs.data() + off,
                            e.vector.data(), D * sizeof(float));
                pendingInsertIds.push_back(e.pointId);
                if (pendingInsertVecs.size() / D >= insertBatchSize)
                    flushInsertBatch();
                break;
            }
            case EVENT_QUERY: {
                if (!pendingInsertVecs.empty()) flushInsertBatch();
                if (!pendingDeleteIds.empty()) flushDeleteBatch();
                // Eager-merge: if there's any uncommitted RW state (current
                // partial RW with rwCount > 0, OR sealed Temps that haven't
                // triggered the M threshold), force a merge before any
                // queries see the database. mergeNow's Phase B already
                // handles a partial RW correctly (phaseBEnd uses rwCount
                // directly, not tempCapacity), so we don't pre-seal — that
                // would waste slot range up to tempCapacity per partial seal.
                if (eagerMergeOnTransition &&
                    (idx.numActiveTemps > 0 || idx.rwCount > 0)) {
                    mergeNow();
                }
                pendingQueryIdx.push_back(e.queryId);
                // Capture embedded per-query GT (empty if event had none).
                pendingQueryGT.push_back(e.groundtruth);
                if (pendingQueryIdx.size() >= queryBatchSize) launchSearchBatch();
                break;
            }
            case EVENT_DELETE: {
                if (!pendingInsertVecs.empty()) flushInsertBatch();
                if (!pendingQueryIdx.empty()) launchSearchBatch();
                pendingDeleteIds.push_back(e.pointId);
                if (pendingDeleteIds.size() >= deleteBatchSize) flushDeleteBatch();
                break;
            }
            default: break;
        }
    }

    flushInsertBatch();
    flushDeleteBatch();
    launchSearchBatch();

    // Wait for any pending background merge to complete before tallying stats.
    if (mergeThread.joinable()) mergeThread.join();

    auto wallEnd = std::chrono::high_resolution_clock::now();
    stats.totalWallMs = std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();

    computeAllRecall();

    // Padding diagnostic: report whether top-k extraction had to pad with
    // sentinels (= worklist ran out of non-deleted candidates).
    reportPaddingDiagnostic((unsigned)stats.queries);
}

// ============================================================================
// Phase B helper kernel: copy each slot's vector portion (D floats) from
// the graph into a contiguous d_vectors buffer for the insert pipeline.
// One block per slot; threads cooperate on the D-element copy.
// ============================================================================
__global__ void gatherVectorsFromGraphKernel(
    const uint8_t* d_graph,
    const unsigned* d_slots,
    unsigned batchSize,
    GVEC* d_vectorsOut)
{
    unsigned i = blockIdx.x;
    if (i >= batchSize) return;
    unsigned slot = d_slots[i];
    const GVEC* src = (const GVEC*)(d_graph + (size_t)slot * graphEntrySize);
    GVEC* dst = d_vectorsOut + (size_t)i * D;
    for (unsigned d = threadIdx.x; d < D; d += blockDim.x) dst[d] = src[d];
}

void FdaExecutor::phaseBReinsertAllTempRW(InsertBatchBuffers* bufs)
{
    // Iterate over every slot in the immutable region beyond the original LTI:
    //   [originalLtiCount, originalLtiCount + numActiveTemps*T + rwCount)
    // and re-insert it via the insert pipeline (using D→D vector copy from
    // the existing graph). Skip slots whose vertex was deleted.
    //
    // Note: this is the IN-PLACE Phase B variant. An out-of-place version
    // writes to "delta memory" then does an atomic swap. For sequential
    // test workloads (no concurrent searches during merge), in-place is
    // functionally equivalent and avoids a second graph allocation.

    unsigned phaseBStart = idx.ltiCount;
    unsigned phaseBEnd   = idx.ltiCount + idx.numActiveTemps * idx.tempCapacity + idx.rwCount;
    if (phaseBEnd <= phaseBStart) return;

    unsigned totalSlots = phaseBEnd - phaseBStart;
    printf("[Phase B] re-inserting slots [%u, %u) — %u total\n",
           phaseBStart, phaseBEnd, totalSlots);
    fflush(stdout);

    // Host buffer for per-batch deletion-filtered slot lists
    std::vector<unsigned> batchSlots;
    batchSlots.reserve(insertBatchSize);

    // Host-side delete-bit check: copy the delete bitvector to host once
    std::vector<unsigned> h_deleteBits;
    if (deleteList.count > 0) {
        unsigned numWords = (deleteList.capacity + 31) / 32;
        h_deleteBits.resize(numWords);
        gpuErrchk(cudaMemcpy(h_deleteBits.data(), deleteList.d_bits,
                              (size_t)numWords * sizeof(unsigned),
                              cudaMemcpyDeviceToHost));
    }
    auto isDeleted = [&](unsigned slot) -> bool {
        if (h_deleteBits.empty()) return false;
        unsigned word = slot / 32;
        unsigned mask = 1u << (slot % 32);
        return (word < h_deleteBits.size()) && (h_deleteBits[word] & mask);
    };

    unsigned processed = 0;
    unsigned reInserted = 0;
    for (unsigned slot = phaseBStart; slot < phaseBEnd; slot++) {
        if (!isDeleted(slot)) batchSlots.push_back(slot);

        // Flush when batch is full OR at the very end
        bool atEnd = (slot + 1 == phaseBEnd);
        if (batchSlots.size() >= insertBatchSize || (atEnd && !batchSlots.empty())) {
            unsigned bs = batchSlots.size();
            // Copy slot IDs to d_pointIds via the same h_pointIds buffer
            for (unsigned i = 0; i < bs; i++) bufs->h_pointIds[i] = batchSlots[i];
            gpuErrchk(cudaMemcpyAsync(bufs->d_pointIds, bufs->h_pointIds,
                                      (size_t)bs * sizeof(unsigned),
                                      cudaMemcpyHostToDevice, mergeStream));

            // D→D gather: pull each slot's vector into bufs->d_vectors
            gatherVectorsFromGraphKernel<<<bs, BLOCK_D, 0, mergeStream>>>(
                idx.d_graph, bufs->d_pointIds, bs, bufs->d_vectors);

            // Run the regular insert pipeline with d_vectorsPreLoaded=true.
            // Greedy runs on idx.d_graph (which is the Phase-A-cleaned graph),
            // RobustPrune picks new R neighbors, writes back to the same slot,
            // back-edges routed by appendReverseEdgesRoutedKernel.
            unsigned inserted = insertBatchUnified(
                &idx,
                /*h_newVectors=*/ nullptr,
                batchSlots.data(),
                bs, insertSearchL, alpha,
                bufs,
                /*d_deleted=*/ nullptr,      // see workaround note above
                mergeStream,
                /*d_vectorsPreLoaded=*/ true);
            reInserted += inserted;
            batchSlots.clear();
        }
        processed++;
    }
    gpuErrchk(cudaStreamSynchronize(mergeStream));
    printf("[Phase B] done — processed=%u re-inserted=%u (skipped %u deleted)\n",
           processed, reInserted, processed - reInserted);
    fflush(stdout);
}

void FdaExecutor::phaseBReinsertAllTempRWToView(InsertBatchBuffers* bufs,
                                                GraphView* view)
{
    // Out-of-place Phase B: re-insert every sealed Temp vertex into the NEW
    // graph. Per spec, the current RW is NOT absorbed — it's copied verbatim
    // by mergeNow() so RW operations continue undisturbed.
    //
    // Vectors are gathered D→D from the OLD graph (idx.d_graph) — they are
    // identical to what the new graph stores at the same slot (Phase A copied
    // vectors verbatim), but the OLD graph is the authoritative source.

    unsigned phaseBStart = idx.ltiCount;
    unsigned phaseBEnd   = idx.ltiCount + idx.numActiveTemps * idx.tempCapacity; // Temps only
    if (phaseBEnd <= phaseBStart) return;

    unsigned totalSlots = phaseBEnd - phaseBStart;
    printf("[Phase B → new] re-inserting slots [%u, %u) — %u total\n",
           phaseBStart, phaseBEnd, totalSlots);
    fflush(stdout);

    // CRITICAL: Phase B re-inserts old-RW/Temp vertices INTO the new LTI
    // graph (which Phase A has already populated with the consolidated
    // [0, ltiCount) region). Greedy must run from the new LTI's medoid —
    // NOT from rwMedoid (which would have been set by the regular insert
    // path on the last batch before merge). Setting it explicitly here.
    setRuntimeMedoid(idx.ltiMedoid, mergeStream);

    std::vector<unsigned> batchSlots;
    batchSlots.reserve(insertBatchSize);

    std::vector<unsigned> h_deleteBits;
    if (deleteList.count > 0) {
        unsigned numWords = (deleteList.capacity + 31) / 32;
        h_deleteBits.resize(numWords);
        gpuErrchk(cudaMemcpy(h_deleteBits.data(), deleteList.d_bits,
                              (size_t)numWords * sizeof(unsigned),
                              cudaMemcpyDeviceToHost));
    }
    auto isDeleted = [&](unsigned slot) -> bool {
        if (h_deleteBits.empty()) return false;
        unsigned word = slot / 32;
        unsigned mask = 1u << (slot % 32);
        return (word < h_deleteBits.size()) && (h_deleteBits[word] & mask);
    };

    unsigned processed = 0;
    unsigned reInserted = 0;
    for (unsigned slot = phaseBStart; slot < phaseBEnd; slot++) {
        if (!isDeleted(slot)) batchSlots.push_back(slot);

        bool atEnd = (slot + 1 == phaseBEnd);
        if (batchSlots.size() >= insertBatchSize || (atEnd && !batchSlots.empty())) {
            unsigned bs = batchSlots.size();
            for (unsigned i = 0; i < bs; i++) bufs->h_pointIds[i] = batchSlots[i];
            gpuErrchk(cudaMemcpyAsync(bufs->d_pointIds, bufs->h_pointIds,
                                      (size_t)bs * sizeof(unsigned),
                                      cudaMemcpyHostToDevice, mergeStream));

            // D→D gather: source is the OLD graph (idx.d_graph). The new
            // graph will be populated by insertBatchToView writing into view.
            gatherVectorsFromGraphKernel<<<bs, BLOCK_D, 0, mergeStream>>>(
                idx.d_graph, bufs->d_pointIds, bs, bufs->d_vectors);

            unsigned inserted = insertBatchToView(
                &idx,
                view,
                /*h_newVectors=*/ nullptr,
                batchSlots.data(),
                bs, insertSearchL, alpha,
                bufs,
                /*d_deleted=*/ nullptr,
                mergeStream,
                /*d_vectorsPreLoaded=*/ true);
            reInserted += inserted;
            batchSlots.clear();
        }
        processed++;
    }
    gpuErrchk(cudaStreamSynchronize(mergeStream));
    printf("[Phase B → new] done — processed=%u re-inserted=%u (skipped %u deleted)\n",
           processed, reInserted, processed - reInserted);
    fflush(stdout);
}

void FdaExecutor::computeAllRecall() {
    if (!h_groundtruth) return;
    auto t0 = std::chrono::high_resolution_clock::now();
    double total = 0.0;
    unsigned long count = 0;

    // Per-query profiling: split GT IDs by region.
    //   LTI region: ID < ltiCount
    //   RW region:  ID >= ltiCount
    // Tracks how many GT entries are LTI vs RW, and how many of each we find.
    double sumGtLti = 0, sumGtRw = 0;
    double sumHitLti = 0, sumHitRw = 0;

    // Per-batch evolution: average recall for each query batch.
    printf("\n=== Per-batch recall (in workload order) ===\n");
    printf("batch  | avg recall@10 | LTI in GT | RW in GT | LTI hits | RW hits\n");
    fflush(stdout);

    unsigned batchIdx = 0;
    for (const auto& pb : deferredRecall) {
        unsigned bs = pb.queryIdxs.size();
        double batchTotal = 0.0;
        double batchGtLti = 0, batchGtRw = 0;
        double batchHitLti = 0, batchHitRw = 0;

        for (unsigned q = 0; q < bs; q++) {
            unsigned qIdx = pb.queryIdxs[q];
            // Prefer the per-query-time embedded GT (from the workload event)
            // over the static bin-file GT. The embedded GT is what other
            // systems (MVCC, BANG, CPU DiskANN) measure against for these
            // workloads — it's filtered to vertices that are actually in the
            // database when the query fires. The bin-file GT is final-state
            // GT against the full SIFT-1M dataset, which includes vertices
            // not yet inserted at any given query's time.
            const unsigned* gt;
            unsigned gtLen;
            if (q < pb.perQueryGT.size() && !pb.perQueryGT[q].empty()) {
                gt    = pb.perQueryGT[q].data();
                gtLen = (unsigned)pb.perQueryGT[q].size();
                if (gtLen > k) gtLen = k;   // 10-recall@10
            } else {
                gt    = h_groundtruth + (size_t)qIdx * gtK;
                gtLen = k;
            }
            const unsigned* pred_slots = pb.predictedTopK.data() + (size_t)q * k;

            // Translate predicted slots → pointIds before comparing with GT
            unsigned predPointIds[64];   // k ≤ 64 in practice
            for (unsigned i = 0; i < k; i++) {
                unsigned s = pred_slots[i];
                if (s == 0xFFFFFFFFu || s >= slotToPointId.size()) {
                    predPointIds[i] = 0xFFFFFFFFu;
                } else {
                    predPointIds[i] = slotToPointId[s];
                }
            }

            unsigned hits = 0;
            unsigned gtLti = 0, gtRw = 0;
            unsigned hitLti = 0, hitRw = 0;

            // GT classification: pointId < original-L means it's an original
            // LTI vertex (loaded from the .out file). Other pointIds are
            // workload-inserted vertices. Use ORIGINAL ltiCount (the L the
            // workload sees), not idx.ltiCount which may have grown post-merge.
            for (unsigned j = 0; j < gtLen; j++) {
                if (gt[j] < this->ltiCount) gtLti++;
                else                        gtRw++;
            }
            for (unsigned i = 0; i < k; i++) {
                unsigned p = predPointIds[i];
                if (p == 0xFFFFFFFFu) continue;
                for (unsigned j = 0; j < gtLen; j++) {
                    if (p == gt[j]) {
                        hits++;
                        if (p < this->ltiCount) hitLti++;
                        else                    hitRw++;
                        break;
                    }
                }
            }
            double r = 100.0 * hits / (double)k;
            total += r;
            batchTotal += r;
            sumGtLti += gtLti;  sumGtRw += gtRw;
            sumHitLti += hitLti; sumHitRw += hitRw;
            batchGtLti += gtLti; batchGtRw += gtRw;
            batchHitLti += hitLti; batchHitRw += hitRw;
            count++;
        }
        printf("%3u    | %12.2f%% | %8.2f  | %7.2f  | %7.2f  | %6.2f\n",
               batchIdx,
               batchTotal / bs,
               batchGtLti / bs, batchGtRw / bs,
               batchHitLti / bs, batchHitRw / bs);
        batchIdx++;
    }
    fflush(stdout);

    if (count > 0) {
        stats.totalRecall10 = total;
        printf("\n=== Profile summary (over all %lu queries) ===\n", count);
        printf("Avg GT composition: LTI=%.2f/10  RW=%.2f/10\n",
               sumGtLti / count, sumGtRw / count);
        printf("Avg hits:          LTI=%.2f/%.2f (%.1f%%)  RW=%.2f/%.2f (%.1f%%)\n",
               sumHitLti / count, sumGtLti / count,
               (sumGtLti > 0 ? 100.0 * sumHitLti / sumGtLti : 0.0),
               sumHitRw / count, sumGtRw / count,
               (sumGtRw > 0 ? 100.0 * sumHitRw / sumGtRw : 0.0));
        fflush(stdout);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    stats.recallComputeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

} // namespace fda
