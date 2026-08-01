#include "fda2_executor.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>

namespace fda2 {

void Stats::print() const {
    printf("\n=== FreshDiskANN Sequential-Model Statistics ===\n");
    printf("Inserts: %lu (%lu batches)\n", inserts, insertBatches);
    printf("Deletes: %lu (%lu batches)\n", deletes, deleteBatches);
    printf("Queries: %lu (%lu batches)\n", queries, queryBatches);
    printf("Wall time:    %.3f ms\n", wallMs);
    if (inserts) printf("  Insert: %.3f ms (%.1f ins/sec)\n", insertMs, inserts / (insertMs/1000.0));
    if (deletes) printf("  Delete: %.3f ms (%.1f del/sec)\n", deleteMs, deletes / (deleteMs/1000.0));
    if (queries) printf("  Search: %.3f ms (%.1f q/sec)\n", searchMs, queries / (searchMs/1000.0));
    if (queries) printf("  10-recall@10: %.2f%% (compute %.1f ms)\n", recall10 / queries, recallComputeMs);
}

Executor::Executor(uint8_t* h_initialGraph, unsigned numBasePoints_, unsigned medoidId,
                   unsigned capacity,
                   unsigned searchL_, unsigned k_, float alpha_,
                   unsigned insertBatchSize_, unsigned queryBatchSize_, unsigned deleteBatchSize_,
                   float* h_queries, unsigned numQueries,
                   unsigned* h_groundtruth_, unsigned gtK_)
    : numBasePoints(numBasePoints_),
      searchL(searchL_), k(k_), alpha(alpha_),
      insertBatchSize(insertBatchSize_), queryBatchSize(queryBatchSize_),
      deleteBatchSize(deleteBatchSize_),
      numAllQueries(numQueries), h_groundtruth(h_groundtruth_), gtK(gtK_)
{
    // d_dirtyList is shared scratch holding the set of UPDATED vertices.
    // INSERT touches ≤ insertBatchSize*R reverse-edge targets, but the exact
    // Algorithm-5 DELETE updates both p's in-neighbors AND candidate vertices,
    // whose distinct count can exceed deleteBatchSize*R. Cap = capacity (a node
    // can be dirty at most once) covers both ops and can never overflow.
    unsigned dirtyCap = capacity;
    allocateUnifiedGraph(&g, capacity, dirtyCap);
    loadGraphFromHost(&g, h_initialGraph, numBasePoints, medoidId);

    setRuntimeMedoid(medoidId, 0);
    cudaDeviceSynchronize();

    slotToPointId.resize(capacity, UINT_MAX);
    for (unsigned i = 0; i < numBasePoints; i++) {
        slotToPointId[i] = i;
        pointIdToSlot[i] = i;
    }
    hostDeleted.assign(capacity, 0);

    if (h_queries && numQueries > 0) {
        gpuErrchk(cudaMalloc(&d_allQueryVecs, (size_t)numQueries * D * sizeof(float)));
        gpuErrchk(cudaMemcpy(d_allQueryVecs, h_queries,
                             (size_t)numQueries * D * sizeof(float), cudaMemcpyHostToDevice));
    } else d_allQueryVecs = nullptr;

    gpuErrchk(cudaStreamCreate(&stream));
    allocateInsertBuffers(&insertBufs, insertBatchSize);
    allocateDeleteBuffers(&deleteBufs, deleteBatchSize);
    allocateSearchBuffers(&searchBufs, queryBatchSize);
}

Executor::~Executor() {
    freeUnifiedGraph(&g);
    if (d_allQueryVecs) cudaFree(d_allQueryVecs);
    freeInsertBuffers(&insertBufs);
    freeDeleteBuffers(&deleteBufs);
    freeSearchBuffers(&searchBufs);
    cudaStreamDestroy(stream);
}

// ---------------------------------------------------------------------------
// INSERT flush
// ---------------------------------------------------------------------------
void Executor::flushInserts() {
    if (pendInsVecs.empty()) return;
    unsigned bs = pendInsVecs.size() / D;
    auto t0 = std::chrono::high_resolution_clock::now();

    // Assign internal slots, in chunks of insertBatchSize.
    unsigned offset = 0;
    while (offset < bs) {
        unsigned room = bs - offset;
        if (room > insertBatchSize) room = insertBatchSize;
        if (g.numPoints + room > g.capacity) {
            fprintf(stderr, "[flushInserts] capacity %u exhausted (numPoints=%u, want +%u)\n",
                    g.capacity, g.numPoints, room);
            room = (g.capacity > g.numPoints) ? (g.capacity - g.numPoints) : 0;
            if (room == 0) break;
        }
        std::vector<unsigned> slots(room);
        for (unsigned i = 0; i < room; i++) {
            unsigned slot = g.numPoints + i;
            unsigned pid  = pendInsIds[offset + i];
            slots[i] = slot;
            pointIdToSlot[pid] = slot;
            slotToPointId[slot] = pid;
        }
        const float* vp = pendInsVecs.data() + (size_t)offset * D;
        insertBatch(&g, vp, slots.data(), room, searchL, alpha, &insertBufs, stream);
        g.numPoints += room;
        offset += room;
    }
    gpuErrchk(cudaStreamSynchronize(stream));   // phase barrier

    auto t1 = std::chrono::high_resolution_clock::now();
    stats.insertMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    { double _ms = std::chrono::duration<double, std::milli>(t1 - t0).count();  /* MARKER:insertBatches */
      printf("[PSTEP] op=%s idx=%lu n=%u ms=%.4f thr=%.2f\n", "insert", (unsigned long)stats.insertBatches, (unsigned)bs, _ms, _ms>0?(unsigned)bs/(_ms/1000.0):0.0); fflush(stdout); }
    stats.inserts += bs;
    stats.insertBatches++;
    pendInsVecs.clear();
    pendInsIds.clear();
}

// Pick a fresh entry point among live slots.
//   default        : FIRST live slot (scan from 0) — in a sliding window this is
//                    the trailing edge (oldest survivor), far from the active
//                    query region. Diagnostic showed this is the wiki collapse.
//   FDA2_MEDOID_LAST : LAST live slot (scan from top) — the window's leading edge,
//                    inside the active region. A/B test for the entry-point fix.
unsigned Executor::findLiveMedoid() const {
    static const bool useLast = getenv("FDA2_MEDOID_LAST") != nullptr;
    if (useLast) {
        for (long s = (long)g.numPoints - 1; s >= 0; s--)
            if (!hostDeleted[s]) return (unsigned)s;
    } else {
        for (unsigned s = 0; s < g.numPoints; s++)
            if (!hostDeleted[s]) return s;
    }
    return g.medoid;   // degenerate: everything deleted — leave unchanged
}

// ---------------------------------------------------------------------------
// DELETE flush
// ---------------------------------------------------------------------------
void Executor::flushDeletes() {
    if (pendDelIds.empty()) return;
    auto t0 = std::chrono::high_resolution_clock::now();

    // Translate pointIds → slots.
    std::vector<unsigned> slots;
    slots.reserve(pendDelIds.size());
    for (unsigned pid : pendDelIds) {
        auto it = pointIdToSlot.find(pid);
        if (it != pointIdToSlot.end()) slots.push_back(it->second);
    }
    unsigned total = slots.size();
    unsigned off = 0;
    while (off < total) {
        unsigned room = total - off;
        if (room > deleteBatchSize) room = deleteBatchSize;
        deleteBatch(&g, slots.data() + off, room, searchL, alpha, &deleteBufs, stream);
        gpuErrchk(cudaStreamSynchronize(stream));

        // Update the host deleted mirror; if this chunk tombstoned the entry
        // medoid, re-point it to a live node before the next chunk/search.
        bool medoidHit = false;
        for (unsigned i = 0; i < room; i++) {
            unsigned s = slots[off + i];
            if (s < hostDeleted.size()) hostDeleted[s] = 1;
            if (s == g.medoid) medoidHit = true;
        }
        if (medoidHit) {
            unsigned nm = findLiveMedoid();
            g.medoid = nm;
            setRuntimeMedoid(nm, stream);
            gpuErrchk(cudaStreamSynchronize(stream));
        }
        off += room;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    stats.deleteMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    { double _ms = std::chrono::duration<double, std::milli>(t1 - t0).count();  /* MARKER:deleteBatches */
      printf("[PSTEP] op=%s idx=%lu n=%u ms=%.4f thr=%.2f\n", "delete", (unsigned long)stats.deleteBatches, (unsigned)total, _ms, _ms>0?(unsigned)total/(_ms/1000.0):0.0); fflush(stdout); }
    stats.deletes += total;
    stats.deleteBatches++;
    pendDelIds.clear();
}

// ---------------------------------------------------------------------------
// QUERY flush
// ---------------------------------------------------------------------------
void Executor::flushQueries() {
    if (pendQryIdx.empty()) return;
    unsigned bs = pendQryIdx.size();
    auto t0 = std::chrono::high_resolution_clock::now();

    // Gather query vectors with ONE index copy + one kernel (was a per-query
    // cudaMemcpyAsync loop — tens of ms of pure launch overhead per batch).
    gatherQueries(d_allQueryVecs, &searchBufs, pendQryIdx.data(), bs, stream);
    searchBatch(&g, searchBufs.d_queryVecs, bs, k, searchL, &searchBufs, stream);
    gpuErrchk(cudaStreamSynchronize(stream));

    // End the search timer HERE — the recall bookkeeping below is pure
    // measurement overhead (deep-copying the per-query embedded GT) and has no
    // equivalent on the insert side; charging it to searchMs made search look
    // slower than insert.
    auto t1 = std::chrono::high_resolution_clock::now();
    stats.searchMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
    { double _ms = std::chrono::duration<double, std::milli>(t1 - t0).count();  /* MARKER:queryBatches */
      printf("[PSTEP] op=%s idx=%lu n=%u ms=%.4f thr=%.2f\n", "search", (unsigned long)stats.queryBatches, (unsigned)bs, _ms, _ms>0?(unsigned)bs/(_ms/1000.0):0.0); fflush(stdout); }
    stats.queries += bs;
    stats.queryBatches++;

    // Recall bookkeeping (NOT timed). Move, don't copy, the per-query state.
    RecallBatch rb;
    rb.predicted.assign(searchBufs.h_topkIds, searchBufs.h_topkIds + (size_t)bs * k);
    rb.queryIdxs  = std::move(pendQryIdx);
    rb.perQueryGT = std::move(pendQryGT);
    deferredRecall.push_back(std::move(rb));
    pendQryIdx.clear();
    pendQryGT.clear();
}

// ---------------------------------------------------------------------------
// Phased event loop. Any type transition flushes the other pending types first,
// which stream-syncs — guaranteeing different op-types never overlap.
// ---------------------------------------------------------------------------
void Executor::runWorkload(const std::vector<WorkloadEvent>& events) {
    auto wallStart = std::chrono::high_resolution_clock::now();

    for (const auto& e : events) {
        switch (e.type) {
            case EVENT_INSERT: {
                if (!pendQryIdx.empty()) flushQueries();
                if (!pendDelIds.empty()) flushDeletes();
                if (e.vector.size() < D) break;
                size_t off = pendInsVecs.size();
                pendInsVecs.resize(off + D);
                std::memcpy(pendInsVecs.data() + off, e.vector.data(), D * sizeof(float));
                pendInsIds.push_back(e.pointId);
                if (pendInsVecs.size() / D >= insertBatchSize) flushInserts();
                break;
            }
            case EVENT_QUERY: {
                if (!pendInsVecs.empty()) flushInserts();
                if (!pendDelIds.empty()) flushDeletes();
                pendQryIdx.push_back(e.queryId);
                pendQryGT.push_back(e.groundtruth);
                if (pendQryIdx.size() >= queryBatchSize) flushQueries();
                break;
            }
            case EVENT_DELETE: {
                if (!pendInsVecs.empty()) flushInserts();
                if (!pendQryIdx.empty()) flushQueries();
                pendDelIds.push_back(e.pointId);
                if (pendDelIds.size() >= deleteBatchSize) flushDeletes();
                break;
            }
            default: break;
        }
    }
    flushInserts();
    flushDeletes();
    flushQueries();

    auto wallEnd = std::chrono::high_resolution_clock::now();
    stats.wallMs = std::chrono::duration<double, std::milli>(wallEnd - wallStart).count();
    computeAllRecall();
}

// ---------------------------------------------------------------------------
// Recall (10-recall@10). Prefers per-query embedded GT (filtered to the live
// DB at query time) over the static bin GT; translates predicted slots →
// pointIds before comparing.
// ---------------------------------------------------------------------------
void Executor::computeAllRecall() {
    if (!h_groundtruth && deferredRecall.empty()) return;
    auto t0 = std::chrono::high_resolution_clock::now();
    double total = 0.0;
    unsigned long count = 0;
    // Region-split recall: classify each GT neighbour as base (pointId <
    // numBasePoints) or inserted (>=), and track recall per region. Shows which
    // population the queries actually probe and how well each is found.
    unsigned long baseFound=0, baseTot=0, insFound=0, insTot=0;

    unsigned __bidx = 0;  // MARKER:precall
    printf("\n=== Per-batch recall (in workload order) ===\nbatch | avg recall@10\n"); fflush(stdout);
    for (const auto& rb : deferredRecall) {
        unsigned bs = rb.queryIdxs.size();
        double __btot = 0.0;
        for (unsigned q = 0; q < bs; q++) {
            unsigned qIdx = rb.queryIdxs[q];
            const unsigned* gt; unsigned gtLen;
            if (q < rb.perQueryGT.size() && !rb.perQueryGT[q].empty()) {
                gt = rb.perQueryGT[q].data();
                gtLen = (unsigned)rb.perQueryGT[q].size();
                if (gtLen > k) gtLen = k;
            } else if (h_groundtruth) {
                gt = h_groundtruth + (size_t)qIdx * gtK;
                gtLen = k;
            } else continue;

            const unsigned* predSlots = rb.predicted.data() + (size_t)q * k;
            unsigned hits = 0;
            for (unsigned i = 0; i < k; i++) {
                unsigned s = predSlots[i];
                unsigned pid = (s == 0xFFFFFFFFu || s >= slotToPointId.size())
                                 ? 0xFFFFFFFFu : slotToPointId[s];
                if (pid == 0xFFFFFFFFu) continue;
                for (unsigned j = 0; j < gtLen; j++)
                    if (pid == gt[j]) { hits++; break; }
            }
            total += 100.0 * hits / (double)k;
            __btot += 100.0 * hits / (double)k;  // MARKER:precallacc
            count++;

            // Per-region: for each GT neighbour, was it in the predicted top-k?
            for (unsigned j = 0; j < gtLen; j++) {
                unsigned g = gt[j];
                bool found = false;
                for (unsigned i = 0; i < k; i++) {
                    unsigned s = predSlots[i];
                    unsigned pid = (s == 0xFFFFFFFFu || s >= slotToPointId.size())
                                     ? 0xFFFFFFFFu : slotToPointId[s];
                    if (pid == g) { found = true; break; }
                }
                if (g < numBasePoints) { baseTot++; if (found) baseFound++; }
                else                   { insTot++;  if (found) insFound++;  }
            }
        }
        printf("%3u   | %12.2f%%\n", __bidx++, bs ? __btot/bs : 0.0); fflush(stdout);  // MARKER:precallprint
    }
    if (count > 0) stats.recall10 = total;
    if (getenv("FDA2_ADJ_DIAG")) {
        printf("[RECALL-SPLIT] base-GT(id<%u): %lu/%lu=%.2f%%  |  inserted-GT: %lu/%lu=%.2f%%  |  base-GT share=%.1f%%\n",
               numBasePoints,
               baseFound, baseTot, baseTot ? 100.0*baseFound/baseTot : 0.0,
               insFound, insTot, insTot ? 100.0*insFound/insTot : 0.0,
               (baseTot+insTot) ? 100.0*baseTot/(baseTot+insTot) : 0.0);
        fflush(stdout);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    stats.recallComputeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ---------------------------------------------------------------------------
// Post-run adjacency diagnostic.
//
// Copies the final graph + delete bitvector to host and, for every ACTIVE
// (non-deleted) slot, tallies: out-degree, live vs STALE (deleted-target) edges,
// and base-region (slot < base_offset) vs inserted-region neighbours. Split by
// whether the node itself is a base or an inserted node. base_offset is the
// original load-time node count (numBasePoints): slots below it are the pre-built
// graph, slots above are workload inserts.
//
// Reading: a delete-heavy workload that has severed the active window from the
// navigable base core shows up as (a) inserted-active nodes with few base
// neighbours and/or (b) a high stale-edge fraction (greedy wastes its beam on
// tombstoned targets).
// ---------------------------------------------------------------------------
void Executor::dumpAdjacencyDiagnostic() const {
    const unsigned base = numBasePoints;
    const unsigned np   = g.numPoints;

    std::vector<uint8_t> hg((size_t)g.capacity * graphEntrySize);
    cudaMemcpy(hg.data(), g.d_graph,
               (size_t)g.capacity * graphEntrySize, cudaMemcpyDeviceToHost);
    std::vector<unsigned> del(g.capacity);
    cudaMemcpy(del.data(), g.d_deleted,
               (size_t)g.capacity * sizeof(unsigned), cudaMemcpyDeviceToHost);

    struct Acc { double deg=0, live=0, stale=0, baseN=0, insN=0; long n=0; };
    Acc acc[2];                         // [0]=active base node, [1]=active inserted node
    long liveBase=0, delBase=0, liveIns=0, delIns=0;

    for (unsigned s = 0; s < np; s++) {
        bool isBaseNode = (s < base);
        if (del[s]) { if (isBaseNode) delBase++; else delIns++; continue; }
        if (isBaseNode) liveBase++; else liveIns++;

        const uint8_t* e = hg.data() + (size_t)s * graphEntrySize;
        unsigned deg = *(const unsigned*)(e + D*sizeof(GVEC));
        if (deg > R) deg = R;
        const unsigned* nbr = (const unsigned*)(e + D*sizeof(GVEC) + sizeof(unsigned));

        unsigned live=0, stale=0, bN=0, iN=0;
        for (unsigned i = 0; i < deg; i++) {
            unsigned nb = nbr[i];
            if (nb >= g.capacity) continue;
            if (del[nb]) stale++; else live++;
            if (nb < base) bN++; else iN++;
        }
        Acc& a = acc[isBaseNode ? 0 : 1];
        a.deg+=deg; a.live+=live; a.stale+=stale; a.baseN+=bN; a.insN+=iN; a.n++;
    }

    auto pr = [](const char* tag, const Acc& a) {
        if (a.n == 0) { printf("  %-22s n=0\n", tag); return; }
        double inv = 1.0 / a.n;
        double deg = a.deg*inv, stalePct = a.deg>0 ? 100.0*a.stale/a.deg : 0.0;
        printf("  %-22s n=%-8ld avgDeg=%5.1f  live=%5.1f stale=%5.1f (stale%%=%5.1f)  "
               "baseNbr=%5.1f insNbr=%5.1f (baseNbr%%=%5.1f)\n",
               tag, a.n, deg, a.live*inv, a.stale*inv, stalePct,
               a.baseN*inv, a.insN*inv, a.deg>0 ? 100.0*a.baseN/a.deg : 0.0);
    };

    Acc all; all.deg=acc[0].deg+acc[1].deg; all.live=acc[0].live+acc[1].live;
    all.stale=acc[0].stale+acc[1].stale; all.baseN=acc[0].baseN+acc[1].baseN;
    all.insN=acc[0].insN+acc[1].insN; all.n=acc[0].n+acc[1].n;

    printf("\n=== [ADJ-DIAG] base_offset=%u numPoints=%u capacity=%u ===\n",
           base, np, g.capacity);
    printf("  base region:     live=%-8ld deleted=%-8ld\n", liveBase, delBase);
    printf("  inserted region: live=%-8ld deleted=%-8ld\n", liveIns, delIns);
    pr("ACTIVE base nodes:",     acc[0]);
    pr("ACTIVE inserted nodes:", acc[1]);
    pr("ALL active nodes:",      all);
    fflush(stdout);
}

} // namespace fda2
