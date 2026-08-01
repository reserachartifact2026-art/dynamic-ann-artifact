// v11_selftest.cpp — validate the v11 loader end-to-end:
//   1. parse counts (inserts/deletes/queries) match meta.json
//   2. id ranges sane (insert ids >= base_offset, query ids < num_queries set)
//   3. base/query vector lookups resolve
//   4. spot-recompute brute-force GT for the FIRST query of the FIRST search
//      batch against the active set, compare against embedded GT (set overlap).
//
// build: g++ -O2 -std=c++17 v11_selftest.cpp -fopenmp -o v11_selftest
// run:   ./v11_selftest <workload.jsonl> <base> <query> <dim> <metric l2|cosine> <base_offset> <k>

#include "v11_loader.hpp"
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <omp.h>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 8) {
        std::fprintf(stderr,
            "usage: %s <workload> <base> <query> <dim> <l2|cosine> <base_offset> <k> [base_max_rows=1000000]\n", argv[0]);
        return 1;
    }
    std::string wl = argv[1], basePath = argv[2], qPath = argv[3];
    size_t dim = std::stoul(argv[4]);
    std::string metric = argv[5];
    unsigned baseOffset = std::stoul(argv[6]);
    unsigned k = std::stoul(argv[7]);
    // optional 8th arg: cap base rows loaded (default 1M; 0 = all). Avoids
    // pulling a 35M-row wikipedia base when the workload only touches the first 1M.
    size_t baseMaxRows = (argc > 8) ? std::stoul(argv[8]) : 1000000;
    bool cosine = (metric == "cosine");

    std::fprintf(stderr, "[load] base=%s (<= %zu rows) ...\n", basePath.c_str(), baseMaxRows);
    v11::VecStore base = v11::loadVectors(basePath, baseMaxRows);
    std::fprintf(stderr, "[load] base n=%zu dim=%zu\n", base.n, base.dim);
    v11::VecStore query = v11::loadVectors(qPath, 0);
    std::fprintf(stderr, "[load] query n=%zu dim=%zu\n", query.n, query.dim);
    if (base.dim != dim || query.dim != dim) {
        std::fprintf(stderr, "DIM MISMATCH: arg=%zu base=%zu query=%zu\n", dim, base.dim, query.dim);
        return 2;
    }

    auto norm = [&](const float* v) {
        double s = 0; for (size_t d = 0; d < dim; ++d) s += (double)v[d] * v[d];
        return std::sqrt(s);
    };

    // --- pass 1: counts + active set, capture first search batch ---
    std::vector<char> active(base.n, 0);
    for (unsigned i = 0; i < baseOffset && i < base.n; ++i) active[i] = 1;

    long nIns = 0, nDel = 0, nQ = 0, nSearchItems = 0;
    unsigned minInsId = UINT32_MAX, maxInsId = 0, maxQId = 0;

    bool firstBatchDone = false;
    unsigned firstQid = 0;
    std::vector<unsigned> firstGt;
    std::vector<char> activeSnapshot;   // active set at the moment of first search

    // We need active-set state AT the first search. Use per-item stream; when the
    // first query arrives, snapshot active[].
    bool snapshotTaken = false;
    v11::stream(wl,
        [&](unsigned id) { if (!snapshotTaken) { if (id < base.n) active[id] = 1; }
                           nIns++; minInsId = std::min(minInsId, id); maxInsId = std::max(maxInsId, id); },
        [&](unsigned id) { if (!snapshotTaken) { if (id < base.n) active[id] = 0; }
                           nDel++; },
        [&](unsigned qid, const std::vector<unsigned>& gt) {
            nQ++; nSearchItems++;
            maxQId = std::max(maxQId, qid);
            if (!snapshotTaken) {
                snapshotTaken = true;
                activeSnapshot = active;        // freeze active set at first search
                firstQid = qid; firstGt = gt;
                firstBatchDone = true;
            }
        });

    std::fprintf(stderr, "\n=== counts ===\n");
    std::fprintf(stderr, "inserts=%ld deletes=%ld queries=%ld\n", nIns, nDel, nQ);
    std::fprintf(stderr, "insert id range=[%u, %u]  (base_offset=%u)\n", minInsId, maxInsId, baseOffset);
    std::fprintf(stderr, "max query id=%u (query set n=%zu)\n", maxQId, query.n);
    if (minInsId < baseOffset)
        std::fprintf(stderr, "WARN: insert id < base_offset (overlaps base)\n");
    if (maxQId >= query.n)
        std::fprintf(stderr, "WARN: query id >= query set size\n");

    if (!firstBatchDone) { std::fprintf(stderr, "no search events found\n"); return 0; }

    // --- pass 2: brute-force GT for the first query against the snapshot set ---
    std::fprintf(stderr, "\n=== GT spot-check: first query qid=%u, embedded gt size=%zu ===\n",
                 firstQid, firstGt.size());
    std::vector<unsigned> activeIds;
    activeIds.reserve(base.n);
    for (size_t i = 0; i < base.n; ++i) if (activeSnapshot[i]) activeIds.push_back((unsigned)i);
    std::fprintf(stderr, "active count at first search = %zu\n", activeIds.size());

    const float* q = query.row(firstQid);
    double qn = cosine ? norm(q) : 1.0;
    if (qn == 0) qn = 1.0;

    size_t M = activeIds.size();
    std::vector<std::pair<float, unsigned>> scored(M);
    #pragma omp parallel for schedule(static)
    for (long j = 0; j < (long)M; ++j) {
        unsigned id = activeIds[j];
        const float* a = base.row(id);
        double score;
        if (cosine) {
            double dot = 0, an = 0;
            for (size_t d = 0; d < dim; ++d) { dot += (double)q[d] * a[d]; an += (double)a[d] * a[d]; }
            an = std::sqrt(an); if (an == 0) an = 1.0;
            score = -(dot / (qn * an));            // higher cos = better → negate for min-sort
        } else {
            double s = 0;
            for (size_t d = 0; d < dim; ++d) { double df = (double)q[d] - a[d]; s += df * df; }
            score = s;                              // L2: smaller better
        }
        scored[j] = { (float)score, id };
    }
    size_t topk = std::min((size_t)k, M);
    std::partial_sort(scored.begin(), scored.begin() + topk, scored.end());

    std::unordered_set<unsigned> bf;
    for (size_t j = 0; j < topk; ++j) bf.insert(scored[j].second);
    std::unordered_set<unsigned> emb(firstGt.begin(), firstGt.begin() + std::min((size_t)k, firstGt.size()));

    size_t inter = 0;
    for (unsigned id : emb) if (bf.count(id)) inter++;
    std::fprintf(stderr, "brute-force top-%zu vs embedded top-%zu: overlap=%zu/%zu (%.2f%%)\n",
                 topk, emb.size(), inter, emb.size(),
                 emb.empty() ? 0.0 : 100.0 * inter / emb.size());
    std::fprintf(stderr, "bf[:10]  = ");
    for (size_t j = 0; j < std::min((size_t)10, topk); ++j) std::fprintf(stderr, "%u ", scored[j].second);
    std::fprintf(stderr, "\nemb[:10] = ");
    for (size_t j = 0; j < std::min((size_t)10, firstGt.size()); ++j) std::fprintf(stderr, "%u ", firstGt[j]);
    std::fprintf(stderr, "\n");
    return 0;
}
