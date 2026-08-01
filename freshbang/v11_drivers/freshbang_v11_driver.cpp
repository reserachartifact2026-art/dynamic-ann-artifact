// freshbang_v11_driver.cpp — drive FreshBANG on v11 NeurIPS-style workloads.
//
// v11 vs v10: events are pretty-printed multi-line JSON, batched, and carry
// only IDs (no embedded vectors). This driver loads the dataset base + query
// vectors separately and gathers them by ID, using the shared v11 loader.
//
// build: see Makefile (links -lfreshbang). Needs v11_loader.hpp in include path.
//
// usage:
//   ./freshbang_v11_driver <base> <query> <workload.jsonl> <conf.json>
//        --cfg <freshbang.cfg> [--k 10] [--base-rows 500000]
//        [--slot-rows 1000000] [--metric l2|cosine|ip] [--max-cycles N]
//        [--build | --load] [--dtype float|uint8]
//
//   <base>/<query> are .fbin or .fvecs (auto-detected by extension).
//
//   Single binary handles both dtypes: main body is templated on T and both
//   T=float and T=uint8_t instantiations are compiled in; --dtype picks which
//   to run at startup (default float, matching prior single-dtype behavior).
//
//   100M-scale two-phase run: the driver is
//   invoked twice as separate processes sharing the same index/conf on disk.
//     1) --build : SetDatasetParams -> CreateAlgo(kBuild) -> BuildIndex ->
//                  SaveIndex, then exit. No dynamic ops.
//     2) --load  : SetDatasetParams -> CreateAlgo(kLoad), skips BuildIndex
//                  entirely (index already persisted by step 1), then runs
//                  the normal dynamic search/insert/delete workload.
//   Omitting both flags keeps the original single-process legacy behavior
//   (build + run in one invocation) used for 1M/10M.

#include "v11_loader.hpp"
#include "freshbang.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>
#include <unistd.h>

using clk = std::chrono::steady_clock;
static double ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

// Thrown to break out of v11::stream_batched early (--max-cycles).
struct StopStream {};

template <typename DT>
int run(int argc, char** argv) {
    std::string basePath = argv[1], qPath = argv[2], wlPath = argv[3], confPath = argv[4];
    uint32_t k = 10, baseRows = 500000, slotRows = 1000000, maxCycles = 0;
    std::string cfgPath, metric = "l2";
    FreshBANGMode mode = FreshBANGMode::kLegacy;

    for (int i = 5; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&](const char* nm) -> std::string {
            if (i + 1 >= argc) { std::cerr << nm << " needs a value\n"; std::exit(1); }
            return argv[++i];
        };
        if (a == "--k")               k        = std::stoul(val("--k"));
        else if (a == "--base-rows")  baseRows = std::stoul(val("--base-rows"));
        else if (a == "--slot-rows")  slotRows = std::stoul(val("--slot-rows"));
        else if (a == "--max-cycles") maxCycles= std::stoul(val("--max-cycles"));
        else if (a == "--metric")     metric   = val("--metric");
        else if (a == "--cfg")        cfgPath  = val("--cfg");
        else if (a == "--build")      mode     = FreshBANGMode::kBuild;
        else if (a == "--load")       mode     = FreshBANGMode::kLoad;
        else if (a == "--dtype")      val("--dtype");  // consumed by main(), ignore here
    }
    bool cosine = (metric == "cosine");
    bool ip = (metric == "ip");
    if (metric != "l2" && metric != "cosine" && metric != "ip") {
        std::cerr << "unknown metric " << metric << " (use l2|cosine|ip)\n"; return 1;
    }
    if (ip) std::cout << "[fb_v11] metric=ip: native InnerProduct kernel on RAW base/query (no augmentation, no normalization)\n";
    std::cout << "[fb_v11] mode=" << (mode == FreshBANGMode::kBuild ? "build" :
                                       mode == FreshBANGMode::kLoad  ? "load"  : "legacy") << "\n";

    // FreshBANG's CreateAlgo reads "freshbang.conf" from CWD; the cfg path's dir
    // is where the conf lives. Match the v10 driver: chdir there.
    if (!cfgPath.empty()) {
        size_t p = cfgPath.find_last_of('/');
        std::string dir = (p == std::string::npos) ? "." : cfgPath.substr(0, p);
        if (chdir(dir.c_str()) != 0) { std::cerr << "chdir(" << dir << ") failed\n"; return 1; }
        std::cout << "[fb_v11] cfg=" << cfgPath << " (chdir " << dir << ")\n";
    }

    // ---- load vectors (only the rows we need from base) ----
    std::cout << "[fb_v11] loading base " << basePath << " (<= " << slotRows << " rows)\n";
    v11::VecStoreT<DT> base = v11::loadVectors<DT>(basePath, slotRows);
    std::cout << "[fb_v11] base n=" << base.n << " dim=" << base.dim << "\n";
    std::cout << "[fb_v11] loading query " << qPath << "\n";
    v11::VecStoreT<DT> query = v11::loadVectors<DT>(qPath, 0);
    std::cout << "[fb_v11] query n=" << query.n << " dim=" << query.dim << "\n";
    if (base.dim != query.dim) { std::cerr << "dim mismatch base/query\n"; return 2; }
    uint32_t dim = (uint32_t)base.dim;
    uint32_t useRows = (uint32_t)std::min((size_t)baseRows, base.n);

    // Optional L2-normalize for cosine (FreshBANG only offers euclidean / IP;
    // cosine top-k == IP top-k on normalized vectors).
    auto normalizeInPlace = [&](v11::VecStoreT<DT>& vs) {
        static_assert(!std::is_integral<DT>::value || true, "");
        for (size_t i = 0; i < vs.n; ++i) {
            DT* v = vs.data.data() + i * vs.dim;
            double s = 0; for (size_t d = 0; d < vs.dim; ++d) s += (double)v[d] * v[d];
            float inv = (s > 0) ? (float)(1.0 / std::sqrt(s)) : 1.0f;
            for (size_t d = 0; d < vs.dim; ++d) v[d] *= inv;
        }
    };
    if (cosine) { normalizeInPlace(base); normalizeInPlace(query); }

    // ---- FreshBANG init sequence ----
    FreshBANG<DT> fb;
    BuildParams bp;
    bp.dataset_rows = slotRows;
    bp.dataset_dim  = dim;
    bp.distance_measure = (cosine || ip) ? FreshBANG<DT>::kDistanceMeasure_InnerProduct
                                         : FreshBANG<DT>::kDistanceMeasure_Euclidean;
    if (!fb.SetDatasetParams(bp)) { std::cerr << "SetDatasetParams failed\n"; return 1; }
    if (!fb.CreateAlgo(confPath, mode)) { std::cerr << "CreateAlgo failed\n"; return 1; }

    double buildMs = 0.0;
    if (mode != FreshBANGMode::kLoad) {
        std::cout << "[fb_v11] BuildIndex(" << useRows << ")\n";
        auto bt0 = clk::now();
        if (!fb.BuildIndex(base.data.data(), useRows)) { std::cerr << "BuildIndex failed\n"; return 1; }
        auto bt1 = clk::now();
        buildMs = ms(bt0, bt1);
        std::cout << "[fb_v11] build " << buildMs << " ms\n";
        fb.SaveIndex();
    } else {
        std::cout << "[fb_v11] --load mode: skipping BuildIndex, index loaded from disk by CreateAlgo\n";
    }

    if (mode == FreshBANGMode::kBuild) {
        std::cout << "[fb_v11] --build mode: index saved, skipping dynamic workload\n";
        return 0;
    }

    SearchParams sp; sp.recall_at_k = k;
    fb.SetSearchParams(sp);
    fb.SetInsertParams();

    // ---- Warm-up: 3 BatchedSearch on the first query batch, UNTIMED (excluded
    // from final search QPS). Primes the CUDA context, lazy allocations, and
    // caches so measured throughput reflects steady state, not cold start.
    {
        uint32_t wn = (query.n < 10000) ? (uint32_t)query.n : 10000u;
        if (wn > 0) {
            std::vector<DT>       wq((size_t)wn * dim);
            for (uint32_t i = 0; i < wn; ++i)
                std::memcpy(wq.data() + (size_t)i * dim, query.row(i), dim * sizeof(DT));
            std::vector<uint64_t> wo((size_t)wn * k);
            std::vector<uint32_t> wnb((size_t)wn * k);
            std::vector<float>    wd((size_t)wn * k);
            for (int w = 0; w < 3; ++w)
                fb.BatchedSearch(wq.data(), wn, wo.data(), wnb.data(), wd.data());
            std::cout << "[fb_v11] warm-up: 3x BatchedSearch(" << wn << ") done (untimed)\n";
        }
    }

    // ---- metrics ----
    uint64_t nIns = 0, nDel = 0, nQ = 0, insB = 0, delB = 0, qB = 0;
    double insMs = 0, delMs = 0, qMs = 0, recallSum = 0; uint64_t recallN = 0;

    std::vector<DT> insVec;             // gathered insert vectors
    std::vector<DT> qVec;               // gathered query vectors
    std::vector<uint64_t> outIds;
    std::vector<uint32_t> outNbr;
    std::vector<float> outDist;

    auto wall0 = clk::now();

    try {
    v11::stream_batched(wlPath,
        // ---- insert batch: gather base vectors by id ----
        [&](const std::vector<unsigned>& ids) {
            uint32_t n = (uint32_t)ids.size();
            insVec.resize((size_t)n * dim);
            std::vector<uint64_t> id64(n);
            for (uint32_t i = 0; i < n; ++i) {
                unsigned id = ids[i];
                const DT* src = base.row(id);              // id < slotRows guaranteed by gen
                std::memcpy(insVec.data() + (size_t)i * dim, src, dim * sizeof(DT));
                id64[i] = id;
            }
            auto t0 = clk::now();
            fb.BatchedInsert(insVec.data(), n, id64.data());
            double bms = ms(t0, clk::now());
            insMs += bms; nIns += n; insB++;
            std::cout << "[PSTEP] insert batch=" << insB << " n=" << n
                      << " ms=" << bms << " thr=" << (bms>0 ? n*1000.0/bms : 0.0) << " ins/s\n";
        },
        // ---- delete batch ----
        [&](const std::vector<unsigned>& ids) {
            uint32_t n = (uint32_t)ids.size();
            std::vector<uint64_t> id64(ids.begin(), ids.end());
            auto t0 = clk::now();
            fb.BatchedDelete(id64.data(), n);
            double bms = ms(t0, clk::now());
            delMs += bms; nDel += n; delB++;
            std::cout << "[PSTEP] delete batch=" << delB << " n=" << n
                      << " ms=" << bms << " thr=" << (bms>0 ? n*1000.0/bms : 0.0) << " del/s\n";
        },
        // ---- query batch: gather query vectors by point_id, recall vs gt ----
        [&](const std::vector<unsigned>& qids, const std::vector<std::vector<unsigned>>& gt) {
            uint32_t n = (uint32_t)qids.size();
            qVec.resize((size_t)n * dim);
            for (uint32_t i = 0; i < n; ++i)
                std::memcpy(qVec.data() + (size_t)i * dim, query.row(qids[i]), dim * sizeof(DT));
            outIds.assign((size_t)n * k, 0);
            outNbr.assign((size_t)n * k, 0);
            outDist.assign((size_t)n * k, 0.f);
            auto t0 = clk::now();
            fb.BatchedSearch(qVec.data(), n, outIds.data(), outNbr.data(), outDist.data());
            double bms = ms(t0, clk::now());
            qMs += bms; nQ += n; qB++;
            double bRec = 0; uint64_t bN = 0;
            for (uint32_t i = 0; i < n; ++i) {
                if (i >= gt.size() || gt[i].empty()) continue;
                std::unordered_set<unsigned> g(gt[i].begin(),
                    gt[i].begin() + std::min((size_t)k, gt[i].size()));
                int hits = 0;
                for (uint32_t j = 0; j < k; ++j)
                    if (g.count((unsigned)outIds[(size_t)i * k + j])) hits++;
                double rc = (double)hits / (double)std::min((size_t)k, gt[i].size());
                recallSum += rc; recallN++; bRec += rc; bN++;
            }
            std::cout << "[PSTEP] search batch=" << qB << " q=" << n
                      << " ms=" << bms << " qps=" << (bms>0 ? n*1000.0/bms : 0.0)
                      << " recall@" << k << "=" << (bN ? 100.0*bRec/bN : 0.0) << "%\n";
            if (maxCycles && delB >= maxCycles) throw StopStream{};
        });
    } catch (const StopStream&) {
        std::cout << "[fb_v11] reached --max-cycles=" << maxCycles << ", stopping early\n";
    }

    double wallMs = ms(wall0, clk::now());

    std::cout << "\n=== FreshBANG v11 Statistics ===\n";
    std::cout << "Total ops:      " << (nIns + nDel + nQ) << " (ins " << nIns
              << " + del " << nDel << " + q " << nQ << ")\n";
    std::cout << "Inserts:        " << nIns << " (in " << insB << " batches)\n";
    std::cout << "Deletes:        " << nDel << " (in " << delB << " batches)\n";
    std::cout << "Queries:        " << nQ  << " (in " << qB  << " batches)\n";
    std::cout << "Build time:     " << buildMs << " ms\n";
    std::cout << "Wall (post-build): " << wallMs << " ms\n";
    if (nIns) std::cout << "  Insert: " << insMs << " ms (" << (nIns * 1000.0 / insMs) << " ins/s)\n";
    if (nDel) std::cout << "  Delete: " << delMs << " ms (" << (nDel * 1000.0 / delMs) << " del/s)\n";
    if (nQ) {
        std::cout << "  Query:  " << qMs << " ms (" << (nQ * 1000.0 / qMs) << " q/s)\n";
        std::cout << "  Recall@" << k << ": " << (recallN ? 100.0 * recallSum / recallN : 0.0) << "%\n";
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr <<
          "usage: " << argv[0] << " <base> <query> <workload.jsonl> <conf.json>\n"
          "   [--cfg PATH] [--k 10] [--base-rows 500000] [--slot-rows 1000000]\n"
          "   [--metric l2|cosine|ip] [--max-cycles N] [--build | --load]\n"
          "   [--dtype float|uint8]\n";
        return 1;
    }
    std::string dtype = "float";
    for (int i = 5; i < argc; ++i) {
        if (std::string(argv[i]) == "--dtype" && i + 1 < argc) { dtype = argv[i + 1]; break; }
    }
    if (dtype == "float")  { return run<float>(argc, argv); }
    if (dtype == "uint8")  { return run<uint8_t>(argc, argv); }
    std::cerr << "unknown --dtype " << dtype << " (use float|uint8)\n";
    return 1;
}
