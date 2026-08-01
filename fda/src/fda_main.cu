// FreshDiskANN (GPU) — Naive Implementation (paper-faithful, unified-graph).
//
// Same CLI shape as the previous fda_main for drop-in compatibility.

#include <cuda_runtime.h>
#include "fda_executor.h"
#include "fda_insert.h"
#include "../baseline_src/dynamic/workload.h"
#include "v11_loader.hpp"   // shared v11 (NeurIPS-style) workload loader
#include "metric.hpp"       // l2 / cosine / ip via loader-side transforms
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>

static size_t fileSize(const char* path) {
    struct stat st;
    if (stat(path, &st) != 0) return 0;
    return (size_t)st.st_size;
}

static uint8_t* loadGraphHost(const char* path, size_t* outNumPoints) {
    size_t bytes = fileSize(path);
    if (bytes == 0) { fprintf(stderr, "Cannot stat graph file: %s\n", path); exit(1); }
    if (bytes % graphEntrySize != 0) {
        fprintf(stderr, "Graph file size %zu not divisible by entry size %zu\n",
                bytes, (size_t)graphEntrySize);
        exit(1);
    }
    *outNumPoints = bytes / graphEntrySize;
    uint8_t* buf = (uint8_t*)malloc(bytes);
    std::ifstream f(path, std::ios::binary);
    f.read((char*)buf, bytes);
    return buf;
}

// Infer element dtype from file extension.
// Returns "float32", "uint8", or "int8".
static std::string inferDtype(const char* path) {
    std::string p(path);
    auto ends = [&](const char* s) {
        size_t sl = strlen(s);
        return p.size() >= sl && p.compare(p.size() - sl, sl, s) == 0;
    };
    if (ends(".u8bin") || ends(".bvecs")) return "uint8";
    if (ends(".i8bin"))                   return "int8";
    return "float32";
}

// Load query vectors from a binary file and return as float32, applying the
// metric transform. Reads T natively then converts to float (no-op when T=float).
//   l2     : output dim = file dim.
//   cosine : output dim = file dim, each row unit-normalized.
//   ip     : output dim = file dim + 1, each row gets a trailing 0 (query augment).
template <typename T>
static float* loadQueryAsFloat(const char* path, unsigned* outN, unsigned* outDim,
                               metric::Metric m = metric::Metric::L2) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    unsigned n, fdim;
    f.read((char*)&n, sizeof(unsigned));
    f.read((char*)&fdim, sizeof(unsigned));
    unsigned od = (m == metric::Metric::IP) ? fdim + 1 : fdim;   // ip appends one dim
    *outN = n; *outDim = od;
    size_t total = (size_t)n * fdim;
    std::vector<T> tmp(total);
    f.read((char*)tmp.data(), total * sizeof(T));
    float* buf = (float*)malloc((size_t)n * od * sizeof(float));
    for (size_t r = 0; r < n; r++) {
        float* dst = buf + r * od;
        const T* src = tmp.data() + r * fdim;
        for (unsigned i = 0; i < fdim; i++) dst[i] = static_cast<float>(src[i]);
        if (m == metric::Metric::IP)          dst[fdim] = 0.0f;   // query augment
        else if (m == metric::Metric::Cosine) metric::normalize(dst, fdim);
    }
    return buf;
}

static float* loadFloatBin(const char* path, unsigned* outN, unsigned* outDim) {
    return loadQueryAsFloat<float>(path, outN, outDim);
}

// Populate events from a v11 workload using base vectors of type T, applying the
// metric transform so insert vectors match the graph's space (always float, dim D):
//   l2     : copy (dim D).
//   cosine : copy + unit-normalize (dim D).
//   ip      : augment to D = base.dim+1 = [v, sqrt(M²-||v||²)], M = max base norm.
template <typename T>
static void populateV11Events(const char* baseDataPath, const char* workloadPath,
                               std::vector<WorkloadEvent>& events,
                               metric::Metric m = metric::Metric::L2) {
    v11::VecStore<T> base = v11::loadVectors<T>(baseDataPath, /*max_rows=*/0);
    unsigned od = (m == metric::Metric::IP) ? (unsigned)base.dim + 1 : (unsigned)base.dim;
    if (od != (unsigned)D) {
        fprintf(stderr, "[v11] base-data dim %zu (+%d for ip) = %u != compile-time D %d\n",
                base.dim, m == metric::Metric::IP ? 1 : 0, od, D);
        exit(1);
    }
    float maxNorm = 0.0f;
    if (m == metric::Metric::IP) {
        std::vector<float> tmp(base.dim);
        for (size_t r = 0; r < base.n; r++) {
            const T* v = base.row(r);
            for (size_t i = 0; i < base.dim; i++) tmp[i] = (float)v[i];
            double nn = metric::norm2(tmp.data(), (unsigned)base.dim);
            if (nn > (double)maxNorm * maxNorm) maxNorm = (float)std::sqrt(nn);
        }
    }
    printf("[v11] base=%s n=%zu dim=%zu dtype=%s metric=%s (workload=%s)\n",
           baseDataPath, base.n, base.dim,
           sizeof(T)==1 ? (std::is_signed<T>::value ? "int8" : "uint8") : "float32",
           metric::name(m), workloadPath);
    fflush(stdout);
    v11::stream(workloadPath,
        [&](unsigned id) {
            WorkloadEvent e; e.type = EVENT_INSERT; e.pointId = id;
            const T* v = base.row(id);
            e.vector.resize(od);
            for (size_t i = 0; i < base.dim; i++) e.vector[i] = (float)v[i];
            if (m == metric::Metric::IP)
                metric::augmentDB(e.vector.data(), (unsigned)base.dim, maxNorm, e.vector.data());
            else if (m == metric::Metric::Cosine)
                metric::normalize(e.vector.data(), (unsigned)base.dim);
            events.push_back(std::move(e));
        },
        [&](unsigned id) {
            WorkloadEvent e; e.type = EVENT_DELETE; e.pointId = id;
            events.push_back(std::move(e));
        },
        [&](unsigned qid, const std::vector<unsigned>& gt) {
            WorkloadEvent e; e.type = EVENT_QUERY; e.queryId = qid;
            e.groundtruth = gt;
            events.push_back(std::move(e));
        });
    printf("[v11] expanded events: %zu\n", events.size());
    fflush(stdout);
}

static unsigned* loadUintBin(const char* path, unsigned* outN, unsigned* outK) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    unsigned n, k;
    f.read((char*)&n, sizeof(unsigned));
    f.read((char*)&k, sizeof(unsigned));
    *outN = n; *outK = k;
    unsigned* buf = (unsigned*)malloc((size_t)n * k * sizeof(unsigned));
    f.read((char*)buf, (size_t)n * k * sizeof(unsigned));
    return buf;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr,
            "Usage: %s <graph.out> <workload.jsonl> <queries.bin> <gt.bin> "
            "[--searchL N] [--k K] [--m M] [--t T] "
            "[--del-cap N] [--insert-batch N] [--query-batch N] [--alpha A] [--base-data PATH]\n"
            "  --m: max sealed Temp Indexes before merge fires (default 4)\n"
            "  --t: per-Temp/RW capacity (default = workload-inserts / m)\n"
            "  --base-data: dataset base vectors (.fbin/.fvecs) — enables v11 batched/IDs-only workloads\n"
            "  --metric l2|cosine|ip: distance (default l2). cosine normalizes at load; ip uses the\n"
            "           BANG extra-dim reduction (build the graph with D = real_dim+1 for ip).\n",
            argv[0]);
        return 1;
    }
    const char* graphPath    = argv[1];
    const char* workloadPath = argv[2];
    const char* queryPath    = argv[3];
    const char* gtPath       = argv[4];
    const char* baseDataPath = nullptr;   // --base-data: enables v11 workload mode
    const char* dtypeArg     = nullptr;   // --dtype: float32|uint8|int8 (auto-detected if omitted)
    const char* metricArg    = nullptr;   // --metric: l2|cosine|ip (default l2)

    unsigned searchL              = 100;
    unsigned k                    = 10;
    unsigned maxTemps             = 4;       // M — sealed temps before merge
    unsigned tempCapacityCli      = 0;       // T — 0 = auto-size from workload / M
    bool     eagerMerge           = false;   // --eager-merge: force merge at every insert→query / delete→query transition
    unsigned deleteListCapacity   = 2000000;
    unsigned insertBatchSize      = 1024;
    unsigned queryBatchSize       = 10000;
    unsigned deleteBatchSize      = 10000;
    float    alpha                = 1.2f;
    // Separate worklist size for insert greedy. 0 = use the same value as
    // --searchL (legacy behavior). Default 0 → match --searchL so behavior is
    // unchanged out of the box. Lower (e.g. 50) shrinks per-step worklist
    // merge work; needs empirical tuning per dataset (recall vs throughput).
    unsigned insertSearchL        = 0;

    for (int i = 5; i < argc; i++) {
        if (!strcmp(argv[i], "--searchL") && i+1 < argc) searchL = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--k") && i+1 < argc) k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--m") && i+1 < argc) maxTemps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--t") && i+1 < argc) tempCapacityCli = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--del-cap") && i+1 < argc) deleteListCapacity = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--insert-batch") && i+1 < argc) insertBatchSize = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--query-batch") && i+1 < argc) queryBatchSize = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--alpha") && i+1 < argc) alpha = atof(argv[++i]);
        else if (!strcmp(argv[i], "--insert-searchL") && i+1 < argc) insertSearchL = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--eager-merge")) eagerMerge = true;
        else if (!strcmp(argv[i], "--base-data") && i+1 < argc) baseDataPath = argv[++i];
        else if (!strcmp(argv[i], "--dtype")     && i+1 < argc) dtypeArg     = argv[++i];
        else if (!strcmp(argv[i], "--metric")    && i+1 < argc) metricArg    = argv[++i];
    }

    metric::Metric met = metric::parse(metricArg);

    // Load base graph for LTI
    size_t numBasePoints = 0;
    uint8_t* h_graph = loadGraphHost(graphPath, &numBasePoints);
    printf("Loaded base graph: %zu points\n", numBasePoints);
    printf("Distance metric: %s\n", metric::name(met));

    // cosine: unit-normalize the graph's stored base vectors so graph, queries and
    // inserts share the unit sphere (L2 rank == cosine rank). ip: the graph must
    // already be built on the augmented (D = real+1) vectors.
    if (met == metric::Metric::Cosine) {
        for (size_t s = 0; s < numBasePoints; s++)
            metric::normalize((float*)(h_graph + s * graphEntrySize), D);
    }

    // Resolve dtype: explicit --dtype overrides auto-detection from extension.
    std::string dtype = dtypeArg ? std::string(dtypeArg) : inferDtype(queryPath);
    if (baseDataPath && !dtypeArg) dtype = inferDtype(baseDataPath);
    printf("Dataset dtype: %s\n", dtype.c_str());

    // Load queries as float32 (converted from native type + metric transform).
    unsigned nq, qDim;
    float* h_queries;
    if      (dtype == "uint8")  h_queries = loadQueryAsFloat<uint8_t>(queryPath, &nq, &qDim, met);
    else if (dtype == "int8")   h_queries = loadQueryAsFloat<int8_t> (queryPath, &nq, &qDim, met);
    else                        h_queries = loadQueryAsFloat<float>  (queryPath, &nq, &qDim, met);
    if (qDim != D) {
        fprintf(stderr, "Query dim %u != compile-time D %d\n", qDim, D);
        return 1;
    }
    unsigned ngt, gtK;
    unsigned* h_gt = loadUintBin(gtPath, &ngt, &gtK);
    printf("Loaded %u queries (dim=%u), GT k=%u\n", nq, qDim, gtK);

    // Parse workload. Two paths:
    //  - v11 (NeurIPS-style, batched, IDs-only): given via --base-data. Insert
    //    vectors are gathered from the dataset base file by id; queries resolve
    //    by queryId into queries.bin (loaded above); GT is embedded per query.
    //  - legacy v10 jsonl (embedded vectors): the original Workload parser.
    std::vector<WorkloadEvent> events;
    if (baseDataPath) {
      try {
        if      (dtype == "uint8") populateV11Events<uint8_t>(baseDataPath, workloadPath, events, met);
        else if (dtype == "int8")  populateV11Events<int8_t> (baseDataPath, workloadPath, events, met);
        else                       populateV11Events<float>  (baseDataPath, workloadPath, events, met);
      } catch (const std::exception& ex) {
        fprintf(stderr, "[v11] error: %s\n", ex.what());
        return 1;
      }
    } else {
        Workload workload(workloadPath);
        fflush(stdout);
        workload.printSummary();
        fflush(stdout);
        events.reserve(workload.size());
        for (size_t i = 0; i < workload.size(); i++) events.push_back(workload[i]);
        printf("Loaded events into vector: %zu\n", events.size());
        fflush(stdout);
    }

    // Count total inserts to size temp capacity if not specified.
    unsigned numInserts = 0;
    for (const auto& e : events) if (e.type == EVENT_INSERT) numInserts++;

    unsigned ltiCount = (unsigned)numBasePoints;
    unsigned tempCapacity = tempCapacityCli;
    if (tempCapacity == 0) {
        // Default: each temp holds numInserts/M vertices (so we get exactly M
        // seals over the course of the workload, triggering 1 merge near end).
        tempCapacity = (numInserts + maxTemps - 1) / maxTemps;
        if (tempCapacity == 0) tempCapacity = 1;
    }

    // Total slot capacity must cover: original L + total workload inserts +
    // one extra T for the active RW range above the absorbed region.
    // After every merge, ltiCount grows to include the merged Temps' slots
    // (in-place Phase B keeps slot positions stable). For an N-insert
    // workload, the maximum ltiCount reached is L + N, plus we need T more
    // slots above that for the currently-filling RW.
    // FIX: allocate L+(M+1)*T (matches allocateUnifiedIndex assert + fda_index.h). The old
    // L+numInserts+T under-allocated by (M*T - numInserts) whenever numInserts was not
    // divisible by M (e.g. clustered workloads: glove, msturing) -> capacity assert failure.
    unsigned unifiedCapacity = ltiCount + (maxTemps + 1) * tempCapacity;

    // Bump delete bitvector to cover full capacity
    if (deleteListCapacity < unifiedCapacity) deleteListCapacity = unifiedCapacity;

    printf("Config: searchL=%u insertSearchL=%u k=%u L=%u T=%u M=%u capacity=%u delCap=%u α=%.2f\n",
           searchL, insertSearchL ? insertSearchL : searchL,
           k, ltiCount, tempCapacity, maxTemps, unifiedCapacity,
           deleteListCapacity, alpha);
    printf("Workload: %u inserts → ~%u seals (T=%u each)\n",
           numInserts, (numInserts + tempCapacity - 1) / tempCapacity, tempCapacity);
    fflush(stdout);

    // searchL must be large enough to hold k surviving entries after deletes
    // skip some. With searchL < 2*k, extractTopKFromWorklist often pads the
    // result tail with sentinels (0xFFFFFFFF / FLT_MAX) when deleted IDs eat
    // into the worklist's top-k. 2*k is a conservative lower bound — finer
    // tuning depends on the workload's delete density.
    if (searchL < 2 * k) {
        fprintf(stderr,
                "[warn] searchL=%u < 2*k=%u — top-k may pad sentinels when deletes "
                "are concentrated in the worklist head. Bump searchL to at least %u.\n",
                searchL, 2 * k, 2 * k);
        fflush(stderr);
    }

    // MEDOID from vamana.h is sized for the canonical 500K/1M LTI. For smaller
    // LTI (e.g., 5K base graph) clamp it to a valid vertex inside the LTI
    // range — otherwise greedy starts at an empty RW slot and never explores.
    unsigned medoidId = MEDOID;
    if (medoidId >= ltiCount) {
        medoidId = ltiCount / 2;
        printf("MEDOID %d out of LTI range; clamped to %u\n", MEDOID, medoidId);
    }
    printf("Using medoid: %u\n", medoidId);
    fflush(stdout);

    fda::FdaExecutor exec(h_graph, (unsigned)numBasePoints, medoidId,
                          ltiCount, tempCapacity, maxTemps,
                          unifiedCapacity,
                          deleteListCapacity,
                          searchL, k, alpha,
                          insertBatchSize, queryBatchSize, deleteBatchSize,
                          insertSearchL,
                          h_queries, nq, h_gt, gtK);
    if (eagerMerge) {
        printf("Eager-merge mode enabled (force merge at insert→query / delete→query transitions)\n");
        fflush(stdout);
        exec.setEagerMergeOnTransition(true);
    }
    exec.runWorkload(events);
    exec.getStats().print();
    // fda::dumpInsertTimings();  // disabled — see fda_insert.cu

    free(h_graph);
    free(h_queries);
    free(h_gt);
    return 0;
}
