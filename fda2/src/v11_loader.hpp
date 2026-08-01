// v11_loader.hpp — shared loader for v11 NeurIPS-style streaming workloads.
//
// v11 format (one workload file):
//   A sequence of pretty-printed JSON objects concatenated together. Each
//   object is one BATCHED operation:
//     {"op_type":"insert","step":N,"vector_ids":[id,...],"num_vectors":M}
//     {"op_type":"delete","step":N,"vector_ids":[id,...],"num_vectors":M}
//     {"op_type":"search","step":N,"point_id":[qid,...],"groundtruth":[[...],...]}
//   Vectors are NOT embedded — only IDs. Insert/delete ids index the dataset
//   BASE vectors; search point_ids index the QUERY set; groundtruth[i] is the
//   per-query top-k id list computed against the active set at that step.
//
// This header is dependency-free (no nlohmann, no CUDA) and callback-based so
// every consumer (MVCC / BANG / FDA executors, the FreshBANG driver) can adapt
// it to its own event representation without ODR clashes.
//
// VecStore<T> is templatized on the native dataset element type:
//   float    — float32 (.fbin, .fvecs)
//   uint8_t  — uint8   (.u8bin, .bvecs)
//   int8_t   — int8    (.i8bin)
// Kernels and the graph always work in float32; conversion from T to float
// happens at the I/O boundary (WorkloadEvent.vector or loadQueryAsFloat).
//
// Usage:
//   v11::VecStore<uint8_t> base = v11::loadVectors<uint8_t>("base.u8bin");
//   v11::stream("workload.jsonl",
//       [&](unsigned id) {
//           const uint8_t* v = base.row(id);
//           e.vector.assign(v, v + base.dim);   // uint8→float implicit
//       },
//       [&](unsigned id) { /* delete */ },
//       [&](unsigned qid, const std::vector<unsigned>& gt) { /* query */ });
//
// The callbacks fire once per EXPANDED item (per id / per query).

#ifndef V11_LOADER_HPP
#define V11_LOADER_HPP

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <stdexcept>

namespace v11 {

// ---------------------------------------------------------------------------
// Vector storage: row-major [n x dim], native element type T.
//   T = float    → float32 (.fbin, .fvecs)
//   T = uint8_t  → uint8   (.u8bin, .bvecs)
//   T = int8_t   → int8    (.i8bin)
// Only the first `max_rows` rows are read (0 = all).
// ---------------------------------------------------------------------------
template <typename T = float>
struct VecStore {
    std::vector<T> data;   // n * dim, native type
    size_t n   = 0;
    size_t dim = 0;
    const T* row(size_t i) const { return data.data() + i * dim; }
};

// Binary format: [uint32 n][uint32 dim][n*dim T]
// Covers .fbin (T=float), .u8bin (T=uint8_t), .i8bin (T=int8_t).
template <typename T>
inline VecStore<T> loadBin(const std::string& path, size_t max_rows = 0) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("loadBin: cannot open " + path);
    uint32_t n = 0, dim = 0;
    if (std::fread(&n, 4, 1, f) != 1 || std::fread(&dim, 4, 1, f) != 1) {
        std::fclose(f); throw std::runtime_error("loadBin: short header " + path);
    }
    // Trust the file size over the header: some bases carry a stale row count (e.g. the
    // msturing-100m file is the first 100M rows of msturing-1B but keeps n=1e9). Clamp
    // n to what the file actually holds so we do not over-read ("short data").
    std::fseek(f, 0, SEEK_END);
    long _fsz = std::ftell(f);
    if (_fsz > 8 && dim > 0) {
        size_t _drows = (size_t)((_fsz - 8) / ((long)dim * (long)sizeof(T)));
        if (_drows < (size_t)n) { fprintf(stderr, "[v11] loadBin: header n=%u but file has %zu rows; clamping\n", n, _drows); n = (uint32_t)_drows; }
    }
    std::fseek(f, 8, SEEK_SET);
    size_t rows = (max_rows && max_rows < n) ? max_rows : n;
    VecStore<T> vs; vs.n = rows; vs.dim = dim;
    vs.data.resize(rows * (size_t)dim);
    if (std::fread(vs.data.data(), sizeof(T), rows * (size_t)dim, f) != rows * (size_t)dim) {
        std::fclose(f); throw std::runtime_error("loadBin: short data " + path);
    }
    std::fclose(f);
    return vs;
}

// Vecs format: repeated [int32 dim][dim T]
// Covers .fvecs (T=float), .bvecs (T=uint8_t).
template <typename T>
inline VecStore<T> loadVecs(const std::string& path, size_t max_rows = 0) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("loadVecs: cannot open " + path);
    int32_t dim0 = 0;
    if (std::fread(&dim0, 4, 1, f) != 1) { std::fclose(f); throw std::runtime_error("loadVecs: empty " + path); }
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    size_t rec = 4 + (size_t)dim0 * sizeof(T);
    size_t n_total = (size_t)bytes / rec;
    size_t rows = (max_rows && max_rows < n_total) ? max_rows : n_total;
    VecStore<T> vs; vs.n = rows; vs.dim = (size_t)dim0;
    vs.data.resize(rows * (size_t)dim0);
    for (size_t i = 0; i < rows; ++i) {
        int32_t d;
        if (std::fread(&d, 4, 1, f) != 1 || d != dim0) { std::fclose(f); throw std::runtime_error("loadVecs: dim mismatch " + path); }
        if (std::fread(vs.data.data() + i * (size_t)dim0, sizeof(T), dim0, f) != (size_t)dim0) {
            std::fclose(f); throw std::runtime_error("loadVecs: short row " + path);
        }
    }
    std::fclose(f);
    return vs;
}

// Dispatch on extension:
//   .fbin / .bin  → loadBin<T>
//   .u8bin        → loadBin<T>  (caller passes T=uint8_t)
//   .i8bin        → loadBin<T>  (caller passes T=int8_t)
//   .fvecs        → loadVecs<T> (caller passes T=float)
//   .bvecs        → loadVecs<T> (caller passes T=uint8_t)
template <typename T = float>
inline VecStore<T> loadVectors(const std::string& path, size_t max_rows = 0) {
    auto endsWith = [&](const char* sfx) {
        size_t sl = std::strlen(sfx);
        return path.size() >= sl && path.compare(path.size() - sl, sl, sfx) == 0;
    };
    if (endsWith(".fvecs") || endsWith(".bvecs"))
        return loadVecs<T>(path, max_rows);
    return loadBin<T>(path, max_rows);   // .fbin / .u8bin / .i8bin / .bin
}

// ---------------------------------------------------------------------------
// Streaming JSON object splitter: yields one top-level {...} object at a time,
// tracking brace depth so multi-line arrays don't confuse it. Calls handle(obj).
// ---------------------------------------------------------------------------
template <class Handle>
inline void forEachObject(const std::string& path, Handle handle) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("forEachObject: cannot open " + path);
    std::string buf;
    buf.reserve(1 << 20);
    int depth = 0;
    bool inStr = false, esc = false;
    char chunk[1 << 16];
    size_t got;
    while ((got = std::fread(chunk, 1, sizeof(chunk), f)) > 0) {
        for (size_t i = 0; i < got; ++i) {
            char c = chunk[i];
            buf.push_back(c);
            if (esc) { esc = false; continue; }
            if (c == '\\') { esc = true; continue; }
            if (c == '"') { inStr = !inStr; continue; }
            if (inStr) continue;
            if (c == '{') depth++;
            else if (c == '}') {
                depth--;
                if (depth == 0) { handle(buf); buf.clear(); }
            }
        }
    }
    std::fclose(f);
}

// ---------------------------------------------------------------------------
// Tiny field extractors over a single object's text. v11 objects are small
// (a few thousand ints at most) so linear scans are fine.
// ---------------------------------------------------------------------------
inline std::string getStr(const std::string& o, const char* key) {
    std::string k = std::string("\"") + key + "\"";
    size_t p = o.find(k); if (p == std::string::npos) return "";
    p = o.find(':', p); if (p == std::string::npos) return "";
    p = o.find('"', p); if (p == std::string::npos) return "";
    size_t e = o.find('"', p + 1); if (e == std::string::npos) return "";
    return o.substr(p + 1, e - p - 1);
}

// Parse the integer array value of `key` into out (cleared first).
inline void getUintArray(const std::string& o, const char* key, std::vector<unsigned>& out) {
    out.clear();
    std::string k = std::string("\"") + key + "\"";
    size_t p = o.find(k); if (p == std::string::npos) return;
    p = o.find('[', p); if (p == std::string::npos) return;
    size_t e = o.find(']', p); if (e == std::string::npos) return;
    const char* s = o.c_str() + p + 1;
    const char* end = o.c_str() + e;
    while (s < end) {
        while (s < end && (*s < '0' || *s > '9')) ++s;
        if (s >= end) break;
        unsigned v = 0;
        while (s < end && *s >= '0' && *s <= '9') { v = v * 10u + (unsigned)(*s - '0'); ++s; }
        out.push_back(v);
    }
}

// Parse a 2-D integer array (groundtruth: [[...],[...]]) → rows.
inline void getUint2D(const std::string& o, const char* key,
                      std::vector<std::vector<unsigned>>& rows) {
    rows.clear();
    std::string k = std::string("\"") + key + "\"";
    size_t p = o.find(k); if (p == std::string::npos) return;
    p = o.find('[', p); if (p == std::string::npos) return;   // outer [
    size_t outerEnd = o.rfind(']'); if (outerEnd == std::string::npos) return;
    size_t i = p + 1;
    while (i < outerEnd) {
        size_t rs = o.find('[', i); if (rs == std::string::npos || rs >= outerEnd) break;
        size_t re = o.find(']', rs); if (re == std::string::npos) break;
        std::vector<unsigned> row;
        const char* s = o.c_str() + rs + 1;
        const char* end = o.c_str() + re;
        while (s < end) {
            while (s < end && (*s < '0' || *s > '9')) ++s;
            if (s >= end) break;
            unsigned v = 0;
            while (s < end && *s >= '0' && *s <= '9') { v = v * 10u + (unsigned)(*s - '0'); ++s; }
            row.push_back(v);
        }
        rows.push_back(std::move(row));
        i = re + 1;
    }
}

// ---------------------------------------------------------------------------
// Per-item streaming: callbacks fire once per expanded id / query.
//   onInsert(unsigned id)
//   onDelete(unsigned id)
//   onQuery (unsigned queryId, const std::vector<unsigned>& gt)
// Order within the file is preserved (timeline correctness).
// ---------------------------------------------------------------------------
template <class OnInsert, class OnDelete, class OnQuery>
inline void stream(const std::string& path,
                   OnInsert onInsert, OnDelete onDelete, OnQuery onQuery) {
    std::vector<unsigned> ids;
    std::vector<std::vector<unsigned>> gt;
    forEachObject(path, [&](const std::string& obj) {
        std::string op = getStr(obj, "op_type");
        if (op == "insert") {
            getUintArray(obj, "vector_ids", ids);
            for (unsigned id : ids) onInsert(id);
        } else if (op == "delete") {
            getUintArray(obj, "vector_ids", ids);
            for (unsigned id : ids) onDelete(id);
        } else if (op == "search") {
            getUintArray(obj, "point_id", ids);
            getUint2D(obj, "groundtruth", gt);
            for (size_t i = 0; i < ids.size(); ++i) {
                static const std::vector<unsigned> empty;
                onQuery(ids[i], i < gt.size() ? gt[i] : empty);
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Per-batch streaming: callbacks receive whole batches (for drivers that want
// to call a BatchedInsert/Search API directly, e.g. FreshBANG).
//   onInsertBatch(const std::vector<unsigned>& ids)
//   onDeleteBatch(const std::vector<unsigned>& ids)
//   onQueryBatch (const std::vector<unsigned>& qids,
//                 const std::vector<std::vector<unsigned>>& gt)
// ---------------------------------------------------------------------------
template <class OnInsertB, class OnDeleteB, class OnQueryB>
inline void stream_batched(const std::string& path,
                           OnInsertB onInsertB, OnDeleteB onDeleteB, OnQueryB onQueryB) {
    std::vector<unsigned> ids;
    std::vector<std::vector<unsigned>> gt;
    forEachObject(path, [&](const std::string& obj) {
        std::string op = getStr(obj, "op_type");
        if (op == "insert")      { getUintArray(obj, "vector_ids", ids); onInsertB(ids); }
        else if (op == "delete") { getUintArray(obj, "vector_ids", ids); onDeleteB(ids); }
        else if (op == "search") {
            getUintArray(obj, "point_id", ids);
            getUint2D(obj, "groundtruth", gt);
            onQueryB(ids, gt);
        }
    });
}

} // namespace v11

#endif // V11_LOADER_HPP
