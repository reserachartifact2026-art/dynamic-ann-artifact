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
// Usage:
//   v11::VecStore base(v11::loadVectors("base.fbin"));   // or loadVectors of .fvecs
//   v11::stream("workload.jsonl",
//       [&](unsigned id)              { /* insert: base.row(id) */ },
//       [&](unsigned id)              { /* delete id */ },
//       [&](unsigned qid, const std::vector<unsigned>& gt) { /* query */ });
//
// The callbacks fire once per EXPANDED item (per id / per query), so a consumer
// that wants per-vector events gets them for free; a consumer that wants whole
// batches can use stream_batched() instead.

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
// Vector storage: row-major [n x dim] float32, loaded from .fbin or .fvecs.
// Only the first `max_rows` rows are read (0 = all) so we don't pull a 35M-row
// wikipedia base when the workload only touches the first 1M.
// ---------------------------------------------------------------------------
template <typename T = float>
struct VecStoreT {
    std::vector<T> data;   // n * dim
    size_t n   = 0;
    size_t dim = 0;
    const T* row(size_t i) const { return data.data() + i * dim; }
};
using VecStore = VecStoreT<float>;   // default: float32 (unchanged for existing callers)

// .fbin : [uint32 n][uint32 dim][n*dim float32]
template <typename T = float>
inline VecStoreT<T> loadFbin(const std::string& path, size_t max_rows = 0) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("loadFbin: cannot open " + path);
    uint32_t n = 0, dim = 0;
    if (std::fread(&n, 4, 1, f) != 1 || std::fread(&dim, 4, 1, f) != 1) {
        std::fclose(f); throw std::runtime_error("loadFbin: short header " + path);
    }
    size_t rows = (max_rows && max_rows < n) ? max_rows : n;
    VecStoreT<T> vs; vs.n = rows; vs.dim = dim;
    vs.data.resize(rows * (size_t)dim);
    if (std::fread(vs.data.data(), sizeof(T), rows * (size_t)dim, f) != rows * (size_t)dim) {
        std::fclose(f); throw std::runtime_error("loadFbin: short data " + path);
    }
    std::fclose(f);
    return vs;
}

// .fvecs : repeated [int32 dim][dim float32]
template <typename T = float>
inline VecStoreT<T> loadFvecs(const std::string& path, size_t max_rows = 0) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("loadFvecs: cannot open " + path);
    int32_t dim0 = 0;
    if (std::fread(&dim0, 4, 1, f) != 1) { std::fclose(f); throw std::runtime_error("loadFvecs: empty " + path); }
    std::fseek(f, 0, SEEK_END);
    long bytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    size_t rec = 4 + (size_t)dim0 * sizeof(T);
    size_t n_total = (size_t)bytes / rec;
    size_t rows = (max_rows && max_rows < n_total) ? max_rows : n_total;
    VecStoreT<T> vs; vs.n = rows; vs.dim = (size_t)dim0;
    vs.data.resize(rows * (size_t)dim0);
    for (size_t i = 0; i < rows; ++i) {
        int32_t d;
        if (std::fread(&d, 4, 1, f) != 1 || d != dim0) { std::fclose(f); throw std::runtime_error("loadFvecs: dim mismatch " + path); }
        if (std::fread(vs.data.data() + i * (size_t)dim0, sizeof(T), dim0, f) != (size_t)dim0) {
            std::fclose(f); throw std::runtime_error("loadFvecs: short row " + path);
        }
    }
    std::fclose(f);
    return vs;
}

// Dispatch on extension.
template <typename T = float>
inline VecStoreT<T> loadVectors(const std::string& path, size_t max_rows = 0) {
    if (path.size() >= 6 && path.compare(path.size() - 6, 6, ".fvecs") == 0)
        return loadFvecs<T>(path, max_rows);
    return loadFbin<T>(path, max_rows);   // .fbin / .bin / .u8bin
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
