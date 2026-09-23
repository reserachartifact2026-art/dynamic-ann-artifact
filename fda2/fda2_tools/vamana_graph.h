#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <iostream>

namespace fda2 {
using IdxT = uint32_t;

enum class VectorType { Float32, UInt8 };

struct Graph {
    uint64_t n_rows = 0;
    uint32_t dim = 0;
    uint32_t max_degree = 0;
    VectorType vector_type = VectorType::Float32;

    // Fixed-capacity adjacency storage. Only [offset, offset+row_degree)
    // is valid for a row.
    std::vector<uint32_t> row_degree;
    std::vector<IdxT> adj;

    IdxT* neighbors(IdxT u) {
        return adj.data() + static_cast<size_t>(u) * max_degree;
    }

    const IdxT* neighbors(IdxT u) const {
        return adj.data() + static_cast<size_t>(u) * max_degree;
    }

    uint32_t degree(IdxT u) const { return row_degree[u]; }

    bool has_edge(IdxT u, IdxT v) const {
        if (u >= n_rows) return false;
        const auto* p = neighbors(u);
        for (uint32_t j = 0; j < row_degree[u]; ++j)
            if (p[j] == v) return true;
        return false;
    }
};

struct Options {
    VectorType vector_type = VectorType::Float32;
    uint64_t nodes = 0;
    uint32_t dim = 0;
    uint32_t degree = 0;
};

inline uint64_t record_size(const Options& o) {
    uint64_t vb = o.vector_type == VectorType::Float32
        ? uint64_t(o.dim) * sizeof(float)
        : uint64_t(o.dim) * sizeof(uint8_t);
    return vb + sizeof(uint32_t) + uint64_t(o.degree) * sizeof(IdxT);
}

inline uint64_t get_file_size(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("Cannot open graph: " + path);
    auto p = in.tellg();
    if (p < 0) throw std::runtime_error("Cannot determine file size");
    return static_cast<uint64_t>(p);
}

inline Graph load_graph(const std::string& path, const Options& o, bool verbose=true) {
    const uint64_t rec = record_size(o);
    const uint64_t expected = o.nodes * rec;
    const uint64_t actual = get_file_size(path);
    if (actual != expected) {
        throw std::runtime_error(
            "FDA2 Vamana file-size mismatch: expected " +
            std::to_string(expected) + ", got " + std::to_string(actual));
    }

    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open graph: " + path);

    Graph g;
    g.n_rows = o.nodes;
    g.dim = o.dim;
    g.max_degree = o.degree;
    g.vector_type = o.vector_type;
    g.row_degree.resize(static_cast<size_t>(o.nodes));
    g.adj.resize(static_cast<size_t>(o.nodes) * o.degree);

    const uint64_t vector_bytes = o.vector_type == VectorType::Float32
        ? uint64_t(o.dim) * sizeof(float)
        : uint64_t(o.dim);
    std::vector<char> vecbuf(static_cast<size_t>(vector_bytes));

    for (uint64_t u = 0; u < o.nodes; ++u) {
        in.read(vecbuf.data(), static_cast<std::streamsize>(vector_bytes));
        if (!in) throw std::runtime_error("Error reading vector at node " + std::to_string(u));

        uint32_t d = 0;
        in.read(reinterpret_cast<char*>(&d), sizeof(d));
        if (!in) throw std::runtime_error("Error reading degree at node " + std::to_string(u));
        if (d > o.degree) throw std::runtime_error("Degree exceeds R at node " + std::to_string(u));
        g.row_degree[u] = d;

        auto* dst = g.adj.data() + static_cast<size_t>(u) * o.degree;
        in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(uint64_t(o.degree) * sizeof(IdxT)));
        if (!in) throw std::runtime_error("Error reading adjacency at node " + std::to_string(u));

        if (verbose && u > 0 && u % 1000000ULL == 0)
            std::cerr << "Loaded " << u << " / " << o.nodes << " nodes\r" << std::flush;
    }
    if (verbose) std::cerr << "\n";

    if (verbose) {
        std::cout << "Loaded FDA2 Vamana graph\n"
                  << "  Nodes       : " << g.n_rows << "\n"
                  << "  Dimension   : " << g.dim << "\n"
                  << "  R           : " << g.max_degree << "\n"
                  << "  Vector type : " << (g.vector_type == VectorType::Float32 ? "float32" : "uint8") << "\n"
                  << "  Record size : " << rec << " bytes\n";
    }
    return g;
}

inline std::vector<std::vector<IdxT>> build_reverse_graph(const Graph& g) {
    std::vector<uint64_t> counts(static_cast<size_t>(g.n_rows), 0);
    for (IdxT u = 0; u < g.n_rows; ++u) {
        const auto* p = g.neighbors(u);
        for (uint32_t j = 0; j < g.row_degree[u]; ++j) {
            IdxT v = p[j];
            if (v < g.n_rows && v != u) ++counts[v];
        }
    }
    std::vector<std::vector<IdxT>> incoming(static_cast<size_t>(g.n_rows));
    for (IdxT v = 0; v < g.n_rows; ++v)
        incoming[v].reserve(static_cast<size_t>(counts[v]));
    for (IdxT u = 0; u < g.n_rows; ++u) {
        const auto* p = g.neighbors(u);
        for (uint32_t j = 0; j < g.row_degree[u]; ++j) {
            IdxT v = p[j];
            if (v < g.n_rows && v != u) incoming[v].push_back(u);
        }
    }
    return incoming;
}

inline void sort_adjacency(Graph& g) {
    #pragma omp parallel for schedule(dynamic, 1024)
    for (int64_t i = 0; i < static_cast<int64_t>(g.n_rows); ++i) {
        auto* p = g.neighbors(static_cast<IdxT>(i));
        std::sort(p, p + g.row_degree[i]);
    }
}

} // namespace fda2
