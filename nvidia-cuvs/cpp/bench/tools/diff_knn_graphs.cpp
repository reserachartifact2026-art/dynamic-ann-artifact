#include <raft/core/host_mdarray.hpp>
#include <raft/core/serialize.hpp>
#include <raft/core/resources.hpp>
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <vector>
#include <stdexcept>
#include <cstring>

using IdxT = uint32_t;

// -----------------------------
// Read CAGRA / kNN graph
// -----------------------------
struct Graph {
    uint32_t n_rows;
    uint32_t dim;
    uint32_t degree;
    int metric;
    std::vector<IdxT> adj; // size = n_rows * degree
};

Graph load_graph(const std::string& path)
{
    raft::resources handle;
    Graph G;

    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("Cannot open graph file: " + path);

    char dtype_string[4];
    is.read(dtype_string, 4);  
    int version = raft::deserialize_scalar<int>(handle, is);
    G.n_rows   = raft::deserialize_scalar<uint32_t>(handle, is);
    G.dim      = raft::deserialize_scalar<uint32_t>(handle, is);
    G.degree   = raft::deserialize_scalar<uint32_t>(handle, is);
    G.metric   = raft::deserialize_scalar<int>(handle, is);

    std::cout << "Loaded: " << path << "\n";
    std::cout << "  n_rows=" << G.n_rows << " degree=" << G.degree << " dim=" << G.dim << "\n";

    G.adj.resize((size_t)G.n_rows * G.degree);
    auto md = raft::make_host_matrix<IdxT, int64_t>(G.n_rows, G.degree);
    raft::deserialize_mdspan(handle, is, md.view());
/*
    // Copy into vector
    for (size_t i = 0; i < G.adj.size(); i++) {
        G.adj[i] = md.data()[i];
    }
*/

auto* ptr = md.view().data_handle();
std::memcpy(G.adj.data(), ptr, G.adj.size() * sizeof(IdxT));

    return G;
}

// -----------------------------
// Main diff logic
// -----------------------------
int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <gold_graph.bin> <new_graph.bin> [print_mismatches]\n";
        return 1;
    }

    int inspect_row = -1;

    // Search for --row argument
    for (int i = 3; i < argc; i++) {
    if (std::string(argv[i]) == "--row" && i + 1 < argc) {
        inspect_row = std::stoi(argv[i + 1]);
    }
    }


    bool print_mismatches = 0;
    // Search for --print argument
    for (int i = 3; i < argc; i++) {
    if (std::string(argv[i]) == "--print" ) {
        print_mismatches = 1;
    }
}



    Graph gold = load_graph(argv[1]);
    Graph test = load_graph(argv[2]);

    if (gold.n_rows != test.n_rows) {
        std::cerr << "ERROR: n_rows differ: gold=" << gold.n_rows
                  << " test=" << test.n_rows << "\n";
        return 1;
    }
    if (gold.degree != test.degree) {
        std::cerr << "ERROR: graph degree mismatch: gold=" << gold.degree
                  << " test=" << test.degree << "\n";
        return 1;
    }

    uint32_t N = gold.n_rows;
    uint32_t K = gold.degree;

    size_t total_wrong = 0;

    std::cout << "\nComparing graphs...\n";

    for (uint32_t row = 0; row < N; row++) {

if (inspect_row >= 0) {
    int r = inspect_row;
    if (r < 0 || r >= int(N)) {
        std::cerr << "Row out of range: " << r << "\n";
        return 1;
    }

    const IdxT* g = &gold.adj[r * K];
    const IdxT* t = &test.adj[r * K];

    std::unordered_set<IdxT> gold_set(g, g + K);
    std::unordered_set<IdxT> test_set(t, t + K);

    std::vector<IdxT> missing, extra, common;

    for (auto v : gold_set)
        if (!test_set.count(v)) missing.push_back(v);

    for (auto v : test_set)
        if (!gold_set.count(v)) extra.push_back(v);

    for (auto v : test_set)
        if (gold_set.count(v)) common.push_back(v);

    std::cout << "\n===== Row " << r << " diff =====\n";
    std::cout << "Gold neighbors : ";
    for (uint32_t j = 0; j < K; j++) std::cout << g[j] << " ";
    std::cout << "\n";

    std::cout << "Test neighbors : ";
    for (uint32_t j = 0; j < K; j++) std::cout << t[j] << " ";
    std::cout << "\n";

    std::cout << "\nMissing (in gold but not in test): ";
    for (auto v : missing) std::cout << v << " ";
    if (missing.empty()) std::cout << "None";
    std::cout << "\n";

    std::cout << "Extra (in test but not in gold): ";
    for (auto v : extra) std::cout << v << " ";
    if (extra.empty()) std::cout << "None";
    std::cout << "\n";

    std::cout << "Common neighbors: ";
    for (auto v : common) std::cout << v << " ";
    std::cout << "\n";

    double err = double(extra.size()) / double(K);
    std::cout << "\nRow accuracy: " << (1.0 - err)
              << "  (wrong=" << extra.size() << "/" << K << ")\n";

    return 0; // Do NOT run full diff
}


        const IdxT* g = &gold.adj[row * K];
        const IdxT* t = &test.adj[row * K];

        std::unordered_set<IdxT> gold_set(g, g + K);

        size_t wrong = 0;
        for (uint32_t j = 0; j < K; j++) {
            if (gold_set.count(t[j]) == 0) {
                wrong++;
            }
        }

        total_wrong += wrong;

        if (print_mismatches && wrong > 0) {
            std::cout << "Row " << row << ": wrong=" << wrong << "/" << K << "\n";
            std::cout << "  gold: ";
            for (uint32_t j = 0; j < K; j++) std::cout << g[j] << " ";
            std::cout << "\n  test: ";
            for (uint32_t j = 0; j < K; j++) std::cout << t[j] << " ";
            std::cout << "\n";
        }




    }

    double total_slots = double(N) * K;
    double error_rate  = double(total_wrong) / total_slots;

    std::cout << "\n===== FINAL SCORE =====\n";
    std::cout << "Total wrong neighbors = " << total_wrong << " / " << total_slots << "\n";
    std::cout << "Error rate = " << error_rate << "\n";
    std::cout << "Accuracy   = " << (1.0 - error_rate) << "\n";

    return 0;
}

