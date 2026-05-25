#define RAFT_SYSTEM_LITTLE_ENDIAN 1

#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/core/host_mdarray.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <map>

using IdxT = uint32_t;

// ================================================================
// Graph
// ================================================================

struct Graph {
    uint32_t n_rows;
    uint32_t degree;
    std::vector<IdxT> adj;
};

Graph load_graph(const std::string& path)
{
    raft::resources handle;
    Graph G;

    std::ifstream is(path, std::ios::binary);

    if (!is)
        throw std::runtime_error("Cannot open graph file: " + path);

    char dtype_string[4];
    is.read(dtype_string, 4);

    int version   = raft::deserialize_scalar<int>(handle, is);

    G.n_rows      = raft::deserialize_scalar<uint32_t>(handle, is);

    uint32_t dim  = raft::deserialize_scalar<uint32_t>(handle, is);

    G.degree      = raft::deserialize_scalar<uint32_t>(handle, is);

    int metric =
        raft::deserialize_scalar<int>(handle, is);

    std::cout << "Loaded graph:\n";
    std::cout << "  n_rows  = " << G.n_rows << "\n";
    std::cout << "  degree  = " << G.degree << "\n";
    std::cout << "  dim     = " << dim << "\n";
    std::cout << "  metric  = " << metric << "\n\n";

    auto md = raft::make_host_matrix<IdxT, int64_t>(
        G.n_rows,
        G.degree);

    raft::deserialize_mdspan(handle, is, md.view());

    G.adj.resize(size_t(G.n_rows) * G.degree);

    std::memcpy(
        G.adj.data(),
        md.view().data_handle(),
        G.adj.size() * sizeof(IdxT));

    return G;
}

// ================================================================
// Row Range Validation Helper
// ================================================================

struct RowRange {
    bool enabled = false;
    int start = 0;
    int end = -1;

    bool contains(int v) const {
        if (!enabled) return true;
        return v >= start && v <= end;
    }
};

// ================================================================
// SCC Result
// ================================================================

struct SCCResult {
    int num_scc;
    std::vector<int> comp;
    std::vector<int> size;
};

// ================================================================
// Tarjan SCC
// ================================================================

class TarjanSCC {
public:
    TarjanSCC(
        const Graph& G,
        const RowRange& range)
        : N(G.n_rows),
          K(G.degree),
          G(G),
          range(range),
          index(N, -1),
          low(N, 0),
          onstack(N, false),
          comp(N, -1),
          currentIndex(0),
          sccCount(0)
    {}

    SCCResult run()
    {
        int start =
            range.enabled ? range.start : 0;

        int end =
            range.enabled ? range.end : (N - 1);

        for (int v = start; v <= end; v++) {

            if (index[v] == -1)
                dfs(v);
        }

        // Count SCC sizes only within range
        std::vector<int> sizes(sccCount, 0);

        for (int v = start; v <= end; v++) {

            if (comp[v] >= 0)
                sizes[comp[v]]++;
        }

        return { sccCount, comp, sizes };
    }

private:
    int N, K;

    const Graph& G;
    const RowRange& range;

    std::vector<int> index;
    std::vector<int> low;
    std::vector<bool> onstack;
    std::vector<int> comp;

    std::stack<int> S;

    int currentIndex;
    int sccCount;

    void dfs(int v)
    {
        index[v] = low[v] = currentIndex++;

        S.push(v);

        onstack[v] = true;

        const IdxT* nbrs = &G.adj[v * K];

        for (int j = 0; j < K; j++) {

            int w = nbrs[j];

            if (w >= N)
                continue;

            // ----------------------------------------------------
            // Bail out if traversal leaves permitted range
            // ----------------------------------------------------
            if (!range.contains(w)) {

                std::cerr
                    << "ERROR: Traversal escaped row range.\n"
                    << "Edge: "
                    << v
                    << " -> "
                    << w
                    << "\n";

                std::exit(1);
            }

            if (index[w] == -1) {

                dfs(w);

                low[v] =
                    std::min(low[v], low[w]);
            }
            else if (onstack[w]) {

                low[v] =
                    std::min(low[v], index[w]);
            }
        }

        // --------------------------------------------------------
        // SCC root
        // --------------------------------------------------------
        if (low[v] == index[v]) {

            while (true) {

                int w = S.top();

                S.pop();

                onstack[w] = false;

                comp[w] = sccCount;

                if (w == v)
                    break;
            }

            sccCount++;
        }
    }
};

// ================================================================
// WCC Result
// ================================================================

struct WCCResult {
    int num_wcc;
    std::vector<int> comp;
    std::vector<int> size;
};

// ================================================================
// Weakly Connected Components
// ================================================================

WCCResult compute_wcc(
    const Graph& G,
    const RowRange& range)
{
    int N = G.n_rows;
    int K = G.degree;

    std::vector<int> comp(N, -1);

    int wccCount = 0;

    // ------------------------------------------------------------
    // Build undirected graph
    // ------------------------------------------------------------
    std::vector<std::vector<int>> undirected(N);

    int start =
        range.enabled ? range.start : 0;

    int end =
        range.enabled ? range.end : (N - 1);

    for (int u = start; u <= end; u++) {

        const IdxT* nbrs = &G.adj[u * K];

        for (int j = 0; j < K; j++) {

            int v = nbrs[j];

            if (v >= N || v == u)
                continue;

            // ----------------------------------------------------
            // Bail out if edge leaves row range
            // ----------------------------------------------------
            if (!range.contains(v)) {

                std::cerr
                    << "ERROR: Traversal escaped row range.\n"
                    << "Edge: "
                    << u
                    << " -> "
                    << v
                    << "\n";

                std::exit(1);
            }

            undirected[u].push_back(v);

            undirected[v].push_back(u);
        }
    }

    // ------------------------------------------------------------
    // BFS
    // ------------------------------------------------------------
    std::vector<int> stack;

    stack.reserve(N);

    for (int i = start; i <= end; i++) {

        if (comp[i] != -1)
            continue;

        comp[i] = wccCount;

        stack.clear();

        stack.push_back(i);

        for (size_t p = 0; p < stack.size(); p++) {

            int u = stack[p];

            for (int v : undirected[u]) {

                if (comp[v] == -1) {

                    comp[v] = wccCount;

                    stack.push_back(v);
                }
            }
        }

        wccCount++;
    }

    // ------------------------------------------------------------
    // Count sizes
    // ------------------------------------------------------------
    std::vector<int> sizes(wccCount, 0);

    for (int i = start; i <= end; i++) {

        if (comp[i] >= 0)
            sizes[comp[i]]++;
    }

    return { wccCount, comp, sizes };
}

// ================================================================
// Degree Statistics
// ================================================================

void print_degree_stats(
    const Graph& G,
    const RowRange& range)
{
    int N = G.n_rows;
    int K = G.degree;

    int start =
        range.enabled ? range.start : 0;

    int end =
        range.enabled ? range.end : (N - 1);

    int total_nodes =
        end - start + 1;

    long long total_out = 0;
    long long total_in  = 0;

    std::vector<int> indeg(N, 0);
    std::vector<int> outdeg(N, 0);

    // ------------------------------------------------------------
    // Compute degrees
    // ------------------------------------------------------------
    for (int u = start; u <= end; u++) {

        const IdxT* nbrs = &G.adj[u * K];

        for (int j = 0; j < K; j++) {

            int v = nbrs[j];

            if (v >= N || v == u)
                continue;

            // ----------------------------------------------------
            // Bail out if edge leaves range
            // ----------------------------------------------------
            if (!range.contains(v)) {

                std::cerr
                    << "ERROR: Traversal escaped row range.\n"
                    << "Edge: "
                    << u
                    << " -> "
                    << v
                    << "\n";

                std::exit(1);
            }

            outdeg[u]++;

            indeg[v]++;
        }
    }

    int min_out = -1;
    int max_out = 0;

    int min_in  = -1;
    int max_in  = 0;

    for (int i = start; i <= end; i++) {

        total_out += outdeg[i];
        total_in  += indeg[i];

        if (min_out == -1 || outdeg[i] < min_out)
            min_out = outdeg[i];

        if (outdeg[i] > max_out)
            max_out = outdeg[i];

        if (min_in == -1 || indeg[i] < min_in)
            min_in = indeg[i];

        if (indeg[i] > max_in)
            max_in = indeg[i];
    }

    double avg_out =
        double(total_out) / total_nodes;

    double avg_in =
        double(total_in) / total_nodes;

    int zero_in_degree = 0;

    for (int i = start; i <= end; i++) {

        if (indeg[i] == 0)
            zero_in_degree++;
    }

    double var_out = 0.0;
    double var_in  = 0.0;

    for (int i = start; i <= end; i++) {

        var_out +=
            (outdeg[i] - avg_out) *
            (outdeg[i] - avg_out);

        var_in +=
            (indeg[i] - avg_in) *
            (indeg[i] - avg_in);
    }

    var_out /= total_nodes;
    var_in  /= total_nodes;

    double std_out = std::sqrt(var_out);
    double std_in  = std::sqrt(var_in);

    std::cout << "\n=============================\n";
    std::cout << "Degree Statistics\n";
    std::cout << "=============================\n";

    std::cout << "Row Range : "
              << start
              << " - "
              << end
              << "\n";

    std::cout << "Rows Considered : "
              << total_nodes
              << "\n";

    std::cout << "\nOUT-DEGREE:\n";
    std::cout << "  Average = " << avg_out << "\n";
    std::cout << "  Min     = " << min_out << "\n";
    std::cout << "  Max     = " << max_out << "\n";
    std::cout << "  Std Dev = " << std_out << "\n";

    std::cout << "\nIN-DEGREE:\n";
    std::cout << "  Average = " << avg_in << "\n";
    std::cout << "  Min     = " << min_in << "\n";
    std::cout << "  Max     = " << max_in << "\n";
    std::cout << "  Std Dev = " << std_in << "\n";
    std::cout << "  Zero Degree Nodes = "
              << zero_in_degree << "\n";

    std::cout << "  Zero Degree %     = "
              << (100.0 * zero_in_degree / total_nodes)
              << "%\n";
}

// ================================================================
// Main
// ================================================================

int main(int argc, char** argv)
{
    RowRange range;

    int argi = 1;

    // ------------------------------------------------------------
    // Parse optional row range
    // ------------------------------------------------------------
    if (argc > 1 &&
        std::string(argv[argi]) == "--row-range")
    {
        if (argc <= argi + 2) {

            std::cerr
                << "--row-range requires start end\n";

            return 1;
        }

        range.enabled = true;

        range.start =
            std::stoi(argv[argi + 1]);

        range.end =
            std::stoi(argv[argi + 2]);

        argi += 3;
    }

    // ------------------------------------------------------------
    // Usage
    // ------------------------------------------------------------
    if (argc - argi < 1) {

        std::cerr
            << "Usage:\n";

        std::cerr
            << "  "
            << argv[0]
            << " graph.bin\n";

        std::cerr
            << "  "
            << argv[0]
            << " --row-range start end graph.bin\n";

        return 1;
    }

    // ------------------------------------------------------------
    // Load graph
    // ------------------------------------------------------------
    Graph G =
        load_graph(argv[argi]);

    // ------------------------------------------------------------
    // Validate row range
    // ------------------------------------------------------------
    if (range.enabled) {

        if (range.start < 0 ||
            range.end >= (int)G.n_rows ||
            range.start > range.end)
        {
            std::cerr
                << "Invalid row range\n";

            return 1;
        }
    }

    // ------------------------------------------------------------
    // Degree stats
    // ------------------------------------------------------------
    print_degree_stats(G, range);

    // ------------------------------------------------------------
    // SCC
    // ------------------------------------------------------------
    std::cout
        << "\nComputing STRONGLY connected components...\n";

    TarjanSCC solver(G, range);

    auto result = solver.run();

    std::cout
        << "\nStrongly Connected Components: "
        << result.num_scc
        << "\n\n";


for (int c = 0; c < result.num_scc; c++) {

    std::cout
        << "  SCC "
        << c
        << " size = "
        << result.size[c]
        << "  sample nodes: ";

    int printed = 0;

    int start =
        range.enabled ? range.start : 0;

    int end =
        range.enabled ? range.end : (G.n_rows - 1);

    for (int v = start;
         v <= end && printed < 10;
         v++)
    {
        if (result.comp[v] == c) {

            std::cout << v << " ";

            printed++;
        }
    }

    std::cout << "\n";
}

    int largest = 0;

    for (int i = 1; i < result.num_scc; i++) {

        if (result.size[i] > result.size[largest])
            largest = i;
    }

    std::cout
        << "\nLargest SCC: "
        << largest
        << " (size = "
        << result.size[largest]
        << ")\n";

    // ------------------------------------------------------------
    // WCC
    // ------------------------------------------------------------
    std::cout
        << "\nComputing WEAKLY connected components...\n";

    auto wcc =
        compute_wcc(G, range);

    std::cout
        << "\nWeakly Connected Components: "
        << wcc.num_wcc
        << "\n\n";

    for (int i = 0; i < wcc.num_wcc; i++) {

        std::cout
            << "  WCC "
            << i
            << " size = "
            << wcc.size[i]
            << "\n";
    }

    int largest_wcc = 0;

    for (int i = 1; i < wcc.num_wcc; i++) {

        if (wcc.size[i] >
            wcc.size[largest_wcc])
        {
            largest_wcc = i;
        }
    }

    std::cout
        << "\nLargest WCC: "
        << largest_wcc
        << " (size = "
        << wcc.size[largest_wcc]
        << ")\n";

    return 0;
}
