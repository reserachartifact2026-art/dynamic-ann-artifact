#include <raft/core/resources.hpp>
#include <raft/core/serialize.hpp>
#include <raft/core/host_mdarray.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <stack>
#include <cstring>
#include <stdexcept>

using IdxT = uint32_t;

// ---------------------------------------------
// Load cuVS / CAGRA kNN graph
// ---------------------------------------------
struct Graph {
    uint32_t n_rows;
    uint32_t degree;
    std::vector<IdxT> adj; // adjacency list: n_rows * degree
};

Graph load_graph(const std::string& path)
{
    raft::resources handle;
    Graph G;

    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("Cannot open graph file: " + path);

    char dtype_string[4];
    is.read(dtype_string, 4);

    int version   = raft::deserialize_scalar<int>(handle, is);
    G.n_rows      = raft::deserialize_scalar<uint32_t>(handle, is);
    uint32_t dim  = raft::deserialize_scalar<uint32_t>(handle, is);
    G.degree      = raft::deserialize_scalar<uint32_t>(handle, is);
    int metric = raft::deserialize_scalar<int>(handle, is);

    std::cout << "Loaded graph:\n";
    std::cout << "  n_rows  = " << G.n_rows << "\n";
    std::cout << "  degree  = " << G.degree << "\n";
    std::cout << "  dim     = " << dim << "\n";
    std::cout << "  metric  = " << metric << "\n\n";

    auto md = raft::make_host_matrix<IdxT, int64_t>(G.n_rows, G.degree);
    raft::deserialize_mdspan(handle, is, md.view());

    G.adj.resize(size_t(G.n_rows) * G.degree);
    std::memcpy(G.adj.data(), md.view().data_handle(),
                G.adj.size() * sizeof(IdxT));

    return G;
}

// ---------------------------------------------
// TARJAN's Strongly Connected Components
// ---------------------------------------------
struct SCCResult {
    int num_scc;
    std::vector<int> comp;  // comp[i] = SCC index for node i
    std::vector<int> size;  // size of each SCC
};

class TarjanSCC {
public:
    TarjanSCC(const Graph& G)
        : N(G.n_rows), K(G.degree), G(G),
          index(N, -1), low(N, 0), onstack(N, false), comp(N, -1),
          currentIndex(0), sccCount(0) {}

    SCCResult run() {
        for (int v = 0; v < N; v++) {
            if (index[v] == -1) dfs(v);
        }

        // Count SCC sizes
        std::vector<int> sizes(sccCount, 0);
        for (int v = 0; v < N; v++)
            sizes[comp[v]]++;

        return { sccCount, comp, sizes };
    }

private:
    int N, K;
    const Graph& G;

    std::vector<int> index;    // DFS order index
    std::vector<int> low;      // low-link values
    std::vector<bool> onstack;
    std::vector<int> comp;     // component assignment

    std::stack<int> S;
    int currentIndex;
    int sccCount;

    void dfs(int v) {
        index[v] = low[v] = currentIndex++;
        S.push(v);
        onstack[v] = true;

        const IdxT* nbrs = &G.adj[v * K];

        for (int j = 0; j < K; j++) {
            int w = nbrs[j];
            if (w >= N) continue; // out of bounds safety

            if (index[w] == -1) {
                dfs(w);
                low[v] = std::min(low[v], low[w]);
            } 
            else if (onstack[w]) {
                low[v] = std::min(low[v], index[w]);
            }
        }

        // If v is root of an SCC
        if (low[v] == index[v]) {
            while (true) {
                int w = S.top(); S.pop();
                onstack[w] = false;
                comp[w] = sccCount;
                if (w == v) break;
            }
            sccCount++;
        }
    }
};

// ---------------------------------------------
// Weakly Connected Components (undirected BFS)
// ---------------------------------------------
struct WCCResult {
    int num_wcc;
    std::vector<int> comp;   // comp[i] = WCC id of node i
    std::vector<int> size;   // size of each WCC
};

WCCResult compute_wcc(const Graph& G)
{
    int N = G.n_rows;
    int K = G.degree;

    std::vector<int> comp(N, -1);
    int wccCount = 0;

    // Build undirected adjacency list
    std::vector<std::vector<int>> undirected(N);
    undirected.reserve(N);

    for (int u = 0; u < N; u++) {
        const IdxT* nbrs = &G.adj[u * K];
        for (int j = 0; j < K; j++) {
            int v = nbrs[j];
            if (v >= N || v == u) continue;

            undirected[u].push_back(v);
            undirected[v].push_back(u);  // reverse edge
        }
    }

    // BFS over undirected graph
    std::vector<int> stack;
    stack.reserve(N);

    for (int i = 0; i < N; i++) {
        if (comp[i] != -1) continue;

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

    // Count WCC sizes
    std::vector<int> sizes(wccCount, 0);
    for (int i = 0; i < N; i++)
        sizes[comp[i]]++;

    return { wccCount, comp, sizes };
}

#include <cmath>
#include <map>

void print_degree_stats(const Graph& G)
{
    int N = G.n_rows;
    int K = G.degree;

    long long total_out = 0, total_in = 0;

    std::vector<int> indeg(N, 0);
    std::vector<int> outdeg(N, 0);

    // Compute degrees
    for (int u = 0; u < N; u++) {
        const IdxT* nbrs = &G.adj[u * K];

        for (int j = 0; j < K; j++) {
            int v = nbrs[j];
            if (v >= N || v == u) continue;

            outdeg[u]++;
            indeg[v]++;
        }
    }

    int min_out = outdeg[0], max_out = outdeg[0];
    int min_in  = indeg[0],  max_in  = indeg[0];

    for (int i = 0; i < N; i++) {
        total_out += outdeg[i];
        total_in  += indeg[i];

        min_out = std::min(min_out, outdeg[i]);
        max_out = std::max(max_out, outdeg[i]);

        min_in = std::min(min_in, indeg[i]);
        max_in = std::max(max_in, indeg[i]);
    }

    double avg_out = double(total_out) / N;
    double avg_in  = double(total_in) / N;

    int zero_in_degree = 0;

    // Count zero in-degree nodes
    for (int i = 0; i < N; i++) {
        if (indeg[i] == 0)
	    zero_in_degree++;
    }


    // Standard deviation
    double var_out = 0.0, var_in = 0.0;

    for (int i = 0; i < N; i++) {
        var_out += (outdeg[i] - avg_out) * (outdeg[i] - avg_out);
        var_in  += (indeg[i] - avg_in)   * (indeg[i] - avg_in);
    }

    var_out /= N;
    var_in  /= N;

    double std_out = std::sqrt(var_out);
    double std_in  = std::sqrt(var_in);

    std::cout << "\n=============================\n";
    std::cout << "Degree Statistics\n";
    std::cout << "=============================\n";

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
    std::cout << "  Zero Degree Nodes = " << zero_in_degree << "\n";
    std::cout << "  Zero Degree %     = "
          << (100.0 * zero_in_degree / N) << "%\n";


    // Histogram for in-degree
    std::map<int,int> hist_in, hist_out;

    for (int i = 0; i < N; i++) {
        hist_in[indeg[i]]++;
        hist_out[outdeg[i]]++;
    }

    std::cout << "\n=============================\n";
    std::cout << "In-Degree Histogram (top 20 bins)\n";
    std::cout << "degree : count\n";

    int shown = 0;
    for (auto& p : hist_in) {
        std::cout << "  " << p.first << " : " << p.second << "\n";
        if (++shown == 20) break;
    }

    std::cout << "\n=============================\n";
    std::cout << "Out-Degree Histogram (top 20 bins)\n";
    std::cout << "degree : count\n";

    shown = 0;
    for (auto& p : hist_out) {
        std::cout << "  " << p.first << " : " << p.second << "\n";
        if (++shown == 20) break;
    }
}

// ---------------------------------------------
// Main
// ---------------------------------------------
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <graph.bin>\n";
        return 1;
    }

    Graph G = load_graph(argv[1]);

    print_degree_stats(G);

    std::cout << "Computing STRONGLY connected components (directed)...\n";

    TarjanSCC solver(G);
    auto result = solver.run();

    std::cout << "\nStrongly Connected Components: " 
              << result.num_scc << "\n\n";

    for (int i = 0; i < result.num_scc; i++) {
        std::cout << "  SCC " << i << " size = " << result.size[i] << "\n";
    }

    // Find largest SCC
    int largest = 0;
    for (int i = 1; i < result.num_scc; i++)
        if (result.size[i] > result.size[largest])
            largest = i;

    std::cout << "\nLargest SCC: " << largest 
              << "  (size = " << result.size[largest] << ")\n";


    std::cout << "\nComputing WEAKLY connected components (undirected)...\n";

    auto wcc = compute_wcc(G);

    std::cout << "\nWeakly Connected Components: "
              << wcc.num_wcc << "\n\n";

    for (int i = 0; i < wcc.num_wcc; i++) {
        std::cout << "  WCC " << i << " size = " << wcc.size[i] << "\n";
    }

    // Find largest WCC
    int largest_wcc = 0;
    for (int i = 1; i < wcc.num_wcc; i++)
        if (wcc.size[i] > wcc.size[largest_wcc])
            largest_wcc = i;

    std::cout << "\nLargest WCC: " << largest_wcc
              << "  (size = " << wcc.size[largest_wcc] << ")\n";

for (int c = 0; c < wcc.num_wcc; c++) {
    std::cout << "WCC " << c << " sample nodes: ";
    int printed = 0;
    for (int i = 0; i < G.n_rows && printed < 10; i++) {
        if (wcc.comp[i] == c) {
            std::cout << i << " ";
            printed++;
        }
    }
    std::cout << "\n";
}


    return 0;
}

