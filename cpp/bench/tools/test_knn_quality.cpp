#define RAFT_SYSTEM_LITTLE_ENDIAN 1

#include <raft/core/serialize.hpp>
#include <raft/core/host_mdarray.hpp>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <cstring>
#include <omp.h>

using IdxT = uint32_t;

// ================================================================
// Graph
// ================================================================

struct Graph {
    uint32_t n_rows;
    uint32_t dim;
    uint32_t degree;
    std::vector<IdxT> adj;
};

Graph load_graph(const std::string& path)
{
    std::ifstream is(path, std::ios::binary);

    if (!is)
        throw std::runtime_error("Cannot open graph file");

    raft::resources handle;

    Graph G;

    // ------------------------------------------------------------
    // dtype string
    // ------------------------------------------------------------
    char dtype[4];
    is.read(dtype, 4);

    // ------------------------------------------------------------
    // version
    // ------------------------------------------------------------
    int version = raft::deserialize_scalar<int>(handle, is);

    // ------------------------------------------------------------
    // graph fields
    // ------------------------------------------------------------
    G.n_rows = raft::deserialize_scalar<uint32_t>(handle, is);
    G.dim    = raft::deserialize_scalar<uint32_t>(handle, is);
    G.degree = raft::deserialize_scalar<uint32_t>(handle, is);

    // ------------------------------------------------------------
    // metric
    // ------------------------------------------------------------
    int metric = raft::deserialize_scalar<int>(handle, is);

    std::cout << "Loaded graph:\n";
    std::cout << "  n_rows = " << G.n_rows << "\n";
    std::cout << "  dim    = " << G.dim << "\n";
    std::cout << "  degree = " << G.degree << "\n";

    // ------------------------------------------------------------
    // adjacency matrix
    // ------------------------------------------------------------
    auto md = raft::make_host_matrix<IdxT, int64_t>(
        G.n_rows,
        G.degree);

    raft::deserialize_mdspan(handle, is, md.view());

    G.adj.resize(size_t(G.n_rows) * G.degree);

    std::memcpy(
        G.adj.data(),
        md.view().data_handle(),
        G.adj.size() * sizeof(IdxT));

    // ------------------------------------------------------------
    // trailing content map
    // ------------------------------------------------------------
    uint32_t content_map =
        raft::deserialize_scalar<uint32_t>(handle, is);

    return G;
}

// ================================================================
// FBIN Loader
// ================================================================

std::vector<float> load_fbin(
    const std::string& path,
    uint32_t& n,
    uint32_t& dim)
{
    std::ifstream is(path, std::ios::binary);

    if (!is)
        throw std::runtime_error("Cannot open fbin");

    is.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    is.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

    std::vector<float> data(size_t(n) * dim);

    is.read(reinterpret_cast<char*>(data.data()),
            data.size() * sizeof(float));

    return data;
}

// ================================================================
// L2 Distance
// ================================================================

inline float l2(
    const float* a,
    const float* b,
    uint32_t dim)
{
    float s = 0.0f;

    for (uint32_t i = 0; i < dim; i++) {

        float d = a[i] - b[i];
        s += d * d;
    }

    return s;
}

// ================================================================
// Main
// ================================================================

int main(int argc, char** argv)
{
    bool check_order = false;
    int order_check_k = -1;

    bool use_row_range = false;
    int row_start = 0;
    int row_end = -1;

    int argi = 1;

    // ============================================================
    // Parse CLI options
    // ============================================================

    while (argi < argc) {

        std::string arg = argv[argi];

        // --------------------------------------------------------
        // --check-order <K>
        // --------------------------------------------------------
        if (arg == "--check-order") {

            check_order = true;

            if (argi + 1 >= argc) {

                std::cerr
                    << "--check-order requires a value\n";

                return 1;
            }

            order_check_k = std::stoi(argv[argi + 1]);

            argi += 2;
        }

        // --------------------------------------------------------
        // --row-range <start> <end>
        // --------------------------------------------------------
        else if (arg == "--row-range") {

            if (argi + 2 >= argc) {

                std::cerr
                    << "--row-range requires start and end\n";

                return 1;
            }

            use_row_range = true;

            row_start = std::stoi(argv[argi + 1]);
            row_end   = std::stoi(argv[argi + 2]);

            argi += 3;
        }

        else {
            break;
        }
    }

    // ============================================================
    // Usage
    // ============================================================

    if (argc - argi < 2) {

        std::cerr << "Usage:\n";

        std::cerr
            << "  " << argv[0]
            << " [--row-range start end]"
            << " graph.bin vectors.fbin\n";

        std::cerr
            << "  " << argv[0]
            << " --check-order <K>"
            << " [--row-range start end]"
            << " graph.bin vectors.fbin\n";

        return 1;
    }

    // ============================================================
    // Load graph
    // ============================================================

    Graph G = load_graph(argv[argi]);

    // ============================================================
    // Load vectors
    // ============================================================

    uint32_t n_vecs, dim;

    auto vecs = load_fbin(
        argv[argi + 1],
        n_vecs,
        dim);

    // ============================================================
    // Validation
    // ============================================================

    if (dim != G.dim || n_vecs < G.n_rows) {

        std::cerr << "Dimension mismatch\n";
        return 1;
    }

    int N = G.n_rows;
    int K = G.degree;

    // ------------------------------------------------------------
    // Validate ordering prefix K
    // ------------------------------------------------------------

    if (check_order) {

        if (order_check_k <= 0 ||
            order_check_k > K)
        {
            std::cerr
                << "Invalid --check-order K.\n"
                << "Must satisfy: 1 <= K <= graph_degree("
                << K << ")\n";

            return 1;
        }
    }

    // ------------------------------------------------------------
    // Validate row range
    // ------------------------------------------------------------

    if (use_row_range) {

        if (row_start < 0 ||
            row_end >= N ||
            row_start > row_end)
        {
            std::cerr
                << "Invalid row range\n";

            return 1;
        }
    }
    else {

        row_start = 0;
        row_end   = N - 1;
    }

    int total_rows_considered =
        row_end - row_start + 1;

    // ============================================================
    // DEFAULT MODE:
    // Recall@K against brute-force groundtruth
    // ============================================================

    if (!check_order)
    {
        double total_recall = 0.0;

        #pragma omp parallel
        {
            std::vector<std::pair<float,int>> dist;

            dist.reserve(N);

            double local_sum = 0.0;

            #pragma omp for schedule(dynamic, 4)
            for (int i = row_start; i <= row_end; i++) {

                dist.clear();

                const float* vi =
                    &vecs[size_t(i) * dim];

                // ------------------------------------------------
                // brute-force GT
                // ------------------------------------------------
                for (int j = 0; j < N; j++) {

                    if (i == j)
                        continue;

                    const float* vj =
                        &vecs[size_t(j) * dim];

                    dist.emplace_back(
                        l2(vi, vj, dim),
                        j);
                }

                // ------------------------------------------------
                // top-K GT
                // ------------------------------------------------
                std::nth_element(
                    dist.begin(),
                    dist.begin() + K,
                    dist.end());

                // ------------------------------------------------
                // GT set
                // ------------------------------------------------
                std::unordered_set<int> gt;

                for (int k = 0; k < K; k++)
                    gt.insert(dist[k].second);

                // ------------------------------------------------
                // compare with graph neighbors
                // ------------------------------------------------
                int correct = 0;

                const IdxT* nbrs =
                    &G.adj[i * K];

                for (int j = 0; j < K; j++) {

                    if (gt.count(nbrs[j]))
                        correct++;
                }

                local_sum += double(correct) / K;
            }

            #pragma omp atomic
            total_recall += local_sum;
        }

        double avg =
            total_recall / total_rows_considered;

        std::cout << "\n========================\n";
        std::cout << "Recall@K\n";
        std::cout << "========================\n";

        std::cout << "Nodes                : "
                  << N << "\n";

        std::cout << "K                    : "
                  << K << "\n";

        std::cout << "Row Range            : "
                  << row_start
                  << " - "
                  << row_end
                  << "\n";

        std::cout << "Rows Considered      : "
                  << total_rows_considered
                  << "\n";

        std::cout << "Avg Recall           : "
                  << avg << "\n";
    }

    // ============================================================
    // OPTIONAL MODE:
    // Neighbor ordering quality
    // ============================================================

    else
    {
        double sorted_rows = 0.0;

        #pragma omp parallel
        {
            double local_sorted = 0.0;

            #pragma omp for schedule(dynamic, 1024)
            for (int i = row_start; i <= row_end; i++) {

                const float* vi =
                    &vecs[size_t(i) * dim];

                const IdxT* nbrs =
                    &G.adj[i * K];

                bool sorted = true;

                float prev_dist = -1.0f;

                // ------------------------------------------------
                // check only prefix:
                // first order_check_k neighbors
                // ------------------------------------------------
                for (int j = 0; j < order_check_k; j++) {

                    int nid = nbrs[j];

                    // --------------------------------------------
                    // invalid neighbor check
                    // --------------------------------------------
                    if (nid >= N) {

                        sorted = false;
                        break;
                    }

                    const float* vn =
                        &vecs[size_t(nid) * dim];

                    float d = l2(vi, vn, dim);

                    // --------------------------------------------
                    // ascending distance order check
                    // --------------------------------------------
                    if (j > 0 && d < prev_dist) {

                        sorted = false;
                        break;
                    }

                    prev_dist = d;
                }

                if (sorted)
                    local_sorted += 1.0;
            }

            #pragma omp atomic
            sorted_rows += local_sorted;
        }

        double score =
            sorted_rows / total_rows_considered;

        std::cout << "\n========================\n";
        std::cout << "Adjacency Distance Ordering Quality\n";
        std::cout << "========================\n";

        std::cout << "Nodes                  : "
                  << N << "\n";

        std::cout << "Graph Degree           : "
                  << K << "\n";

        std::cout << "Checked Prefix Length  : "
                  << order_check_k << "\n";

        std::cout << "Row Range              : "
                  << row_start
                  << " - "
                  << row_end
                  << "\n";

        std::cout << "Rows Considered        : "
                  << total_rows_considered
                  << "\n";

        std::cout << "Rows Sorted            : "
                  << sorted_rows << "\n";

        std::cout << "Ordering Quality Score : "
                  << score << "\n";
    }

    return 0;
}
