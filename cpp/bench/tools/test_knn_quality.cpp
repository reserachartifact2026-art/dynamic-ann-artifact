#define RAFT_SYSTEM_LITTLE_ENDIAN 1
#include <raft/core/serialize.hpp>
#include <raft/core/host_mdarray.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <omp.h>

using IdxT = uint32_t;

// ---------------- Graph ----------------
struct Graph {
    uint32_t n_rows;
    uint32_t dim;
    uint32_t degree;
    std::vector<IdxT> adj;
};

Graph load_graph(const std::string& path)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("Cannot open graph file");

    raft::resources handle;

    Graph G;

    // 1. dtype
    char dtype[4];
    is.read(dtype, 4);

    // 2. version
    int version = raft::deserialize_scalar<int>(handle, is);

    // 3. fields (IMPORTANT: use RAFT!)
    G.n_rows = raft::deserialize_scalar<uint32_t>(handle, is);
    G.dim    = raft::deserialize_scalar<uint32_t>(handle, is);
    G.degree = raft::deserialize_scalar<uint32_t>(handle, is);

    // 4. metric
    int metric = raft::deserialize_scalar<int>(handle, is);

    std::cout << "Loaded graph:\n";
    std::cout << "  n_rows = " << G.n_rows << "\n";
    std::cout << "  dim    = " << G.dim << "\n";
    std::cout << "  degree = " << G.degree << "\n";

    // 5. adjacency (also RAFT!)
    auto md = raft::make_host_matrix<IdxT, int64_t>(G.n_rows, G.degree);
    raft::deserialize_mdspan(handle, is, md.view());

    G.adj.resize(size_t(G.n_rows) * G.degree);
    std::memcpy(G.adj.data(), md.view().data_handle(),
                G.adj.size() * sizeof(IdxT));

    // 6. trailing field (DON’T skip)
    uint32_t content_map = raft::deserialize_scalar<uint32_t>(handle, is);

    return G;
}


// ---------------- FBIN ----------------
std::vector<float> load_fbin(
    const std::string& path,
    uint32_t& n,
    uint32_t& dim)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("Cannot open fbin");

    is.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    is.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

    std::vector<float> data(size_t(n) * dim);
    is.read(reinterpret_cast<char*>(data.data()),
            data.size() * sizeof(float));

    return data;
}

// ---------------- L2 ----------------
inline float l2(const float* a, const float* b, uint32_t dim)
{
    float s = 0;
    for (uint32_t i = 0; i < dim; i++) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

// ---------------- Main ----------------
int main(int argc, char** argv)
{
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " graph.bin vectors.fbin\n";
        return 1;
    }

    Graph G = load_graph(argv[1]);

    uint32_t n_vecs, dim;
    auto vecs = load_fbin(argv[2], n_vecs, dim);

    if (dim != G.dim || n_vecs < G.n_rows) {
        std::cerr << "Dimension mismatch\n";
        return 1;
    }

    int N = G.n_rows;
    int K = G.degree;

    double total_recall = 0.0;

    #pragma omp parallel
    {
        std::vector<std::pair<float,int>> dist;
        dist.reserve(N);

        double local_sum = 0.0;

        #pragma omp for schedule(dynamic, 4)
        for (int i = 0; i < N; i++) {

            dist.clear();
            const float* vi = &vecs[size_t(i) * dim];

            // brute force
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                const float* vj = &vecs[size_t(j) * dim];
                dist.emplace_back(l2(vi, vj, dim), j);
            }

            // top-K
            std::nth_element(dist.begin(),
                             dist.begin() + K,
                             dist.end());

            // build GT set
            std::unordered_set<int> gt;
            for (int k = 0; k < K; k++)
                gt.insert(dist[k].second);

            // compare
            int correct = 0;
            const IdxT* nbrs = &G.adj[i * K];

            for (int j = 0; j < K; j++) {
                if (gt.count(nbrs[j])) correct++;
            }

            local_sum += double(correct) / K;
        }

        #pragma omp atomic
        total_recall += local_sum;
    }

    double avg = total_recall / N;

    std::cout << "\n========================\n";
    std::cout << "Recall@K\n";
    std::cout << "========================\n";
    std::cout << "Nodes      : " << N << "\n";
    std::cout << "K          : " << K << "\n";
    std::cout << "Avg Recall : " << avg << "\n";

    return 0;
}
