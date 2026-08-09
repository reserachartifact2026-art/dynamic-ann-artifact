#define RAFT_SYSTEM_LITTLE_ENDIAN 1

#include <raft/core/host_mdarray.hpp>
#include <raft/core/serialize.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <omp.h>

using IdxT = uint32_t;

struct Graph {
    uint32_t n_rows;
    uint32_t dim;
    uint32_t degree;
    std::vector<IdxT> adj;
};

struct ReverseGraph {
    std::vector<std::vector<IdxT>> incoming;
};

struct ClusterResult {
    std::vector<uint32_t> assignment;
    std::vector<std::vector<IdxT>> members;
};

struct NodeStats {
    uint32_t cluster = 0;
    size_t gt = 0;
    size_t recovered = 0;
    double recall = 100.0;
};

struct ClusterStats {
    size_t nodes = 0;
    size_t gt_total = 0;
    size_t recovered_total = 0;
    size_t full_recall_nodes = 0;
    size_t zero_gt_nodes = 0;
    double recall_sum = 0.0;
};

Graph load_graph(const std::string& path)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) { throw std::runtime_error("Cannot open graph file"); }

    raft::resources handle;
    Graph G;

    char dtype[4];
    is.read(dtype, 4);

    int version = raft::deserialize_scalar<int>(handle, is);
    (void)version;

    G.n_rows = raft::deserialize_scalar<uint32_t>(handle, is);
    G.dim    = raft::deserialize_scalar<uint32_t>(handle, is);
    G.degree = raft::deserialize_scalar<uint32_t>(handle, is);

    int metric = raft::deserialize_scalar<int>(handle, is);
    (void)metric;

    auto md = raft::make_host_matrix<IdxT, int64_t>(G.n_rows, G.degree);
    raft::deserialize_mdspan(handle, is, md.view());

    G.adj.resize(size_t(G.n_rows) * G.degree);
    std::memcpy(G.adj.data(), md.view().data_handle(), G.adj.size() * sizeof(IdxT));

    uint32_t content_map = raft::deserialize_scalar<uint32_t>(handle, is);
    (void)content_map;

    return G;
}

std::vector<float> load_fbin(const std::string& path, uint32_t& n, uint32_t& dim)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) { throw std::runtime_error("Cannot open fbin file"); }

    is.read(reinterpret_cast<char*>(&n), sizeof(uint32_t));
    is.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

    std::vector<float> data(size_t(n) * dim);
    is.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));

    return data;
}

inline float l2(const float* a, const float* b, uint32_t dim)
{
    float s = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        s += d * d;
    }
    return s;
}

ReverseGraph build_reverse_graph(const Graph& G)
{
    ReverseGraph RG;
    RG.incoming.resize(G.n_rows);

    std::vector<size_t> counts(G.n_rows, 0);
    for (IdxT u = 0; u < G.n_rows; ++u) {
        const IdxT* nbrs = &G.adj[size_t(u) * G.degree];
        for (uint32_t j = 0; j < G.degree; ++j) {
            IdxT v = nbrs[j];
            if (v < G.n_rows) { counts[v]++; }
        }
    }

    for (IdxT v = 0; v < G.n_rows; ++v) {
        RG.incoming[v].reserve(counts[v]);
    }

    for (IdxT u = 0; u < G.n_rows; ++u) {
        const IdxT* nbrs = &G.adj[size_t(u) * G.degree];
        for (uint32_t j = 0; j < G.degree; ++j) {
            IdxT v = nbrs[j];
            if (v < G.n_rows) { RG.incoming[v].push_back(u); }
        }
    }

    return RG;
}

ClusterResult kmeans_cluster(
    const std::vector<float>& vecs,
    uint32_t n,
    uint32_t dim,
    uint32_t k,
    int iterations)
{
    ClusterResult result;
    result.assignment.assign(n, 0);
    result.members.assign(k, {});

    std::vector<float> centroids(size_t(k) * dim, 0.0f);
    std::vector<float> next_centroids(size_t(k) * dim, 0.0f);
    std::vector<size_t> counts(k, 0);

    for (uint32_t c = 0; c < k; ++c) {
        uint32_t src = static_cast<uint32_t>(
            std::min<uint64_t>(n - 1, uint64_t(c) * n / k));
        std::copy_n(&vecs[size_t(src) * dim], dim, &centroids[size_t(c) * dim]);
    }

    for (int iter = 0; iter < iterations; ++iter) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(n); ++i) {
            const float* x = &vecs[size_t(i) * dim];
            uint32_t best = 0;
            float best_dist = std::numeric_limits<float>::max();

            for (uint32_t c = 0; c < k; ++c) {
                float d = l2(x, &centroids[size_t(c) * dim], dim);
                if (d < best_dist) {
                    best_dist = d;
                    best = c;
                }
            }

            result.assignment[i] = best;
        }

        std::fill(next_centroids.begin(), next_centroids.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (uint32_t i = 0; i < n; ++i) {
            uint32_t c = result.assignment[i];
            counts[c]++;

            const float* x = &vecs[size_t(i) * dim];
            float* dst = &next_centroids[size_t(c) * dim];

            for (uint32_t d = 0; d < dim; ++d) {
                dst[d] += x[d];
            }
        }

        for (uint32_t c = 0; c < k; ++c) {
            if (counts[c] == 0) {
                uint32_t src = static_cast<uint32_t>(
                    std::min<uint64_t>(n - 1, uint64_t(c) * n / k));
                std::copy_n(&vecs[size_t(src) * dim], dim, &centroids[size_t(c) * dim]);
                continue;
            }

            float inv = 1.0f / static_cast<float>(counts[c]);
            float* dst = &centroids[size_t(c) * dim];
            const float* src = &next_centroids[size_t(c) * dim];

            for (uint32_t d = 0; d < dim; ++d) {
                dst[d] = src[d] * inv;
            }
        }
    }

    for (IdxT i = 0; i < n; ++i) {
        result.members[result.assignment[i]].push_back(i);
    }

    return result;
}

bool adjacency_contains(const Graph& G, IdxT u, IdxT target)
{
    const IdxT* nbrs = &G.adj[size_t(u) * G.degree];
    for (uint32_t j = 0; j < G.degree; ++j) {
        if (nbrs[j] == target) { return true; }
    }
    return false;
}

void print_usage(const char* prog)
{
    std::cerr
        << "Usage:\n"
        << "  " << prog << " --clusters <k> [--iters <n>] [--row-range <start> <end>]\n"
        << "      [--print-missing <limit>] graph.bin vectors.fbin\n\n"
        << "Notes:\n"
        << "  For each target node, this scans the target's own cluster for incoming edges\n"
        << "  and compares that recovered set against the full reverse-graph ground truth.\n";
}

int main(int argc, char** argv)
{
    uint32_t num_clusters = 0;
    int iterations = 10;
    int print_missing_limit = 0;
    bool use_row_range = false;
    IdxT row_start = 0;
    IdxT row_end = 0;

    int argi = 1;
    while (argi < argc) {
        std::string arg = argv[argi];

        if (arg == "--clusters") {
            if (argi + 1 >= argc) {
                std::cerr << "--clusters requires a value\n";
                return 1;
            }
            num_clusters = static_cast<uint32_t>(std::stoul(argv[argi + 1]));
            argi += 2;
        } else if (arg == "--iters") {
            if (argi + 1 >= argc) {
                std::cerr << "--iters requires a value\n";
                return 1;
            }
            iterations = std::stoi(argv[argi + 1]);
            argi += 2;
        } else if (arg == "--row-range") {
            if (argi + 2 >= argc) {
                std::cerr << "--row-range requires start and end\n";
                return 1;
            }
            use_row_range = true;
            row_start = static_cast<IdxT>(std::stoul(argv[argi + 1]));
            row_end = static_cast<IdxT>(std::stoul(argv[argi + 2]));
            argi += 3;
        } else if (arg == "--print-missing") {
            if (argi + 1 >= argc) {
                std::cerr << "--print-missing requires a value\n";
                return 1;
            }
            print_missing_limit = std::stoi(argv[argi + 1]);
            argi += 2;
        } else {
            break;
        }
    }

    if (argc - argi < 2 || num_clusters == 0 || iterations <= 0) {
        print_usage(argv[0]);
        return 1;
    }

    Graph G = load_graph(argv[argi]);

    uint32_t n_vecs = 0;
    uint32_t dim = 0;
    auto vecs = load_fbin(argv[argi + 1], n_vecs, dim);

    if (n_vecs < G.n_rows || dim != G.dim) {
        std::cerr << "Vector file does not match graph dimensions\n";
        return 1;
    }

    if (num_clusters > G.n_rows) {
        std::cerr << "--clusters must be <= number of graph nodes\n";
        return 1;
    }

    if (!use_row_range) {
        row_start = 0;
        row_end = G.n_rows - 1;
    }

    if (row_start > row_end || row_end >= G.n_rows) {
        std::cerr << "Invalid row range\n";
        return 1;
    }

    std::cout << "Loaded graph\n";
    std::cout << "  Nodes      : " << G.n_rows << "\n";
    std::cout << "  Degree     : " << G.degree << "\n";
    std::cout << "  Dimension  : " << G.dim << "\n\n";

    std::cout << "Building reverse graph...\n";
    ReverseGraph RG = build_reverse_graph(G);

    std::cout << "Clustering dataset...\n";
    ClusterResult clusters = kmeans_cluster(vecs, G.n_rows, G.dim, num_clusters, iterations);

    std::vector<NodeStats> node_stats(G.n_rows);
    std::vector<ClusterStats> cluster_stats(num_clusters);

    std::cout << "Scanning cluster-local incoming edges...\n";

    #pragma omp parallel
    {
        std::vector<ClusterStats> local_cluster_stats(num_clusters);

        #pragma omp for schedule(dynamic, 128)
        for (int target_i = static_cast<int>(row_start);
             target_i <= static_cast<int>(row_end);
             ++target_i)
        {
            IdxT target = static_cast<IdxT>(target_i);
            uint32_t c = clusters.assignment[target];
            const auto& members = clusters.members[c];
            const auto& gt = RG.incoming[target];

            size_t recovered = 0;
            for (IdxT candidate : members) {
                if (adjacency_contains(G, candidate, target)) {
                    recovered++;
                }
            }

            double recall = gt.empty()
                ? 100.0
                : 100.0 * static_cast<double>(recovered) / static_cast<double>(gt.size());

            node_stats[target].cluster = c;
            node_stats[target].gt = gt.size();
            node_stats[target].recovered = recovered;
            node_stats[target].recall = recall;

            ClusterStats& cs = local_cluster_stats[c];
            cs.nodes++;
            cs.gt_total += gt.size();
            cs.recovered_total += recovered;
            cs.recall_sum += recall;
            if (gt.empty()) { cs.zero_gt_nodes++; }
            if (recovered == gt.size()) { cs.full_recall_nodes++; }
        }

        #pragma omp critical
        {
            for (uint32_t c = 0; c < num_clusters; ++c) {
                cluster_stats[c].nodes += local_cluster_stats[c].nodes;
                cluster_stats[c].gt_total += local_cluster_stats[c].gt_total;
                cluster_stats[c].recovered_total += local_cluster_stats[c].recovered_total;
                cluster_stats[c].full_recall_nodes += local_cluster_stats[c].full_recall_nodes;
                cluster_stats[c].zero_gt_nodes += local_cluster_stats[c].zero_gt_nodes;
                cluster_stats[c].recall_sum += local_cluster_stats[c].recall_sum;
            }
        }
    }

    size_t total_nodes = 0;
    size_t total_gt = 0;
    size_t total_recovered = 0;
    size_t total_full = 0;
    size_t total_zero_gt = 0;
    double total_recall_sum = 0.0;

    std::cout << "\n========================\n";
    std::cout << "Cluster Incoming Edge Quality\n";
    std::cout << "========================\n";
    std::cout << "Clusters             : " << num_clusters << "\n";
    std::cout << "K-means Iterations   : " << iterations << "\n";
    std::cout << "Row Range            : " << row_start << " - " << row_end << "\n";
    std::cout << "Rows Considered      : " << (row_end - row_start + 1) << "\n\n";

    std::cout
        << std::left
        << std::setw(10) << "Cluster"
        << std::setw(12) << "Members"
        << std::setw(12) << "Checked"
        << std::setw(14) << "GT_IN"
        << std::setw(14) << "Found_IN"
        << std::setw(14) << "AvgRecall"
        << std::setw(14) << "EdgeRecall"
        << std::setw(12) << "Full"
        << "\n";

    for (uint32_t c = 0; c < num_clusters; ++c) {
        const ClusterStats& cs = cluster_stats[c];
        double avg_recall = cs.nodes == 0 ? 0.0 : cs.recall_sum / cs.nodes;
        double edge_recall = cs.gt_total == 0
            ? 100.0
            : 100.0 * static_cast<double>(cs.recovered_total) / static_cast<double>(cs.gt_total);

        std::cout
            << std::left
            << std::setw(10) << c
            << std::setw(12) << clusters.members[c].size()
            << std::setw(12) << cs.nodes
            << std::setw(14) << cs.gt_total
            << std::setw(14) << cs.recovered_total
            << std::setw(14) << std::fixed << std::setprecision(2) << avg_recall
            << std::setw(14) << std::fixed << std::setprecision(2) << edge_recall
            << std::setw(12) << cs.full_recall_nodes
            << "\n";

        total_nodes += cs.nodes;
        total_gt += cs.gt_total;
        total_recovered += cs.recovered_total;
        total_full += cs.full_recall_nodes;
        total_zero_gt += cs.zero_gt_nodes;
        total_recall_sum += cs.recall_sum;
    }

    double overall_avg_recall = total_nodes == 0 ? 0.0 : total_recall_sum / total_nodes;
    double overall_edge_recall = total_gt == 0
        ? 100.0
        : 100.0 * static_cast<double>(total_recovered) / static_cast<double>(total_gt);

    std::cout << "\n========================\n";
    std::cout << "Final Summary\n";
    std::cout << "========================\n";
    std::cout << "Targets Checked       : " << total_nodes << "\n";
    std::cout << "Total GT Incoming     : " << total_gt << "\n";
    std::cout << "Recovered In-Cluster  : " << total_recovered << "\n";
    std::cout << "Average Node Recall   : " << std::fixed << std::setprecision(2)
              << overall_avg_recall << "%\n";
    std::cout << "Incoming Edge Recall  : " << std::fixed << std::setprecision(2)
              << overall_edge_recall << "%\n";
    std::cout << "Full Recall Nodes     : " << total_full << " ("
              << (total_nodes == 0 ? 0.0 : 100.0 * total_full / total_nodes) << "%)\n";
    std::cout << "Zero-GT Nodes         : " << total_zero_gt << "\n";

    if (print_missing_limit > 0) {
        int printed = 0;
        std::cout << "\nMissing Incoming Examples\n";
        std::cout << "-------------------------\n";

        for (IdxT target = row_start; target <= row_end && printed < print_missing_limit; ++target) {
            const NodeStats& ns = node_stats[target];
            if (ns.recovered == ns.gt) { continue; }

            std::unordered_set<IdxT> same_cluster_members(
                clusters.members[ns.cluster].begin(),
                clusters.members[ns.cluster].end());

            std::cout << "Target " << target
                      << " cluster=" << ns.cluster
                      << " recovered=" << ns.recovered
                      << "/" << ns.gt
                      << " missing:";

            int missing_for_node = 0;
            for (IdxT incoming : RG.incoming[target]) {
                if (same_cluster_members.count(incoming) == 0) {
                    std::cout << " " << incoming;
                    missing_for_node++;
                }
            }
            if (missing_for_node == 0) {
                std::cout << " none-outside-cluster";
            }
            std::cout << "\n";
            printed++;
        }
    }

    return 0;
}
