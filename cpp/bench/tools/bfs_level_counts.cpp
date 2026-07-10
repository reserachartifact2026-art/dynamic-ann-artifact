// bfs_level_counts.cpp

#define RAFT_SYSTEM_LITTLE_ENDIAN 1

#include <raft/core/host_mdarray.hpp>
#include <raft/core/serialize.hpp>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>

using IdxT = uint32_t;

struct Graph
{
    uint32_t n_rows;
    uint32_t dim;
    uint32_t degree;

    std::vector<IdxT> adj;
};

Graph load_graph(const std::string& path)
{
    std::ifstream is(path, std::ios::binary);

    if (!is) {
        throw std::runtime_error("Cannot open graph file");
    }

    raft::resources handle;
    Graph G;

    char dtype[4];
    is.read(dtype, 4);

    int version =
        raft::deserialize_scalar<int>(handle, is);

    G.n_rows =
        raft::deserialize_scalar<uint32_t>(handle, is);

    G.dim =
        raft::deserialize_scalar<uint32_t>(handle, is);

    G.degree =
        raft::deserialize_scalar<uint32_t>(handle, is);

    int metric =
        raft::deserialize_scalar<int>(handle, is);

    auto md =
        raft::make_host_matrix<IdxT, int64_t>(
            G.n_rows,
            G.degree);

    raft::deserialize_mdspan(
        handle,
        is,
        md.view());

    G.adj.resize(
        size_t(G.n_rows) * G.degree);

    std::memcpy(
        G.adj.data(),
        md.view().data_handle(),
        sizeof(IdxT) * G.adj.size());

    uint32_t content_map =
        raft::deserialize_scalar<uint32_t>(handle, is);

    (void)dtype;
    (void)version;
    (void)metric;
    (void)content_map;

    return G;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr
            << "Usage:\n\n"
            << argv[0]
            << " <graph> <start_node>\n";

        return 1;
    }

    std::string graph_file = argv[1];
    IdxT start = static_cast<IdxT>(std::strtoul(argv[2], nullptr, 10));

    Graph G = load_graph(graph_file);

    if (start >= G.n_rows) {
        std::cerr << "Invalid start node.\n";
        return 1;
    }

    std::vector<uint8_t> visited(G.n_rows, 0);
    std::queue<IdxT> q;

    visited[start] = 1;
    q.push(start);

    size_t cumulative_visited = 1;
    int level = 0;

    std::cout << "level,nodes_visited,cumulative_visited\n";

    while (!q.empty()) {
        size_t frontier_size = q.size();
        size_t discovered_next = 0;

        std::cout
            << level << ","
            << frontier_size << ","
            << cumulative_visited << "\n";

        for (size_t i = 0; i < frontier_size; ++i) {
            IdxT u = q.front();
            q.pop();

            const IdxT* nbrs =
                &G.adj[size_t(u) * G.degree];

            for (uint32_t j = 0; j < G.degree; ++j) {
                IdxT v = nbrs[j];

                if (v >= G.n_rows) {
                    continue;
                }

                if (!visited[v]) {
                    visited[v] = 1;
                    q.push(v);
                    discovered_next++;
                }
            }
        }

        cumulative_visited += discovered_next;
        level++;
    }

    if (cumulative_visited != G.n_rows) {
        std::cerr
            << "WARNING: BFS reached "
            << cumulative_visited
            << " / "
            << G.n_rows
            << " nodes from start node "
            << start
            << ".\n";
    }

    return 0;
}
