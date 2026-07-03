// discover_cagra_graph.cpp

#define RAFT_SYSTEM_LITTLE_ENDIAN 1

#include <raft/core/host_mdarray.hpp>
#include <raft/core/serialize.hpp>

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cstring>

using IdxT = uint32_t;

//==============================================================
// Graph
//==============================================================

struct Graph
{
    uint32_t n_rows;
    uint32_t dim;
    uint32_t degree;

    std::vector<IdxT> adj;
};

//==============================================================
// Load serialized CAGRA graph
//==============================================================

Graph load_graph(const std::string& path)
{
    std::ifstream is(path, std::ios::binary);

    if (!is) {
        throw std::runtime_error("Cannot open graph file");
    }

    raft::resources handle;

    Graph G;

    //---------------------------------------
    // dtype
    //---------------------------------------

    char dtype[4];
    is.read(dtype, 4);

    //---------------------------------------
    // version
    //---------------------------------------

    int version =
        raft::deserialize_scalar<int>(handle, is);

    //---------------------------------------
    // graph properties
    //---------------------------------------

    G.n_rows =
        raft::deserialize_scalar<uint32_t>(handle, is);

    G.dim =
        raft::deserialize_scalar<uint32_t>(handle, is);

    G.degree =
        raft::deserialize_scalar<uint32_t>(handle, is);

    int metric =
        raft::deserialize_scalar<int>(handle, is);

    //---------------------------------------
    // adjacency matrix
    //---------------------------------------

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

    //---------------------------------------
    // trailing content map
    //---------------------------------------

    uint32_t content_map =
        raft::deserialize_scalar<uint32_t>(handle, is);

    std::cout << "Loaded graph\n";
    std::cout << "Nodes      : " << G.n_rows << "\n";
    std::cout << "Degree     : " << G.degree << "\n";
    std::cout << "Dimension  : " << G.dim << "\n";
    std::cout << "Metric     : " << metric << "\n\n";

    return G;
}

//==============================================================
// BFS node info
//==============================================================

struct NodeInfo
{
    int level = -1;
};

//==============================================================
// Helper printing
//==============================================================

void print_separator()
{
    std::cout
        << "========================================================\n";
}

//==============================================================
// Main
//==============================================================

int main(int argc, char** argv)
{
    if (argc != 4) {

        std::cerr
            << "Usage:\n\n"
            << argv[0]
            << " <graph.bin> <target_node> <levels>\n";

        return 1;
    }

    //----------------------------------------------------------
    // Parse arguments
    //----------------------------------------------------------

    std::string graph_file =
        argv[1];

    int target =
        std::atoi(argv[2]);

    int max_levels =
        std::atoi(argv[3]);

    //----------------------------------------------------------

    Graph G =
        load_graph(graph_file);

    if (target < 0 ||
        target >= (int)G.n_rows)
    {
        std::cerr
            << "Invalid target node.\n";

        return 1;
    }

    if (max_levels < 0)
    {
        std::cerr
            << "Invalid number of levels.\n";

        return 1;
    }

    //----------------------------------------------------------
    // Groundtruth incoming neighbours
    //----------------------------------------------------------

    std::unordered_set<IdxT> gt_incoming;

    //----------------------------------------------------------
    // Recovered incoming neighbours
    //----------------------------------------------------------

    std::unordered_set<IdxT> recovered;

    //----------------------------------------------------------
    // Discovery level
    //----------------------------------------------------------

    std::vector<NodeInfo> info(
        G.n_rows);

    //----------------------------------------------------------
    // BFS
    //----------------------------------------------------------

    std::queue<IdxT> q;

    std::vector<bool> visited(
        G.n_rows,
        false);

    visited[target] = true;

    info[target].level = 0;

    q.push(target);

    print_separator();

    std::cout
        << "Target Node      : "
        << target
        << "\n";

    std::cout
        << "Discovery Levels : "
        << max_levels
        << "\n";

    print_separator();

    //==========================================================
    // PART 2 STARTS HERE
    //==========================================================


    //----------------------------------------------------------
    // Phase 1:
    // Compute groundtruth incoming neighbours
    //----------------------------------------------------------

    std::cout
        << "\nScanning graph for groundtruth incoming neighbours...\n";

/*    for (IdxT u = 0; u < G.n_rows; ++u) {

        const IdxT* nbrs =
            &G.adj[size_t(u) * G.degree];

        for (uint32_t j = 0; j < G.degree; ++j) {

            if (nbrs[j] == static_cast<IdxT>(target)) {
                gt_incoming.insert(u);
                break;
            }
        }
    }
*/

IdxT gt_start = 0;
IdxT gt_end   = G.n_rows - 1;

if (argc == 6) {
    gt_start = std::atoi(argv[4]);
    gt_end   = std::atoi(argv[5]);
}

for (IdxT u = gt_start; u <= gt_end; ++u) {

    const IdxT* nbrs = &G.adj[size_t(u) * G.degree];

    for (uint32_t j = 0; j < G.degree; ++j) {

        if (nbrs[j] == target) {
            gt_incoming.insert(u);
            break;
        }
    }
}

    std::cout
        << "Groundtruth incoming neighbours : "
        << gt_incoming.size()
        << "\n\n";

    if (!gt_incoming.empty()) {

        std::cout
            << "Groundtruth nodes\n"
            << "-----------------\n";

        int cnt = 0;

        for (auto v : gt_incoming) {

            std::cout << v << " ";

            cnt++;

            if (cnt % 10 == 0)
                std::cout << "\n";
        }

        std::cout << "\n\n";
    }

    //----------------------------------------------------------
    // BFS
    //----------------------------------------------------------

    size_t total_visited = 1;

    int current_level = 0;

    while (!q.empty() &&
           current_level <= max_levels)
    {
        size_t frontier_size = q.size();

        size_t newly_found = 0;

        std::vector<IdxT> found_this_level;

        print_separator();

        std::cout
            << "Level "
            << current_level
            << "\n";

        print_separator();

        //------------------------------------------------------
        // Process one BFS level
        //------------------------------------------------------

        for (size_t f = 0;
             f < frontier_size;
             ++f)
        {
            IdxT u = q.front();
            q.pop();

            const IdxT* nbrs =
                &G.adj[size_t(u) * G.degree];

            //--------------------------------------------------
            // Step 1
            // Examine adjacency list for reverse edge
            //--------------------------------------------------

            for (uint32_t j = 0;
                 j < G.degree;
                 ++j)
            {
                if (nbrs[j] ==
                    static_cast<IdxT>(target))
                {
                    auto res =
                        recovered.insert(u);

                    if (res.second) {

                        found_this_level.push_back(u);

                        newly_found++;
                    }

                    break;
                }
            }

            //--------------------------------------------------
            // Step 2
            // Expand BFS
            //--------------------------------------------------

            if (current_level == max_levels)
                continue;

            for (uint32_t j = 0;
                 j < G.degree;
                 ++j)
            {
                IdxT v = nbrs[j];

                if (!visited[v]) {

                    visited[v] = true;

                    info[v].level =
                        current_level + 1;

                    q.push(v);

                    total_visited++;
                }
            }
        }

for (auto node : gt_incoming)
{
    if (visited[node] && recovered.count(node)==0)
    {
        std::cout
            << "BUG: "
            << node
            << " visited but not recovered\n";
    }
}

        //------------------------------------------------------
        // Print statistics
        //------------------------------------------------------

        std::cout
            << "Frontier size           : "
            << frontier_size
            << "\n";

        std::cout
            << "Visited so far          : "
            << total_visited
            << "\n";

        std::cout
            << "Recovered this level    : "
            << newly_found
            << "\n";

        std::cout
            << "Recovered total         : "
            << recovered.size()
            << " / "
            << gt_incoming.size();

        double recall = 0.0;

        if (!gt_incoming.empty()) {

            recall =
                100.0 *
                recovered.size() /
                gt_incoming.size();
        }

        std::cout
            << " ("
            << std::fixed
            << std::setprecision(2)
            << recall
            << "%)\n";

        if (!found_this_level.empty()) {

            std::cout
                << "\nRecovered at this level\n"
                << "-----------------------\n";

            for (auto node : found_this_level) {

                std::cout
                    << node
                    << " ";
            }

            std::cout << "\n";
        }

        std::cout << "\n";

        current_level++;
    }

    //==========================================================
    // PART 3 STARTS HERE
    //==========================================================
    //----------------------------------------------------------
    // Final Summary
    //----------------------------------------------------------

    print_separator();

    std::cout
        << "SUMMARY\n";

    print_separator();

    std::cout
        << "Target Node              : "
        << target
        << "\n";

    std::cout
        << "Discovery Levels         : "
        << max_levels
        << "\n";

    std::cout
    << "Groundtruth scan range : "
    << gt_start
    << " - "
    << gt_end
    << "\n";




    std::cout
        << "Groundtruth Incoming     : "
        << gt_incoming.size()
        << "\n";

    std::cout
        << "Recovered Incoming       : "
        << recovered.size()
        << "\n";

    double final_recall = 0.0;

    if (!gt_incoming.empty()) {
        final_recall =
            100.0 *
            recovered.size() /
            gt_incoming.size();
    }

    std::cout
        << "Coverage                 : "
        << std::fixed
        << std::setprecision(2)
        << final_recall
        << "%\n";

    //----------------------------------------------------------
    // Recovered nodes
    //----------------------------------------------------------

    std::cout
        << "\nRecovered Incoming Nodes\n";

    print_separator();

    if (recovered.empty()) {

        std::cout << "None\n";

    } else {

        std::cout
            << std::left
            << std::setw(15) << "Node"
            << std::setw(10) << "Level"
            << "\n";

        print_separator();

        std::vector<IdxT> recovered_vec(
            recovered.begin(),
            recovered.end());

        std::sort(
            recovered_vec.begin(),
            recovered_vec.end());

        for (auto node : recovered_vec) {

            std::cout
                << std::left
                << std::setw(15)
                << node
                << std::setw(10)
                << info[node].level
                << "\n";
        }
    }

    //----------------------------------------------------------
    // Missing nodes
    //----------------------------------------------------------

    std::cout
        << "\nMissing Incoming Nodes\n";

    print_separator();

    size_t missing = 0;

    for (auto node : gt_incoming) {

        if (recovered.find(node) ==
            recovered.end())
        {
            std::cout
                << node
                << "\n";

            missing++;
        }
    }

    if (missing == 0) {

        std::cout
            << "None\n";
    }

    //----------------------------------------------------------
    // Discovery statistics
    //----------------------------------------------------------

    print_separator();

    std::cout
        << "Discovery Statistics\n";

    print_separator();

    std::vector<size_t> level_count(
        max_levels + 1, 0);

    for (const auto& n : info) {

        if (n.level >= 0 &&
            n.level <= max_levels)
        {
            level_count[n.level]++;
        }
    }

    for (int l = 0;
         l <= max_levels;
         ++l)
    {
        std::cout
            << "Level "
            << std::setw(3)
            << l
            << " : "
            << level_count[l]
            << " nodes\n";
    }

    print_separator();

    return 0;
}

