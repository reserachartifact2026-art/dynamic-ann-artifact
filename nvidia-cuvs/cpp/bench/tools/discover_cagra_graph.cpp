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

struct ReverseGraph
{
    std::vector<std::vector<IdxT>> incoming;
};


//==============================================================
// Load serialized CAGRA graph
//==============================================================

Graph load_graph(const std::string& path, bool verbose)
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

        if (verbose) {
                std::cout << "Loaded graph\n";
                std::cout << "Nodes      : " << G.n_rows << "\n";
                std::cout << "Degree     : " << G.degree << "\n";
                std::cout << "Dimension  : " << G.dim << "\n";
                std::cout << "Metric     : " << metric << "\n\n";
        }

        return G;
}

ReverseGraph build_reverse_graph(
    const Graph& G,
    IdxT start,
    IdxT end,
    bool verbose)
{
    ReverseGraph RG;

    RG.incoming.resize(G.n_rows);

    if (verbose) {
        std::cout
            << "Building reverse graph...\n";
    }

    for (IdxT u = start; u <= end; ++u) {

        const IdxT* nbrs =
            &G.adj[size_t(u) * G.degree];

        for (uint32_t j = 0;
             j < G.degree;
             ++j)
        {
            IdxT v = nbrs[j];

            if (v >= start &&
                v <= end)
            {
                RG.incoming[v].push_back(u);
            }
        }

        if (verbose && (u % 100000) == 0) {
            std::cout
                << "Processed "
                << u
                << " / "
                << end
                << "\r"
                << std::flush;
        }
    }

    if (verbose) {
        std::cout
            << "\nReverse graph built.\n";
    }

    return RG;
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
        bool auto_mode = false;
        bool silent = false;

bool batch_mode = false;
IdxT batch_start = 0;
IdxT batch_end = 0;

        constexpr int AUTO_MAX_LEVEL = 100;
        int max_levels = AUTO_MAX_LEVEL;
        if (argc < 5 || argc > 7) {

                std::cerr
                        << "Usage:\n\n"
                        << argv[0]
                        << " <graph> <target> <scan_start> <scan_end> [--silent]\n"
                        << "or\n"
                        << argv[0]
                        << " <graph> <target> <levels> <scan_start> <scan_end> [--silent]\n"
                        << "or\n"
                        << argv[0]
                        << " <graph> --batch <scan_start> <scan_end> [--silent]\n";

                return 1;
        }


        //----------------------------------------------------------
        // Parse arguments
        //----------------------------------------------------------

        std::string graph_file =
                argv[1];

int target = -1;

        IdxT gt_start;
        IdxT gt_end;
if (std::string(argv[2]) == "--batch") {

    batch_mode = true;

    batch_start = std::atoi(argv[3]);
    batch_end   = std::atoi(argv[4]);

} else {

    target = std::atoi(argv[2]);
}




        //----------------------------------------
        // Detect silent flag
        //----------------------------------------

        int last_arg = argc;

        if (std::string(argv[argc - 1]) == "--silent") {
                silent = true;
                last_arg--;
        }

        //----------------------------------------
        // Auto mode
        //----------------------------------------

        if (batch_mode) {

                auto_mode = true;
                max_levels = AUTO_MAX_LEVEL;

                gt_start = batch_start;
                gt_end   = batch_end;
        }

        else if (last_arg == 5) {

                auto_mode = true;
                max_levels = AUTO_MAX_LEVEL;

                gt_start = std::atoi(argv[3]);
                gt_end   = std::atoi(argv[4]);
        }

        //----------------------------------------
        // Explicit level
        //----------------------------------------

        else {

                auto_mode = false;

                max_levels = std::atoi(argv[3]);

                gt_start = std::atoi(argv[4]);
                gt_end   = std::atoi(argv[5]);
        }

        bool verbose = !silent && !batch_mode;

        Graph G = load_graph(graph_file, verbose);

ReverseGraph RG =
    build_reverse_graph(
        G,
        gt_start,
        gt_end,
        verbose);

        if (!batch_mode) {
                if (target < 0 || target >= static_cast<int>(G.n_rows)) {
                        std::cerr << "Invalid target node.\n";
                        return 1;
                }
        }

        if (gt_start > gt_end || gt_end >= G.n_rows) {
                std::cerr << "Invalid scan range.\n";
                return 1;
        }

        if (max_levels < 0) {
                std::cerr << "Invalid max level.\n";
                return 1;
        }

IdxT first_target =
    batch_mode ? batch_start : target;

IdxT last_target =
    batch_mode ? batch_end : target;

for (target = first_target;
     target <= last_target;
     ++target)
{

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
std::unordered_set<IdxT> outside_range_nodes;

int completed_level=-1;

size_t completed_visited=0;

size_t total_visited=1;

int current_level=0;




        visited[target] = true;

        info[target].level = 0;

        q.push(target);

        if (batch_mode &&
            (target % 1000) == 0)
        {
                std::cerr
                        << "Processed "
                        << target
                        << "\r"
                        << std::flush;
        }

        if (verbose) {
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
        }

        //==========================================================
        // PART 2 STARTS HERE
        //==========================================================


        //----------------------------------------------------------
        // Phase 1:
        // Compute groundtruth incoming neighbours
        //----------------------------------------------------------

        if (verbose) {
                std::cout
                        << "\nScanning graph for groundtruth incoming neighbours...\n";
        }

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

for (IdxT u : RG.incoming[target]) {
    gt_incoming.insert(u);
}

        if (verbose) {
                std::cout
                        << "Groundtruth incoming neighbours : "
                        << gt_incoming.size()
                        << "\n\n";
        }

        if (verbose && !gt_incoming.empty()) {

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

        while (!q.empty() &&
                        current_level <= max_levels)
        {
                size_t frontier_size = q.size();

                size_t newly_found = 0;

                std::vector<IdxT> found_this_level;

                if (verbose) {
                        print_separator();

                        std::cout
                                << "Level "
                                << current_level
                                << "\n";

                        print_separator();
                }

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

                                if (v < gt_start || v > gt_end) {
                                        outside_range_nodes.insert(v);
                                        continue;
                                }

                                if (!visited[v]) {

                                        visited[v] = true;

                                        info[v].level = current_level + 1;

                                        q.push(v);

                                        total_visited++;
                                }

                        } // end for
                }

                for (auto node : gt_incoming)
                {
                        if (visited[node] && recovered.count(node)==0)
                        {
                                if (verbose) {
                                        std::cout
                                                << "BUG: "
                                                << node
                                                << " visited but not recovered\n";
                                }
                        }
                }

                //------------------------------------------------------
                // Print statistics
                //------------------------------------------------------

                double recall = 0.0;

                if (!gt_incoming.empty()) {

                        recall =
                                100.0 *
                                recovered.size() /
                                gt_incoming.size();
                }

                if (verbose) {
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

                        std::cout
                                << " ("
                                << std::fixed
                                << std::setprecision(2)
                                << recall
                                << "%)\n";
                }

                if (completed_level == -1 &&
                                recovered.size() == gt_incoming.size())
                {
                        completed_level = current_level;
                        completed_visited = total_visited;

                        if (auto_mode)
                                break;
                }

                if (verbose && !found_this_level.empty()) {

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

                if (verbose) {
                        std::cout << "\n";
                }

                current_level++;
        }

        //==========================================================
        // PART 3 STARTS HERE
        //==========================================================
        //----------------------------------------------------------
        // Final Summary
        //----------------------------------------------------------

double final_recall = gt_incoming.empty()
        ? 100.0
        : 100.0 *
        recovered.size() /
        gt_incoming.size();

if (verbose) {

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

        std::cout
                << "WARNING: Encountered "
                << outside_range_nodes.size()
                << " nodes outside scan range.\n";

        if (!outside_range_nodes.empty()) {

                std::cout
                        << "Nodes:\n";

                std::vector<IdxT> nodes(
                                outside_range_nodes.begin(),
                                outside_range_nodes.end());

                std::sort(nodes.begin(), nodes.end());

                for (auto n : nodes)
                        std::cout << n << " ";

                std::cout << "\n";
        }
        print_separator();
}

size_t result_visited =
    (completed_level == -1)
        ? total_visited
        : completed_visited;

int result_level =
    completed_level;
std::cout
    << "RESULT,"
    << target << ","
    << recovered.size() << ","
    << gt_incoming.size() << ","
    << final_recall << ","
    << result_visited << ","
    << result_level
    << "\n";
}
// RESULT,
// target,
// recovered,
// groundtruth,
// recall,
// visited_at_end
// level_where_completed_or_minus1
        return 0;
}
