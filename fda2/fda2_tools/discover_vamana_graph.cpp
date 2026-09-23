#include "vamana_graph.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>

using namespace fda2;

/*
 * Discover incoming neighbors of a target node by performing
 * forward BFS from the target.
 *
 * RESULT format:
 *
 * RESULT,target,recovered,total,recall,visited,visited_percent,level
 */
static void run_target(
    const Graph& g,
    IdxT target,
    IdxT rs,
    IdxT re,
    int max_levels,
    bool silent,
    const std::vector<std::vector<IdxT>>& rev)
{
    /*
     * Ground-truth incoming neighbors of target.
     *
     * If u -> target exists, then u is an incoming neighbor.
     */
    std::unordered_set<IdxT> gt(
        rev[target].begin(),
        rev[target].end());

    std::unordered_set<IdxT> recovered;

    /*
     * BFS state.
     */
    std::vector<int> level(g.n_rows, -1);
    std::vector<char> seen(g.n_rows, 0);

    std::queue<IdxT> q;

    q.push(target);
    seen[target] = 1;
    level[target] = 0;

    size_t visited = 1;

    int completed = -1;
    size_t completedVisited = 0;

    int cur = 0;

    /*
     * BFS traversal.
     */
    while (!q.empty() && cur <= max_levels) {

        size_t frontierSize = q.size();
        size_t recoveredThisLevel = 0;

        for (size_t k = 0; k < frontierSize; ++k) {

            IdxT u = q.front();
            q.pop();

            auto* neighbors = g.neighbors(u);

            uint32_t degree = g.degree(u);

            for (uint32_t j = 0; j < degree; ++j) {

                IdxT v = neighbors[j];

                /*
                 * We found an incoming neighbor u -> target.
                 */
                if (v == target && gt.count(u)) {

                    if (recovered.insert(u).second) {
                        ++recoveredThisLevel;
                    }
                }

                /*
                 * Do not expand beyond max_levels.
                 */
                if (cur == max_levels)
                    continue;

                /*
                 * Restrict traversal to [rs, re].
                 */
                if (v >= rs &&
                    v <= re &&
                    !seen[v]) {

                    seen[v] = 1;
                    level[v] = cur + 1;

                    q.push(v);

                    ++visited;
                }
            }
        }

        double recall =
            gt.empty()
                ? 100.0
                : 100.0 *
                      static_cast<double>(recovered.size()) /
                      static_cast<double>(gt.size());

        if (!silent) {

            std::cout
                << "Level "
                << cur
                << ": frontier="
                << frontierSize
                << " visited="
                << visited
                << " recovered_this_level="
                << recoveredThisLevel
                << " recovered="
                << recovered.size()
                << "/"
                << gt.size()
                << " ("
                << std::fixed
                << std::setprecision(2)
                << recall
                << "%)"
                << "\n";
        }

        /*
         * All incoming neighbors have been recovered.
         */
        if (recovered.size() == gt.size() &&
            completed < 0) {

            completed = cur;
            completedVisited = visited;

            break;
        }

        ++cur;
    }

    /*
     * Final statistics.
     */
    double finalRecall =
        gt.empty()
            ? 100.0
            : 100.0 *
                  static_cast<double>(recovered.size()) /
                  static_cast<double>(gt.size());

    size_t finalVisited =
        (completed < 0)
            ? visited
            : completedVisited;

    double visitedPercent =
        100.0 *
        static_cast<double>(finalVisited) /
        static_cast<double>(g.n_rows);

    /*
     * CSV-style result:
     *
     * RESULT,
     * target,
     * recovered,
     * total,
     * recall,
     * visited,
     * visited_percent,
     * completion_level
     */
    std::cout
        << "RESULT,"
        << target
        << ","
        << recovered.size()
        << ","
        << gt.size()
        << ","
        << std::fixed
        << std::setprecision(2)
        << finalRecall
        << ","
        << finalVisited
        << ","
        << visitedPercent
        << ","
        << completed
        << "\n";
}


int main(int argc, char** argv)
{
    Options o;

    int i = 1;

    bool silent = false;
    bool batch = false;

    int target = -1;
    int levels = 100;

    IdxT rs = 0;
    IdxT re = 0;

    IdxT bs = 0;
    IdxT be = 0;

    /*
     * Parse command-line arguments.
     */
    while (i < argc) {

        std::string a = argv[i];

        if (a == "--format") {

            std::string x = argv[++i];

            if (x == "u8")
                o.vector_type = VectorType::UInt8;
            else
                o.vector_type = VectorType::Float32;

            i++;
        }

        else if (a == "--nodes") {

            o.nodes =
                std::stoull(argv[++i]);

            i++;
        }

        else if (a == "--dim") {

            o.dim =
                std::stoul(argv[++i]);

            i++;
        }

        else if (a == "--degree") {

            o.degree =
                std::stoul(argv[++i]);

            i++;
        }

        else if (a == "--silent") {

            silent = true;

            i++;
        }

        /*
         * Batch mode:
         *
         * --batch START END
         */
        else if (a == "--batch") {

            batch = true;

            bs =
                std::stoul(argv[++i]);

            be =
                std::stoul(argv[++i]);

            /*
             * Move past END.
             */
            i++;
        }

        /*
         * Single target:
         *
         * --target T
         */
        else if (a == "--target") {

            target =
                std::stoi(argv[++i]);

            i++;
        }

        /*
         * Maximum BFS levels:
         *
         * --levels L
         */
        else if (a == "--levels") {

            levels =
                std::stoi(argv[++i]);

            i++;
        }

        /*
         * Scan range:
         *
         * --scan-range START END
         */
        else if (a == "--scan-range") {

            rs =
                std::stoul(argv[++i]);

            re =
                std::stoul(argv[++i]);

            /*
             * Move past END.
             */
            i++;
        }

        /*
         * First unrecognized argument is the graph path.
         */
        else {

            break;
        }
    }

    /*
     * Graph path is required.
     */
    if (i >= argc) {

        std::cerr
            << "Usage: "
            << argv[0]
            << " --format f32|u8"
            << " --nodes N"
            << " --dim D"
            << " --degree R"
            << " [--target T|--batch S E]"
            << " --scan-range S E"
            << " [--levels L]"
            << " [--silent]"
            << " graph.out\n";

        return 1;
    }

    /*
     * Load graph.
     */
    Graph g =
        load_graph(
            argv[i],
            o,
            !silent);

    /*
     * Validate scan range.
     */
    if (rs > re ||
        re >= g.n_rows) {

        std::cerr
            << "Invalid scan range: "
            << rs
            << " "
            << re
            << "\n";

        return 1;
    }

    /*
     * Build reverse graph.
     *
     * rev[v] contains all u such that:
     *
     * u -> v
     */
    auto rev =
        build_reverse_graph(g);

    /*
     * Batch mode.
     */
    if (batch) {

        if (bs > be ||
            be >= g.n_rows) {

            std::cerr
                << "Invalid batch range: "
                << bs
                << " "
                << be
                << "\n";

            return 1;
        }

        for (IdxT t = bs;
             t <= be;
             ++t) {

            run_target(
                g,
                t,
                rs,
                re,
                levels,
                silent,
                rev);
        }
    }

    /*
     * Single-target mode.
     */
    else {

        if (target < 0 ||
            static_cast<uint64_t>(target) >=
                g.n_rows) {

            std::cerr
                << "Invalid target: "
                << target
                << "\n";

            return 1;
        }

        run_target(
            g,
            static_cast<IdxT>(target),
            rs,
            re,
            levels,
            silent,
            rev);
    }

    return 0;
}