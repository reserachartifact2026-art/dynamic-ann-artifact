#include "vamana_graph.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

using namespace fda2;

struct Range {
    int64_t s = 0;
    int64_t e = -1;
    bool on = false;

    bool contains(IdxT x) const {
        return !on ||
               (x >= static_cast<IdxT>(s) &&
                x <= static_cast<IdxT>(e));
    }
};


/*
 * ============================================================
 * Weakly Connected Components
 * ============================================================
 *
 * The graph is treated as undirected.
 *
 * We intentionally use an iterative DFS so that large graphs
 * do not overflow the process stack.
 */
static std::vector<int> wcc(
    const Graph& g,
    const Range& r,
    int& count,
    std::vector<int>& sizes)
{
    std::vector<int> comp(g.n_rows, -1);
    std::vector<IdxT> st;

    count = 0;
    sizes.clear();

    int64_t s = r.on ? r.s : 0;
    int64_t e = r.on
                    ? r.e
                    : static_cast<int64_t>(g.n_rows) - 1;

    for (int64_t i = s; i <= e; ++i) {

        if (comp[i] >= 0)
            continue;

        int componentId = count++;
        int componentSize = 0;

        st.clear();
        st.push_back(static_cast<IdxT>(i));

        comp[i] = componentId;

        while (!st.empty()) {

            IdxT u = st.back();
            st.pop_back();

            ++componentSize;

            auto* nb = g.neighbors(u);
            uint32_t degree = g.degree(u);

            for (uint32_t j = 0; j < degree; ++j) {

                IdxT v = nb[j];

                if (v >= g.n_rows)
                    continue;

                if (v == u)
                    continue;

                if (!r.contains(v))
                    continue;

                if (comp[v] >= 0)
                    continue;

                comp[v] = componentId;

                st.push_back(v);
            }
        }

        sizes.push_back(componentSize);
    }

    return comp;
}


/*
 * ============================================================
 * Strongly Connected Components
 * ============================================================
 *
 * Iterative Kosaraju algorithm.
 *
 * Pass 1:
 *   DFS on the original graph and generate finishing order.
 *
 * Pass 2:
 *   DFS on the reverse graph in reverse finishing order.
 *
 * This avoids recursive DFS and therefore avoids stack overflow
 * on large Vamana graphs.
 */
class SCC {

    const Graph& g;
    Range r;

    std::vector<int> comp;

    int cnt = 0;

public:

    SCC(const Graph& g, const Range& r)
        : g(g),
          r(r),
          comp(g.n_rows, -1)
    {
    }


    void run()
    {
        int64_t s = r.on ? r.s : 0;

        int64_t e = r.on
                        ? r.e
                        : static_cast<int64_t>(g.n_rows) - 1;


        /*
         * ----------------------------------------------------
         * First pass:
         * Iterative DFS on original graph.
         * ----------------------------------------------------
         */

        std::vector<char> visited(g.n_rows, 0);

        std::vector<IdxT> order;

        order.reserve(
            static_cast<size_t>(e - s + 1));


        /*
         * DFS frame.
         *
         * next_edge tells us which outgoing edge should
         * be examined next when returning to this vertex.
         */
        struct Frame {
            IdxT u;
            uint32_t next_edge;
        };


        std::vector<Frame> stack;


        for (int64_t start = s; start <= e; ++start) {

            IdxT root =
                static_cast<IdxT>(start);

            if (visited[root])
                continue;


            visited[root] = 1;

            stack.clear();

            stack.push_back({
                root,
                0
            });


            while (!stack.empty()) {

                Frame& frame =
                    stack.back();

                IdxT u = frame.u;

                uint32_t degree =
                    g.degree(u);

                bool descended = false;


                /*
                 * Continue scanning outgoing edges.
                 */
                while (frame.next_edge < degree) {

                    IdxT v =
                        g.neighbors(u)[
                            frame.next_edge++];

                    if (v >= g.n_rows)
                        continue;

                    if (!r.contains(v))
                        continue;

                    if (visited[v])
                        continue;


                    visited[v] = 1;

                    stack.push_back({
                        v,
                        0
                    });

                    descended = true;

                    break;
                }


                /*
                 * No more outgoing edges.
                 *
                 * This is the DFS finishing event.
                 */
                if (!descended) {

                    order.push_back(u);

                    stack.pop_back();
                }
            }
        }


        /*
         * ----------------------------------------------------
         * Second pass:
         * DFS on reversed graph.
         * ----------------------------------------------------
         */

        std::cout
            << "Building reverse graph for SCC...\n";

        auto rev =
            build_reverse_graph(g);


        /*
         * Reset component labels.
         */
        std::fill(
            comp.begin(),
            comp.end(),
            -1);

        cnt = 0;


        std::vector<IdxT> st;


        /*
         * Process vertices in reverse finishing order.
         */
        for (auto it = order.rbegin();
             it != order.rend();
             ++it) {

            IdxT root = *it;

            if (comp[root] != -1)
                continue;


            /*
             * Start a new SCC.
             */
            comp[root] = cnt;

            st.clear();

            st.push_back(root);


            while (!st.empty()) {

                IdxT u =
                    st.back();

                st.pop_back();


                /*
                 * Traverse reverse edges.
                 *
                 * rev[u] contains vertices v such that:
                 *
                 * v -> u
                 */
                for (IdxT v : rev[u]) {

                    if (!r.contains(v))
                        continue;

                    if (comp[v] != -1)
                        continue;


                    comp[v] = cnt;

                    st.push_back(v);
                }
            }

            ++cnt;
        }
    }


    int count() const {
        return cnt;
    }


    const std::vector<int>& components() const {
        return comp;
    }
};


/*
 * ============================================================
 * Main
 * ============================================================
 */

int main(int argc, char** argv)
{
    Options o;

    Range r;

    int i = 1;


    /*
     * --------------------------------------------------------
     * Command-line parsing
     * --------------------------------------------------------
     */

    while (i < argc) {

        std::string a = argv[i];


        if (a == "--format") {

            if (i + 1 >= argc) {
                std::cerr
                    << "Missing value for --format\n";
                return 1;
            }

            std::string x =
                argv[++i];

            if (x == "u8")
                o.vector_type =
                    VectorType::UInt8;
            else if (x == "f32")
                o.vector_type =
                    VectorType::Float32;
            else {
                std::cerr
                    << "Invalid format: "
                    << x << "\n";
                return 1;
            }

            ++i;
        }


        else if (a == "--nodes") {

            if (i + 1 >= argc) {
                std::cerr
                    << "Missing value for --nodes\n";
                return 1;
            }

            o.nodes =
                std::stoull(argv[++i]);

            ++i;
        }


        else if (a == "--dim") {

            if (i + 1 >= argc) {
                std::cerr
                    << "Missing value for --dim\n";
                return 1;
            }

            o.dim =
                std::stoul(argv[++i]);

            ++i;
        }


        else if (a == "--degree") {

            if (i + 1 >= argc) {
                std::cerr
                    << "Missing value for --degree\n";
                return 1;
            }

            o.degree =
                std::stoul(argv[++i]);

            ++i;
        }


        else if (a == "--row-range") {

            if (i + 2 >= argc) {
                std::cerr
                    << "Missing values for --row-range\n";
                return 1;
            }

            r.on = true;

            r.s =
                std::stoll(argv[++i]);

            r.e =
                std::stoll(argv[++i]);

            ++i;
        }


        else {

            /*
             * First unknown argument is the graph path.
             */
            break;
        }
    }


    /*
     * --------------------------------------------------------
     * Graph path
     * --------------------------------------------------------
     */

    if (i >= argc) {

        std::cerr
            << "Usage: "
            << argv[0]
            << " --format f32|u8"
            << " --nodes N"
            << " --dim D"
            << " --degree R"
            << " [--row-range S E]"
            << " graph.out\n";

        return 1;
    }


    /*
     * --------------------------------------------------------
     * Load graph
     * --------------------------------------------------------
     */

    Graph g =
        load_graph(
            argv[i],
            o);


    /*
     * Validate row range.
     */

    if (r.on) {

        if (r.s < 0 ||
            r.e < 0 ||
            r.s > r.e ||
            r.e >=
                static_cast<int64_t>(g.n_rows)) {

            std::cerr
                << "Invalid row range: "
                << r.s
                << " "
                << r.e
                << "\n";

            return 1;
        }
    }


    int64_t s =
        r.on ? r.s : 0;

    int64_t e =
        r.on
            ? r.e
            : static_cast<int64_t>(g.n_rows) - 1;


    /*
     * --------------------------------------------------------
     * Degree statistics
     * --------------------------------------------------------
     */

    std::cout
        << "\nDegree statistics\n";


    uint64_t edges = 0;


    std::vector<uint32_t> indeg(
        g.n_rows,
        0);


    uint32_t mino = UINT32_MAX;
    uint32_t maxo = 0;

    uint32_t mini = UINT32_MAX;
    uint32_t maxi = 0;


    for (int64_t u = s;
         u <= e;
         ++u) {

        uint32_t d = 0;

        auto* nb =
            g.neighbors(
                static_cast<IdxT>(u));

        uint32_t degree =
            g.degree(
                static_cast<IdxT>(u));


        for (uint32_t j = 0;
             j < degree;
             ++j) {

            IdxT v = nb[j];


            if (v >= g.n_rows)
                continue;

            /*
             * Ignore self-loops in the graph statistics,
             * matching the previous implementation.
             */
            if (v == static_cast<IdxT>(u))
                continue;

            if (!r.contains(v))
                continue;


            ++d;

            ++indeg[v];

            ++edges;
        }


        mino =
            std::min(
                mino,
                d);

        maxo =
            std::max(
                maxo,
                d);
    }


    uint64_t sumIn = 0;


    for (int64_t u = s;
         u <= e;
         ++u) {

        mini =
            std::min(
                mini,
                indeg[u]);

        maxi =
            std::max(
                maxi,
                indeg[u]);

        sumIn +=
            indeg[u];
    }


    double n =
        static_cast<double>(
            e - s + 1);


    size_t zeroIn = 0;


    for (int64_t u = s;
         u <= e;
         ++u) {

        if (indeg[u] == 0)
            ++zeroIn;
    }


    std::cout
        << "Edges="
        << edges
        << "\n";

    std::cout
        << "Out min/max="
        << mino
        << "/"
        << maxo
        << " avg="
        << edges / n
        << "\n";

    std::cout
        << "In min/max="
        << mini
        << "/"
        << maxi
        << " avg="
        << sumIn / n
        << "\n";

    std::cout
        << "Zero in-degree="
        << zeroIn
        << " ("
        << 100.0 *
               zeroIn /
               n
        << "%)\n";


    /*
     * --------------------------------------------------------
     * WCC
     * --------------------------------------------------------
     */

    int wc = 0;

    std::vector<int> wccSizes;


    wcc(
        g,
        r,
        wc,
        wccSizes);


    std::cout
        << "\nWCC count="
        << wc
        << "\n";


    if (!wccSizes.empty()) {

        std::cout
            << "Largest WCC="
            << *std::max_element(
                   wccSizes.begin(),
                   wccSizes.end())
            << "\n";
    }


    /*
     * --------------------------------------------------------
     * SCC
     * --------------------------------------------------------
     */

    std::cout
        << "\nComputing SCC...\n";


    SCC scc(
        g,
        r);


    scc.run();


    std::cout
        << "SCC count="
        << scc.count()
        << "\n";


    const auto& components =
        scc.components();


    std::vector<int> sccSizes(
        scc.count(),
        0);


    for (int64_t u = s;
         u <= e;
         ++u) {

        if (components[u] >= 0)
            ++sccSizes[
                components[u]];
    }


    if (!sccSizes.empty()) {

        std::cout
            << "Largest SCC="
            << *std::max_element(
                   sccSizes.begin(),
                   sccSizes.end())
            << "\n";
    }


    return 0;
}
