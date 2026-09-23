#include <omp.h>

#include <algorithm>
#include <array>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

#include "vamana_graph.h"
using namespace fda2;
int main(int argc, char** argv) {
  Options o;
  int i = 1;
  int64_t rs = -1, re = -1;
  bool hist = false;
  while (i < argc) {
    std::string a = argv[i];
    if (a == "--format") {
      std::string x = argv[++i];
      o.vector_type = x == "u8" ? VectorType::UInt8 : VectorType::Float32;
      i++;
    } else if (a == "--nodes") {
      o.nodes = std::stoull(argv[++i]);
      i++;
    } else if (a == "--dim") {
      o.dim = std::stoul(argv[++i]);
      i++;
    } else if (a == "--degree") {
      o.degree = std::stoul(argv[++i]);
      i++;
    } else if (a == "--threads") {
      int t = std::stoi(argv[++i]);
      omp_set_num_threads(t);
      i++;
    } else if (a == "--row-range") {
      rs = std::stoll(argv[++i]);
      re = std::stoll(argv[++i]);
    } else if (a == "--histogram")
      hist = true;
    else
      break;
  }
  if (i >= argc) {
    std::cerr << "Usage: " << argv[0]
              << " --format f32|u8 --nodes N --dim D --degree R graph.out "
                 "[--threads T] [--row-range s e] [--histogram]\n";
    return 1;
  }
  Graph g = load_graph(argv[i], o);
  if (rs < 0) {
    rs = 0;
    re = g.n_rows - 1;
  }
  std::vector<uint64_t> rec(g.n_rows), edges(g.n_rows);
#pragma omp parallel for schedule(dynamic, 1024)
  for (int64_t u = rs; u <= re; ++u) {
    auto* p = g.neighbors(u);
    for (uint32_t j = 0; j < g.degree(u); ++j) {
      IdxT v = p[j];
      if (v < g.n_rows && v != u) {
        ++edges[u];
        if (g.has_edge(v, u)) ++rec[u];
      }
    }
  }
  uint64_t E = 0, R = 0, full = 0;
  std::vector<double> pct(re - rs + 1);
  for (int64_t u = rs; u <= re; ++u) {
    E += edges[u];
    R += rec[u];
    pct[u - rs] = edges[u] ? 100.0 * rec[u] / edges[u] : 0;
    if (edges[u] && edges[u] == rec[u]) ++full;
  }
  std::sort(pct.begin(), pct.end());
  double avg = std::accumulate(pct.begin(), pct.end(), 0.0) / pct.size();
  double med = pct.size() % 2
                   ? pct[pct.size() / 2]
                   : (pct[pct.size() / 2 - 1] + pct[pct.size() / 2]) / 2;
  std::cout << "Valid Directed Edges : " << E
            << "\nReciprocal Edges     : " << R
            << "\nEdge Reciprocity     : " << std::fixed << std::setprecision(4)
            << 100.0 * R / E << "%\nAverage              : " << avg
            << "%\nMedian               : " << med
            << "%\nMinimum              : " << pct.front()
            << "%\nMaximum              : " << pct.back()
            << "%\nFully Bidirectional   : " << full << " ("
            << 100.0 * full / pct.size() << "%)\n";
  if (hist) {
    std::array<uint64_t, 10> h{};
    for (double x : pct) {
      int b = std::min(9, int(x / 10));
      ++h[b];
    }
    for (int b = 0; b < 10; ++b)
      std::cout << b * 10 << "-" << (b + 1) * 10 << "% : " << h[b] << "\n";
  }
}
