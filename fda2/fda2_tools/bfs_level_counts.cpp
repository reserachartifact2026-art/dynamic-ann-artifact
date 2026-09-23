#include <cstdlib>
#include <iostream>
#include <queue>

#include "vamana_graph.h"
using namespace fda2;
int main(int argc, char** argv) {
  Options o;
  int i = 1;
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
    } else
      break;
  }
  if (argc - i < 2) {
    std::cerr
        << "Usage: " << argv[0]
        << " --format f32|u8 --nodes N --dim D --degree R graph start_node\n";
    return 1;
  }
  std::string path = argv[i++];
  IdxT start = std::stoul(argv[i]);
  Graph g = load_graph(path, o);
  if (start >= g.n_rows) return 1;
  std::vector<uint8_t> seen(g.n_rows);
  std::queue<IdxT> q;
  seen[start] = 1;
  q.push(start);
  uint64_t cum = 1;
  int level = 0;
  std::cout << "level,nodes_visited,cumulative_visited\n";
  while (!q.empty()) {
    size_t fs = q.size(), next = 0;
    std::cout << level << "," << fs << "," << cum << "\n";
    for (size_t k = 0; k < fs; ++k) {
      IdxT u = q.front();
      q.pop();
      auto* p = g.neighbors(u);
      for (uint32_t j = 0; j < g.degree(u); ++j) {
        IdxT v = p[j];
        if (v < g.n_rows && !seen[v]) {
          seen[v] = 1;
          q.push(v);
          ++next;
        }
      }
    }
    cum += next;
    ++level;
  }
  if (cum != g.n_rows)
    std::cerr << "WARNING: BFS reached " << cum << " / " << g.n_rows
              << " nodes.\n";
}
