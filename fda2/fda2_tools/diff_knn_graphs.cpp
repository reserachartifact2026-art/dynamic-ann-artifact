#include <iostream>
#include <unordered_set>

#include "vamana_graph.h"
using namespace fda2;
static bool parse(int argc, char** argv, Options& o, int& argi) {
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
  argi = i;
  return true;
}
int main(int argc, char** argv) {
  Options o;
  int i;
  if (!parse(argc, argv, o, i) || argc - i < 2) {
    std::cerr << "Usage: " << argv[0]
              << " --format f32|u8 --nodes N --dim D --degree R gold.out "
                 "test.out [--row N] [--print]\n";
    return 1;
  }
  std::string goldp = argv[i++], testp = argv[i++];
  int row = -1;
  bool print = false;
  while (i < argc) {
    std::string a = argv[i++];
    if (a == "--row")
      row = std::stoi(argv[i++]);
    else if (a == "--print")
      print = true;
  }
  Graph g = load_graph(goldp, o), t = load_graph(testp, o);
  if (g.n_rows != t.n_rows || g.max_degree != t.max_degree) {
    std::cerr << "Graph shape mismatch\n";
    return 1;
  }
  uint64_t wrong = 0, slots = 0;
  auto check_row = [&](uint32_t r) {
    std::unordered_set<IdxT> s;
    for (uint32_t j = 0; j < g.degree(r); ++j) s.insert(g.neighbors(r)[j]);
    uint64_t w = 0;
    for (uint32_t j = 0; j < t.degree(r); ++j)
      if (!s.count(t.neighbors(r)[j])) ++w;
    if (print && w) {
      std::cout << "Row " << r << ": wrong=" << w << "\n";
    }
    return w;
  };
  if (row >= 0) {
    if (uint64_t(row) >= g.n_rows) return 1;
    std::cout << "Row " << row << " wrong=" << check_row(row) << "\n";
    return 0;
  }
  for (uint32_t r = 0; r < g.n_rows; ++r) {
    wrong += check_row(r);
    slots += t.degree(r);
  }
  std::cout << "Total wrong neighbors = " << wrong << " / " << slots
            << "\nAccuracy = " << (slots ? 1.0 - double(wrong) / slots : 1.0)
            << "\n";
}
