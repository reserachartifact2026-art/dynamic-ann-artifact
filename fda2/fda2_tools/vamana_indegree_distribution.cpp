#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "vamana_graph.h"
using namespace fda2;

int main(int argc, char** argv) {
  Options o;
  int i = 1;
  std::string csv;
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
    } else if (a == "--csv") {
      csv = argv[++i];
      i++;
    } else
      break;
  }
  if (i >= argc) {
    std::cerr << "Usage: " << argv[0]
              << " --format f32|u8 --nodes N --dim D --degree R graph.out "
                 "[--csv file]\n";
    return 1;
  }
  const std::string path = argv[i];
  uint64_t rec = record_size(o), actual = get_file_size(path);
  if (actual != o.nodes * rec) {
    std::cerr << "File size mismatch\n";
    return 1;
  }
  std::vector<uint32_t> indeg(o.nodes, 0);
  std::ifstream in(path, std::ios::binary);
  if (!in) return 1;
  uint64_t vb = o.vector_type == VectorType::Float32 ? uint64_t(o.dim) * 4
                                                     : uint64_t(o.dim);
  std::vector<char> vec(vb);
  std::vector<IdxT> nb(o.degree);
  uint64_t edges = 0;
  for (uint64_t u = 0; u < o.nodes; ++u) {
    in.read(vec.data(), vb);
    uint32_t d;
    in.read((char*)&d, 4);
    in.read((char*)nb.data(), uint64_t(o.degree) * 4);
    if (d > o.degree) return 1;
    for (uint32_t j = 0; j < d; ++j) {
      IdxT v = nb[j];
      if (v < o.nodes && v != u) {
        ++indeg[v];
        ++edges;
      }
    }
    if (u && u % 1000000ULL == 0)
      std::cerr << "Processed " << u << " / " << o.nodes << "\r" << std::flush;
  }
  std::cerr << "\n";
  auto s = indeg;
  std::sort(s.begin(), s.end());
  auto pct = [&](double p) {
    double x = p * (s.size() - 1);
    size_t a = x, b = std::min(a + 1, s.size() - 1);
    return s[a] * (1 - (x - a)) + s[b] * (x - a);
  };
  uint64_t sum = std::accumulate(indeg.begin(), indeg.end(), uint64_t(0));
  uint32_t mn = s.front(), mx = s.back();
  size_t z = std::count(indeg.begin(), indeg.end(), 0u);
  std::cout << "Total directed edges : " << edges
            << "\nAverage in-degree    : " << double(sum) / o.nodes
            << "\nMedian in-degree     : " << pct(.5)
            << "\nP90                  : " << pct(.9)
            << "\nP95                  : " << pct(.95)
            << "\nP99                  : " << pct(.99)
            << "\nP99.9                : " << pct(.999)
            << "\nMinimum              : " << mn
            << "\nMaximum              : " << mx
            << "\nZero in-degree nodes : " << z << " (" << 100.0 * z / o.nodes
            << "%)\n";
  std::vector<uint64_t> hist(mx + 1);
  for (auto d : indeg) ++hist[d];
  std::cout << "\nIn-degree distribution\n";
  for (uint32_t d = 0; d <= mx; ++d)
    if (hist[d]) std::cout << d << "," << hist[d] << "\n";
  if (!csv.empty()) {
    std::ofstream out(csv);
    out << "in_degree,count\n";
    for (uint32_t d = 0; d <= mx; ++d)
      if (hist[d]) out << d << "," << hist[d] << "\n";
  }
}
