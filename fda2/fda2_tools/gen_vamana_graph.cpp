#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "vamana_graph.h"
using namespace fda2;
int main(int argc, char** argv) {
  if (argc < 8) {
    std::cerr << "Usage: " << argv[0]
              << " --format f32|u8 --nodes N --dim D --degree R --mode "
                 "zero|random|sequential --output file\n";
    return 1;
  }
  Options o;
  std::string mode, out;
  int i = 1;
  while (i < argc) {
    std::string a = argv[i++];
    if (a == "--format") {
      std::string x = argv[i++];
      o.vector_type = x == "u8" ? VectorType::UInt8 : VectorType::Float32;
    } else if (a == "--nodes")
      o.nodes = std::stoull(argv[i++]);
    else if (a == "--dim")
      o.dim = std::stoul(argv[i++]);
    else if (a == "--degree")
      o.degree = std::stoul(argv[i++]);
    else if (a == "--mode")
      mode = argv[i++];
    else if (a == "--output")
      out = argv[i++];
  }
  if (!o.nodes || !o.dim || !o.degree || out.empty()) return 1;
  std::ofstream os(out, std::ios::binary);
  if (!os) return 1;
  std::mt19937 rng(42);
  std::uniform_int_distribution<uint32_t> dn(0, o.nodes - 1);
  std::uniform_int_distribution<int> db(0, 255);
  std::vector<IdxT> nb(o.degree);
  for (uint64_t u = 0; u < o.nodes; ++u) {
    if (o.vector_type == VectorType::Float32) {
      for (uint32_t d = 0; d < o.dim; ++d) {
        float x = mode == "zero" ? 0.0f
                                 : (mode == "sequential"
                                        ? float((u + d) % 1000)
                                        : float(dn(rng)) / float(o.nodes));
        os.write((char*)&x, 4);
      }
    } else {
      for (uint32_t d = 0; d < o.dim; ++d) {
        uint8_t x = mode == "zero"
                        ? 0
                        : static_cast<uint8_t>(
                              mode == "sequential" ? (u + d) % 256 : db(rng));
        os.write((char*)&x, 1);
      }
    }
    uint32_t deg = o.degree;
    os.write((char*)&deg, 4);
    for (uint32_t j = 0; j < o.degree; ++j)
      nb[j] = mode == "sequential" ? IdxT((u + j + 1) % o.nodes) : dn(rng);
    os.write((char*)nb.data(), uint64_t(o.degree) * 4);
  }
  std::cout << "Wrote FDA2 Vamana graph: " << out << "\n";
  std::cout << "Record size: " << record_size(o) << " bytes\n";
}
