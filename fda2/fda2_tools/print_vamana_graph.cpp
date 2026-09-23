#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "vamana_graph.h"

using namespace fda2;

static std::vector<float> load_fbin(const std::string& p, uint32_t& n,
                                    uint32_t& d) {
  std::ifstream in(p, std::ios::binary);

  if (!in) throw std::runtime_error("Cannot open fbin: " + p);

  in.read(reinterpret_cast<char*>(&n), 4);
  in.read(reinterpret_cast<char*>(&d), 4);

  std::vector<float> x(uint64_t(n) * d);

  in.read(reinterpret_cast<char*>(x.data()), x.size() * sizeof(float));

  return x;
}

static float l2(const float* a, const float* b, uint32_t d) {
  float s = 0;

  for (uint32_t i = 0; i < d; ++i) {
    float x = a[i] - b[i];
    s += x * x;
  }

  return std::sqrt(s);
}

int main(int argc, char** argv) {
  Options o;

  int i = 1;

  bool dist = false;
  std::string fbin;

  // Parse options
  while (i < argc) {
    std::string a = argv[i];

    if (a == "--format") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --format\n";
        return 1;
      }

      std::string x = argv[++i];

      if (x == "u8") {
        o.vector_type = VectorType::UInt8;
      } else if (x == "f32") {
        o.vector_type = VectorType::Float32;
      } else {
        std::cerr << "Invalid format: " << x << ". Use f32 or u8.\n";
        return 1;
      }

      ++i;
    }

    else if (a == "--nodes") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --nodes\n";
        return 1;
      }

      o.nodes = std::stoull(argv[++i]);
      ++i;
    }

    else if (a == "--dim") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --dim\n";
        return 1;
      }

      o.dim = std::stoul(argv[++i]);
      ++i;
    }

    else if (a == "--degree") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for --degree\n";
        return 1;
      }

      o.degree = std::stoul(argv[++i]);
      ++i;
    }

    else if (a == "-d") {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for -d\n";
        return 1;
      }

      dist = true;
      fbin = argv[++i];
      ++i;
    }

    else {
      // First non-option argument is the graph file.
      break;
    }
  }

  // Graph file is mandatory.
  if (i >= argc) {
    std::cerr << "Usage: " << argv[0] << " --format f32|u8"
              << " --nodes N"
              << " --dim D"
              << " --degree R"
              << " graph.out [start] [end]"
              << " [-d vectors.fbin]\n";

    return 1;
  }

  // IMPORTANT:
  // The current argument is the graph path.
  std::string graph_path = argv[i++];

  // Optional start/end node range.
  uint32_t start = 0;
  uint32_t end = o.nodes - 1;

  if (i < argc) {
    start = std::stoul(argv[i++]);
  }

  if (i < argc) {
    end = std::stoul(argv[i++]);
  }

  // Load graph.
  Graph g = load_graph(graph_path, o);

  if (start >= g.n_rows || end >= g.n_rows || start > end) {
    std::cerr << "Invalid node range: " << start << " " << end << "\n";

    return 1;
  }

  // Optional vector data for distance printing.
  std::vector<float> v;

  uint32_t nv = 0;
  uint32_t vd = 0;

  if (dist) {
    if (o.vector_type != VectorType::Float32) {
      std::cerr << "Distance printing currently supports "
                << "float32 vectors only.\n";

      return 1;
    }

    v = load_fbin(fbin, nv, vd);

    if (vd != o.dim || nv < g.n_rows) {
      std::cerr << "Invalid vector file dimensions.\n";

      return 1;
    }
  }

  // Print graph.
  for (uint32_t u = start; u <= end; ++u) {
    std::cout << u << ": ";

    auto* p = g.neighbors(u);

    for (uint32_t j = 0; j < g.degree(u); ++j) {
      IdxT x = p[j];

      std::cout << x;

      if (dist && x < g.n_rows) {
        std::cout << "(" << std::fixed << std::setprecision(3)
                  << l2(&v[uint64_t(u) * o.dim], &v[uint64_t(x) * o.dim], o.dim)
                  << ")";
      }

      std::cout << " ";
    }

    std::cout << "\n";
  }

  return 0;
}
