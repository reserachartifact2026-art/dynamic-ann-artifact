#include <omp.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "vamana_graph.h"
using namespace fda2;

static std::vector<float> load_fbin(const std::string& p, uint32_t& n,
                                    uint32_t& d) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open fbin: " + p);
  in.read((char*)&n, 4);
  in.read((char*)&d, 4);
  std::vector<float> x(uint64_t(n) * d);
  in.read((char*)x.data(), x.size() * 4);
  if (!in) throw std::runtime_error("Failed reading fbin");
  return x;
}
static inline float l2(const float* a, const float* b, uint32_t d) {
  float s = 0;
  for (uint32_t i = 0; i < d; ++i) {
    float z = a[i] - b[i];
    s += z * z;
  }
  return s;
}

int main(int argc, char** argv) {
  Options o;
  bool order = false, bidir = false;
  int Kcheck = -1;
  int64_t r0 = -1, r1 = -1;
  int threads = 0;
  std::string fbin;
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
    } else if (a == "--check-order") {
      order = true;
      Kcheck = std::stoi(argv[++i]);
      i++;
    } else if (a == "--check-bidirectional")
      bidir = true, i++;
    else if (a == "--row-range") {
      r0 = std::stoll(argv[++i]);
      r1 = std::stoll(argv[++i]);
    } else if (a == "--vectors-fbin") {
      fbin = argv[++i];
      i++;
    } else if (a == "--threads") {
      threads = std::stoi(argv[++i]);
      i++;
    } else
      break;
  }
  if (i >= argc || order && bidir) {
    std::cerr << "Usage: " << argv[0]
              << " --format f32|u8 --nodes N --dim D --degree R [--check-order "
                 "K|--check-bidirectional] [--row-range s e] [--vectors-fbin "
                 "file] [--threads T] graph.out\n";
    return 1;
  }
  if (threads) omp_set_num_threads(threads);
  Graph g = load_graph(argv[i], o);
  if (r0 < 0) {
    r0 = 0;
    r1 = g.n_rows - 1;
  }
  if (r0 < 0 || r1 >= int64_t(g.n_rows) || r0 > r1) return 1;
  if (bidir) {
    uint64_t E = 0, R = 0, full = 0;
    std::vector<double> p(r1 - r0 + 1);
#pragma omp parallel for reduction(+ : E, R, full) schedule(dynamic, 1024)
    for (int64_t u = r0; u <= r1; ++u) {
      uint64_t e = 0, r = 0;
      auto* nb = g.neighbors(u);
      for (uint32_t j = 0; j < g.degree(u); ++j) {
        IdxT v = nb[j];
        if (v >= g.n_rows || v == u) continue;
        ++e;
        if (g.has_edge(v, u)) ++r;
      }
      E += e;
      R += r;
      p[u - r0] = e ? 100.0 * r / e : 0;
      if (e && e == r) ++full;
    }
    std::sort(p.begin(), p.end());
    double avg = std::accumulate(p.begin(), p.end(), 0.0) / p.size();
    double med = p.size() % 2 ? p[p.size() / 2]
                              : (p[p.size() / 2 - 1] + p[p.size() / 2]) / 2;
    std::cout << "Edge Reciprocity      : " << 100.0 * R / E
              << "%\nAverage per-node      : " << avg
              << "%\nMedian                 : " << med
              << "%\nFully Bidirectional    : " << full << " ("
              << 100.0 * full / p.size() << "%)\n";
    return 0;
  }
  if (fbin.empty()) {
    std::cerr << "--vectors-fbin is required for distance-order/recall quality "
                 "checks.\n";
    return 1;
  }
  uint32_t nvec, dim;
  auto vec = load_fbin(fbin, nvec, dim);
  if (dim != g.dim || nvec < g.n_rows) {
    std::cerr << "Vector mismatch\n";
    return 1;
  }
  if (order) {
    if (Kcheck <= 0 || Kcheck > int(o.degree)) return 1;
    uint64_t good = 0;
#pragma omp parallel for reduction(+ : good) schedule(dynamic, 1024)
    for (int64_t u = r0; u <= r1; ++u) {
      bool ok = true;
      float prev = -1;
      auto* nb = g.neighbors(u);
      for (int j = 0; j < Kcheck && j < int(g.degree(u)); ++j) {
        IdxT v = nb[j];
        if (v >= g.n_rows) {
          ok = false;
          break;
        }
        float d = l2(&vec[uint64_t(u) * dim], &vec[uint64_t(v) * dim], dim);
        if (j && d < prev) {
          ok = false;
          break;
        }
        prev = d;
      }
      if (ok) ++good;
    }
    std::cout
        << "Ordering quality (rows with nondecreasing first K distances): "
        << 100.0 * good / (r1 - r0 + 1) << "%\n";
    return 0;
  }
  // Exact brute-force Recall@K, retained for small graphs just like the
  // artifact.
  if (g.n_rows > 100000) {
    std::cerr << "Default brute-force Recall@K is disabled for N>100000; use "
                 "--check-order or --check-bidirectional.\n";
    return 2;
  }
  uint32_t K = g.max_degree;
  double sum = 0;
#pragma omp parallel for reduction(+ : sum) schedule(dynamic, 4)
  for (int64_t u = r0; u <= r1; ++u) {
    std::vector<std::pair<float, uint32_t>> d;
    d.reserve(g.n_rows - 1);
    for (uint32_t v = 0; v < g.n_rows; ++v)
      if (v != u)
        d.push_back(
            {l2(&vec[uint64_t(u) * dim], &vec[uint64_t(v) * dim], dim), v});
    std::nth_element(d.begin(), d.begin() + std::min<size_t>(K, d.size()),
                     d.end());
    std::unordered_set<uint32_t> gt;
    for (uint32_t j = 0; j < K && j < d.size(); ++j) gt.insert(d[j].second);
    uint32_t correct = 0;
    for (uint32_t j = 0; j < g.degree(u); ++j)
      if (gt.count(g.neighbors(u)[j])) ++correct;
    sum += double(correct) / K;
  }
  std::cout << "Recall@K: " << sum / (r1 - r0 + 1) << "\n";
}
