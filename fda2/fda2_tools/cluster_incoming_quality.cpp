#include <omp.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include "vamana_graph.h"
using namespace fda2;
static std::vector<float> load_fbin(const std::string& p, uint32_t& n,
                                    uint32_t& d) {
  std::ifstream in(p, std::ios::binary);
  if (!in) throw std::runtime_error("Cannot open fbin");
  in.read((char*)&n, 4);
  in.read((char*)&d, 4);
  std::vector<float> x(uint64_t(n) * d);
  in.read((char*)x.data(), x.size() * 4);
  return x;
}
static float l2(const float* a, const float* b, uint32_t d) {
  float s = 0;
  for (uint32_t i = 0; i < d; ++i) {
    float z = a[i] - b[i];
    s += z * z;
  }
  return s;
}
int main(int argc, char** argv) {
  Options o;
  uint32_t k = 0;
  int iters = 10, limit = 0;
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
    } else if (a == "--clusters") {
      k = std::stoul(argv[++i]);
      i++;
    } else if (a == "--iters") {
      iters = std::stoi(argv[++i]);
      i++;
    } else if (a == "--print-missing") {
      limit = std::stoi(argv[++i]);
      i++;
    } else
      break;
  }
  if (i >= argc || !k) {
    std::cerr << "Usage: ... --clusters K graph.out vectors.fbin\n";
    return 1;
  }
  Graph g = load_graph(argv[i], o);
  uint32_t n, d;
  auto x = load_fbin(argv[i + 1], n, d);
  if (n < g.n_rows || d != g.dim || o.vector_type != VectorType::Float32)
    return 1;
  std::vector<uint32_t> asg(g.n_rows);
  std::vector<float> cent(k * d), next(k * d);
  std::vector<uint64_t> cnt(k);
  for (uint32_t c = 0; c < k; ++c)
    std::copy_n(&x[uint64_t(std::min<uint64_t>(g.n_rows - 1,
                                               uint64_t(c) * g.n_rows / k)) *
                   d],
                d, &cent[uint64_t(c) * d]);
  for (int it = 0; it < iters; ++it) {
#pragma omp parallel for schedule(static)
    for (int64_t u = 0; u < int64_t(g.n_rows); ++u) {
      uint32_t best = 0;
      float bd = std::numeric_limits<float>::max();
      for (uint32_t c = 0; c < k; ++c) {
        float z = l2(&x[uint64_t(u) * d], &cent[uint64_t(c) * d], d);
        if (z < bd) {
          bd = z;
          best = c;
        }
      }
      asg[u] = best;
    }
    std::fill(next.begin(), next.end(), 0);
    std::fill(cnt.begin(), cnt.end(), 0);
    for (uint64_t u = 0; u < g.n_rows; ++u) {
      uint32_t c = asg[u];
      ++cnt[c];
      for (uint32_t j = 0; j < d; ++j)
        next[uint64_t(c) * d + j] += x[u * uint64_t(d) + j];
    }
    for (uint32_t c = 0; c < k; ++c) {
      if (!cnt[c]) continue;
      for (uint32_t j = 0; j < d; ++j)
        cent[uint64_t(c) * d + j] = next[uint64_t(c) * d + j] / cnt[c];
    }
  }
  auto rev = build_reverse_graph(g);
  std::vector<std::vector<IdxT>> members(k);
  for (IdxT u = 0; u < g.n_rows; ++u) members[asg[u]].push_back(u);
  uint64_t gt = 0, found = 0;
  double recallSum = 0;
  for (IdxT v = 0; v < g.n_rows; ++v) {
    uint64_t r = 0;
    for (IdxT u : members[asg[v]])
      if (g.has_edge(u, v)) ++r;
    gt += rev[v].size();
    found += r;
    recallSum += rev[v].empty() ? 100.0 : 100.0 * r / rev[v].size();
  }
  std::cout << "Incoming Edge Recall: " << (gt ? 100.0 * found / gt : 100.0)
            << "%\nAverage Node Recall: " << recallSum / g.n_rows << "%\n";
}
