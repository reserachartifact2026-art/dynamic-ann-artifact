// metric.hpp — distance-metric support via loader-side input transforms.
//
// The graph and all CUDA kernels work in pure L2. Other metrics reduce to L2 by
// transforming vectors at the I/O boundary (same pattern as the dtype loader):
//
//   l2     : identity.
//   cosine : unit-normalize every vector (graph base + queries + inserts). For
//            unit vectors  ||a-b||² = 2 - 2·cos(a,b),  so nearest-L2 == max-cosine.
//   ip     : Max-Inner-Product via the BANG "extra dimension" reduction. A
//            database vector x (dim d) becomes [x, sqrt(M² - ||x||²)] and a query
//            q becomes [q, 0], where M = max database norm. Then
//            ||q_aug - x_aug||²  is monotonically decreasing in  <q,x>,  so
//            nearest-L2 in d+1 dims == max-inner-product in d dims.
//
// IP therefore needs the compile-time D to be (real_dim + 1), and the base graph
// must have been BUILT on the augmented vectors with the same M. cosine/l2 keep
// D == real_dim and work on existing graphs (cosine self-normalizes the graph).

#ifndef METRIC_HPP
#define METRIC_HPP

#include <cmath>
#include <cstring>
#include <cstdint>

namespace metric {

enum class Metric { L2, Cosine, IP };

inline Metric parse(const char* s) {
    if (!s) return Metric::L2;
    if (!std::strcmp(s, "cosine") || !std::strcmp(s, "cos"))                 return Metric::Cosine;
    if (!std::strcmp(s, "ip") || !std::strcmp(s, "mips") || !std::strcmp(s, "inner")) return Metric::IP;
    return Metric::L2;   // "l2" / anything else
}

inline const char* name(Metric m) {
    return m == Metric::Cosine ? "cosine" : (m == Metric::IP ? "ip" : "l2");
}

// Unit-normalize dim elements in place (no-op on a zero vector).
inline void normalize(float* v, unsigned dim) {
    double s = 0.0;
    for (unsigned i = 0; i < dim; i++) s += (double)v[i] * v[i];
    if (s > 0.0) {
        float inv = (float)(1.0 / std::sqrt(s));
        for (unsigned i = 0; i < dim; i++) v[i] *= inv;
    }
}

inline double norm2(const float* v, unsigned dim) {
    double s = 0.0;
    for (unsigned i = 0; i < dim; i++) s += (double)v[i] * v[i];
    return s;
}

// Database augmentation: out[0..dimIn) = in, out[dimIn] = sqrt(M² - ||in||²).
// out must hold dimIn+1 floats. M = max norm over the database (>= every ||in||).
inline void augmentDB(const float* in, unsigned dimIn, float maxNorm, float* out) {
    double s = 0.0;
    for (unsigned i = 0; i < dimIn; i++) { out[i] = in[i]; s += (double)in[i] * in[i]; }
    double extra = (double)maxNorm * maxNorm - s;
    if (extra < 0.0) extra = 0.0;          // guard against fp slack / a too-small M
    out[dimIn] = (float)std::sqrt(extra);
}

// Query augmentation: out[0..dimIn) = in, out[dimIn] = 0.
inline void augmentQuery(const float* in, unsigned dimIn, float* out) {
    for (unsigned i = 0; i < dimIn; i++) out[i] = in[i];
    out[dimIn] = 0.0f;
}

} // namespace metric

#endif // METRIC_HPP
