#ifndef VAMANA_H
#define VAMANA_H

#include <stdint.h>
#include <stdio.h>
#include "timer.h"

#define BF_ENTRIES 399887U   	 // per query, max entries in BF, (prime number)
const unsigned BF_MEMORY = (BF_ENTRIES & 0xFFFFFFFC) + sizeof(unsigned); // 4-byte mem aligned size for actual allocation

typedef enum { INIT, PRUNED, NEIGHBOR } NodeState;

// Number of vertices in graph
#define N 1000000
#define NUM_QUERIES 10000

// Number of dimensions of a vector
#define D 128
// Maximum degree of a vertex
#define R 64
// Worklist size
#define L 150  // Default L for graph construction
#define MAX_L 200  // Maximum L for dynamic search (static array sizing)

// Thread-block dimension for kernels that cooperatively process a D-dim
// vector (vector copies, block-cooperative L2 distance, vector gathers).
// Derived from D so D ≠ 128 doesn't leave threads idle (D<blockDim) or
// over-subscribed (D>>blockDim):
//   - rounded UP to a multiple of 32 (warp size) so warp-shuffle reduces
//     in l2DistBlock are full-warp,
//   - clamped to [32, 256]. The upper bound is structural: l2DistBlock's
//     cross-warp reduce uses smemAccum[8] (= up to 8 warps). Above 256
//     threads the per-warp partials would overflow that array.
//
// For D=128 this is 128 (one thread per dim, matches old hardcoded value).
// For D= 64 → 64;   D=200 → 224;   D=256 → 256;   D=384 → 256 (each
// thread strides over ⌈D/blockDim⌉ dims inside the kernel — still correct,
// just less perfectly parallel).
//
// Use ONLY for kernels that are dim-driven (vector copy + l2DistBlock).
// Do NOT use for kernels that are visited-set-driven (≤MAX_PARENTS_PERQUERY)
// or neighbor-driven (R / R+P / R+MAX_L) — those have their own optimal
// block size unrelated to D.
#define BLOCK_D ((D) < 32 ? 32 : ((D) > 256 ? 256 : (((D) + 31) & ~31)))

// Maximum no. of parents to keep track of.
// Power-of-2 — required by bitonicSortFused inside the fused greedy kernel.
#define MAX_PARENTS_PERQUERY 512
// Maximum no. of reverse index entries to keep track of
#define MAX_REVERSE_INDEX_ENTRIES 500
// Index of medioid node
#define MEDOID 5000

#define VISITED_NODES_SIZE 10000

// Size of one vector in the graph.
// size_t (NOT unsigned): node-id × graphEntrySize must be 64-bit. At 10M+ nodes a
// 32-bit product overflows (e.g. id 6.67M × 644 > 2³²) → wrapped address → illegal
// memory access in the non-fused kernels. size_t promotes every `node *
// graphEntrySize` to 64-bit across search/insert/delete/merge at once.
#ifndef FDA2_STORE_T
#define FDA2_STORE_T float
#endif
using GVEC = FDA2_STORE_T;

const size_t graphEntrySize = D*sizeof(GVEC) + sizeof(unsigned) + R*sizeof(unsigned);
const unsigned reverseIndexEntrySize = (MAX_REVERSE_INDEX_ENTRIES + 1) * sizeof(unsigned);

// Templated entry size — needed by device code that's templated on Idx/Vec
// (used by the fused greedy and out-neighbor kernels ported from BANG/Tree B).
template<typename Idx, typename Vec>
const size_t graphEntrySzT = D*sizeof(Vec) + sizeof(Idx) + R*sizeof(Idx);

// Bloom Filter
__device__ unsigned bf_hashFn1(unsigned x);
__device__ unsigned bf_hashFn2(unsigned x);
__device__ bool bf_check(bool *bf, unsigned x);
__device__ void bf_set(bool *bf, unsigned x);

void generateRandomGraph(uint8_t *graph, unsigned batchStart, unsigned batchSize);

// Greedy Search
__global__ void filterNeighbors(uint8_t *d_graph,
                                bool *d_hasParent,
                                unsigned *d_parents,
                                bool *d_bloomFilters,
                                unsigned *d_neighbors,
                                unsigned *d_neighborsCount,
                                unsigned *d_visitedSet,
                                unsigned *d_visitedSetCount);

__global__ void computeNeighborDists(uint8_t *d_graph,
                                     unsigned *d_neighbors,
                                     unsigned *d_neighborsCount,
                                     GVEC *d_queryVecs,
                                     float *d_neighborDists);

__global__ void sortNeighborDists(unsigned *d_neighbors,
                                  unsigned *d_neighborsCount,
                                  float *d_neighborsDist,
                                  unsigned *d_neighborsAux,
                                  float *d_neighborsDistAux);

// Given a sorted list, find the no. of elements less than or equal to target
__device__ unsigned lowerBound(float arr[], unsigned lo, unsigned hi, float target);
__device__ unsigned upperBound(float arr[], unsigned lo, unsigned hi, float target);

__global__ void mergeIntoWorklist(unsigned *d_worklistCount,
                                  unsigned *d_worklist,
                                  float *d_worklistDist,
                                  unsigned *d_worklistVisited,

                                  unsigned d_neighborsCount,
                                  unsigned *d_neighbors,
                                  float *d_neighborsDist,

                                  bool *d_hasParent,
                                  unsigned *d_parents,
                                  bool *d_nextIter);


// Out Neighbor Computation
__global__ void getNeighbors(uint8_t *d_graph,
                             unsigned batchStart,
                             unsigned *d_neighbors,
                             unsigned *d_neighborsCount);

__global__ void computeVisitedSetDists(uint8_t *d_graph,
                                       unsigned *d_visitedSet,
                                       unsigned *d_visitedSetCount,
                                       GVEC *d_queryVecs,
                                       float *d_visitedSetDists);
  
__global__ void sortVisitedSets(unsigned *d_visitedSet,
                                unsigned *d_visitedSetCount,
                                float *d_visitedSetDists,
                                unsigned *d_visitedSetAux,
                                float *d_visitedSetDistsAux);

__global__ void mergeNeighborsIntoVisitedSet(unsigned *d_visitedSetCount,
                                             unsigned *d_visitedSet,
                                             float *d_visitedSetDists,

                                             unsigned *d_neighborsCount,
                                             unsigned *d_neighbors,
                                             float *d_neighborsDist);


__global__ void pruneOutNeighbors(uint8_t *d_graph,
                                    unsigned batchStart,
                                    unsigned *d_visitedSet,
                                    unsigned *d_visitedSetCount,
                                    float *d_visitedSetDists,
                                    NodeState *d_visitedSetStatus,

                                    GVEC *d_queryVecs,
                                    uint8_t *d_reverseEdgeIndex,
                                    float alpha,
                                    unsigned iter,
                                    bool *d_nextIter);

void greedySearch(uint8_t *d_graph,
                  GVEC *d_queryVecs,
                  unsigned *d_visitedSets,
                  unsigned *d_visitedSetCount,
                  unsigned batchStart,
                  unsigned batchSize,
                  unsigned searchL,
                  unsigned int *d_deleted = nullptr);

// Version-based greedy search for concurrent safety
void greedySearchVersioned(uint8_t *d_graph,
                           unsigned *d_versions,
                           GVEC *d_queryVecs,
                           unsigned *d_visitedSets,
                           unsigned *d_visitedSetCount,
                           unsigned batchStart,
                           unsigned batchSize,
                           unsigned searchL,
                           unsigned int *d_deleted = nullptr);

// Pre-allocated buffers for greedySearchVersioned to avoid per-query cudaMalloc
struct GreedySearchBuffers {
    bool *d_hasParent;
    unsigned *d_parents;
    bool *d_bloomFilters;
    unsigned *d_neighbors;
    unsigned *d_neighborsCount;
    float *d_neighborDists;
    unsigned *d_neighborsAux;
    float *d_neighborDistsAux;
    unsigned *d_worklist;
    unsigned *d_worklistCount;
    float *d_worklistDist;
    bool *d_worklistVisited;
    bool *d_nextIter;
    bool *h_nextIter;    // Pinned host memory for async copy
    unsigned batchSize;  // Size these were allocated for
};

// Pre-allocated buffers for computeOutNeighbors to avoid cudaMalloc overhead
struct OutNeighborsBuffers {
    float *d_visitedSetDists;
    unsigned *d_visitedSetAux;
    float *d_visitedSetDistsAux;
    NodeState *d_visitedSetStatus;
    unsigned *d_neighbors;
    unsigned *d_neighborsCount;
    float *d_neighborsDists;
    unsigned *d_neighborsAux;
    float *d_neighborsDistsAux;
    unsigned batchSize;  // Size these were allocated for
};

// Allocate/free pre-allocated buffers (call once at startup per stream)
void allocateGreedySearchBuffers(GreedySearchBuffers* buffers, unsigned batchSize);
void freeGreedySearchBuffers(GreedySearchBuffers* buffers);

// Version using pre-allocated buffers (much faster for repeated queries)
void greedySearchVersionedPrealloc(uint8_t *d_graph,
                                    unsigned *d_versions,
                                    GVEC *d_queryVecs,
                                    unsigned *d_visitedSets,
                                    unsigned *d_visitedSetCount,
                                    unsigned batchStart,
                                    unsigned batchSize,
                                    unsigned searchL,
                                    unsigned int *d_deleted,
                                    GreedySearchBuffers* buffers,
                                    cudaStream_t stream = 0);

// Override the compile-time MEDOID with a runtime value. Used by callers
// (e.g. FreshDiskANN-Naive) where the actual graph medoid differs from
// the vamana.h MEDOID constant (e.g., small LTI scenarios).
void setRuntimeMedoid(unsigned medoid, cudaStream_t stream = 0);

// ============================================================================
// FreshDiskANN paper-compliant greedy wrappers.
//
// greedySearchLockedPrealloc  — INSERT path. Per-node binary lock acquired
//                               around adjacency reads ("Before reading
//                               the adjacency list of a node, use the lock.").
// greedySearchNoSyncPrealloc  — SEARCH path. No locks, no version check (Note:
//                               "No need to take locks while reading the
//                               adjacency list.").
// ============================================================================
void greedySearchLockedPrealloc(uint8_t *d_graph,
                                 uint8_t *d_locks,
                                 GVEC *d_queryVecs,
                                 unsigned *d_visitedSets,
                                 unsigned *d_visitedSetCount,
                                 unsigned batchStart,
                                 unsigned batchSize,
                                 unsigned searchL,
                                 unsigned int *d_deleted,
                                 GreedySearchBuffers* buffers,
                                 cudaStream_t stream = 0);

void greedySearchNoSyncPrealloc(uint8_t *d_graph,
                                 GVEC *d_queryVecs,
                                 unsigned *d_visitedSets,
                                 unsigned *d_visitedSetCount,
                                 unsigned batchStart,
                                 unsigned batchSize,
                                 unsigned searchL,
                                 unsigned int *d_deleted,
                                 GreedySearchBuffers* buffers,
                                 cudaStream_t stream = 0);


void computeOutNeighbors(uint8_t *d_graph,
                         GVEC *d_queryVecs,
                         unsigned *d_visitedSets,
                         unsigned *d_visitedSetCount,
                         float alpha,
                         uint8_t *d_reverseEdgeIndex,
                         unsigned batchStart,
                         unsigned batchSize);

// Allocate/free OutNeighbors buffers
void allocateOutNeighborsBuffers(OutNeighborsBuffers* buffers, unsigned batchSize);
void freeOutNeighborsBuffers(OutNeighborsBuffers* buffers);

// Pre-allocated version (much faster - no cudaMalloc overhead)
void computeOutNeighborsPrealloc(uint8_t *d_graph,
                                  GVEC *d_queryVecs,
                                  unsigned *d_visitedSets,
                                  unsigned *d_visitedSetCount,
                                  float alpha,
                                  uint8_t *d_reverseEdgeIndex,
                                  unsigned batchStart,
                                  unsigned batchSize,
                                  OutNeighborsBuffers* buffers,
                                  cudaStream_t stream = 0);


void computeReverseEdges(uint8_t *d_graph,
                         uint8_t *d_reverseEdgeIndex,
                         float alpha);

__global__ void computeDists(uint8_t *d_graph,
                             unsigned *d_nodes,
                             unsigned *d_nodeCount,
                             GVEC *d_queryVecs,
                             float *d_dists,
                             unsigned rowSize);


__global__ void sortByDistance(unsigned *d_items,
                               unsigned *d_itemCount,
                               float *d_dists,
                               unsigned *d_itemsAux,
                               float *d_distsAux,
                               unsigned rowSize);

// ===== Fused greedy search (BANG-style single-kernel search) =====
// Same I/O contract as greedySearchVersionedPrealloc but runs the entire
// search inside one kernel launch. Writes the final sorted visited set
// (parents) into d_visitedSets (stride MAX_PARENTS_PERQUERY) and the
// corresponding distances into buffers->d_visitedSetDists for downstream
// L0 augment. d_versions is unused; kept for API symmetry.
void greedySearchFusedPrealloc(uint8_t *d_graph,
                                GVEC    *d_queryVecs,
                                unsigned *d_visitedSets,
                                unsigned *d_visitedSetCount,
                                float    *d_visitedSetDists,
                                unsigned  batchStart,
                                unsigned  batchSize,
                                unsigned  searchL,
                                unsigned int *d_deleted,
                                GreedySearchBuffers* buffers,
                                cudaStream_t stream = 0);

// ===== Fused LOCKED greedy for the INSERT path =====
// Design: per-node uint8 binary lock around every adjacency read.
// entryPoint passed in (not the MEDOID constant). d_visitedSetDists may
// be nullptr if the caller (insert path) doesn't need it.
void greedySearchFusedLockedPrealloc(uint8_t *d_graph,
                                      uint8_t *d_locks,
                                      GVEC    *d_queryVecs,
                                      unsigned *d_visitedSets,
                                      unsigned *d_visitedSetCount,
                                      float    *d_visitedSetDists,
                                      unsigned  batchStart,
                                      unsigned  batchSize,
                                      unsigned  searchL,
                                      unsigned  entryPoint,
                                      GreedySearchBuffers* buffers,
                                      cudaStream_t stream = 0);

#define gpuErrchk(ans) {gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort=true) {
    if(code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif // VAMANA_H
