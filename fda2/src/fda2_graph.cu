#include "fda2_graph.h"
#include <cstdio>
#include <climits>

namespace fda2 {

void allocateUnifiedGraph(UnifiedGraph* g, unsigned capacity, unsigned dirtyCap) {
    size_t graphBytes = (size_t)capacity * graphEntrySize;
    gpuErrchk(cudaMalloc(&g->d_graph, graphBytes));
    gpuErrchk(cudaMemset(g->d_graph, 0, graphBytes));

    gpuErrchk(cudaMalloc(&g->d_locks, (size_t)capacity * sizeof(uint8_t)));
    gpuErrchk(cudaMemset(g->d_locks, 0, (size_t)capacity * sizeof(uint8_t)));

    // Per-slot deleted flag as a full uint array (NOT a bitvector): the baseline
    // greedy kernels read d_deleted[neighbor] as a per-slot value, so this lets
    // search/delete greedy skip tombstoned nodes with no kernel changes.
    gpuErrchk(cudaMalloc(&g->d_deleted, (size_t)capacity * sizeof(unsigned)));
    gpuErrchk(cudaMemset(g->d_deleted, 0, (size_t)capacity * sizeof(unsigned)));

    // Reverse-edge P-scratch.
    gpuErrchk(cudaMalloc(&g->d_addScratch, (size_t)capacity * P_EXTRA * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&g->d_addScratchCount, (size_t)capacity * sizeof(unsigned)));
    gpuErrchk(cudaMemset(g->d_addScratchCount, 0, (size_t)capacity * sizeof(unsigned)));

    gpuErrchk(cudaMalloc(&g->d_dirtyList, (size_t)dirtyCap * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&g->d_dirtyCount, sizeof(unsigned)));
    gpuErrchk(cudaMemset(g->d_dirtyCount, 0, sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&g->d_dirtyMark, (size_t)capacity * sizeof(uint8_t)));
    gpuErrchk(cudaMemset(g->d_dirtyMark, 0, (size_t)capacity * sizeof(uint8_t)));

    g->dirtyCap   = dirtyCap;
    g->numPoints  = 0;
    g->capacity   = capacity;
    g->medoid     = 0;
}

void freeUnifiedGraph(UnifiedGraph* g) {
    if (g->d_graph)        cudaFree(g->d_graph);
    if (g->d_locks)        cudaFree(g->d_locks);
    if (g->d_deleted)      cudaFree(g->d_deleted);
    if (g->d_addScratch)      cudaFree(g->d_addScratch);
    if (g->d_addScratchCount) cudaFree(g->d_addScratchCount);
    if (g->d_dirtyList)    cudaFree(g->d_dirtyList);
    if (g->d_dirtyCount)   cudaFree(g->d_dirtyCount);
    if (g->d_dirtyMark)    cudaFree(g->d_dirtyMark);
    g->d_graph = nullptr; g->d_locks = nullptr; g->d_deleted = nullptr;
    g->d_addScratch = nullptr; g->d_addScratchCount = nullptr;
    g->d_dirtyList = nullptr; g->d_dirtyCount = nullptr; g->d_dirtyMark = nullptr;
    g->numPoints = g->capacity = 0;
}

void loadGraphFromHost(UnifiedGraph* g, const uint8_t* h_graph,
                       unsigned numPoints, unsigned medoidId) {
    if (numPoints > g->capacity) {
        fprintf(stderr, "loadGraphFromHost: numPoints %u > capacity %u\n",
                numPoints, g->capacity);
        return;
    }
    size_t bytes = (size_t)numPoints * graphEntrySize;
    gpuErrchk(cudaMemcpy(g->d_graph, h_graph, bytes, cudaMemcpyHostToDevice));
    g->numPoints = numPoints;
    g->medoid    = medoidId;
}

__global__ void markDeletedKernel(unsigned* d_deleted,
                                  const unsigned* d_ids,
                                  unsigned n, unsigned capacity)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned id = d_ids[i];
    if (id < capacity) d_deleted[id] = 1u;
}

void markDeletedBatch(UnifiedGraph* g, const unsigned* d_ids, unsigned n,
                      cudaStream_t stream) {
    if (n == 0) return;
    unsigned threads = 256;
    unsigned blocks  = (n + threads - 1) / threads;
    markDeletedKernel<<<blocks, threads, 0, stream>>>(
        g->d_deleted, d_ids, n, g->capacity);
}

} // namespace fda2
