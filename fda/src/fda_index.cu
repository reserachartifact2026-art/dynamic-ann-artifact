#include "fda_index.h"
#include <cstdio>
#include <cstring>
#include <climits>

namespace fda {

void allocateUnifiedIndex(UnifiedIndex* idx,
                          unsigned ltiCount,
                          unsigned tempCapacity,
                          unsigned maxTemps,
                          unsigned totalCapacity)
{
    if (maxTemps > 64) {
        fprintf(stderr, "allocateUnifiedIndex: maxTemps %u > 64 (tempMedoids array size)\n",
                maxTemps);
        exit(1);
    }
    // Total slot capacity supplied by caller; must cover all merge growth.
    size_t baselineMinimum = (size_t)ltiCount + (size_t)(maxTemps + 1) * tempCapacity;
    if (totalCapacity < baselineMinimum) {
        fprintf(stderr, "allocateUnifiedIndex: totalCapacity %u < L+(M+1)*T %zu\n",
                totalCapacity, baselineMinimum);
        exit(1);
    }
    size_t capacity = totalCapacity;

    size_t graphBytes = capacity * graphEntrySize;
    gpuErrchk(cudaMalloc(&idx->d_graph, graphBytes));
    gpuErrchk(cudaMemset(idx->d_graph, 0, graphBytes));

    // uint8 binary lock per node. atomicCAS on the byte is implemented
    // via byte-in-word manipulation in fda_locks.cuh.
    gpuErrchk(cudaMalloc(&idx->d_locks, capacity * sizeof(uint8_t)));
    gpuErrchk(cudaMemset(idx->d_locks, 0, capacity * sizeof(uint8_t)));

    idx->ltiCount       = ltiCount;
    idx->tempCapacity   = tempCapacity;
    idx->maxTemps       = maxTemps;
    idx->numActiveTemps = 0;
    idx->rwBase         = ltiCount;     // RW starts right after LTI
    idx->rwCount        = 0;
    idx->ltiMedoid      = 0;
    idx->rwMedoid       = UINT_MAX;     // unset until first insert
    idx->capacity       = (unsigned)capacity;
    for (unsigned i = 0; i < 64; i++) idx->tempMedoids[i] = UINT_MAX;
}

void freeUnifiedIndex(UnifiedIndex* idx) {
    if (idx->d_graph)        cudaFree(idx->d_graph);
    if (idx->d_locks)        cudaFree(idx->d_locks);
    idx->d_graph = nullptr;
    idx->d_locks = nullptr;
    idx->capacity = idx->ltiCount = 0;
}

void loadLTIFromHost(UnifiedIndex* idx, const uint8_t* h_graph,
                     unsigned numPoints, unsigned medoidId)
{
    if (numPoints > idx->ltiCount) {
        fprintf(stderr, "loadLTIFromHost: numPoints %u > ltiCount %u\n",
                numPoints, idx->ltiCount);
        return;
    }
    size_t bytes = (size_t)numPoints * graphEntrySize;
    gpuErrchk(cudaMemcpy(idx->d_graph, h_graph, bytes, cudaMemcpyHostToDevice));
    idx->ltiMedoid = medoidId;
}

void allocateDeltaMemory(uint8_t**  d_graph_out,
                         uint8_t**  d_locks_out,
                         unsigned   capacity)
{
    size_t graphBytes = (size_t)capacity * graphEntrySize;
    gpuErrchk(cudaMalloc(d_graph_out, graphBytes));
    gpuErrchk(cudaMemset(*d_graph_out, 0, graphBytes));

    gpuErrchk(cudaMalloc(d_locks_out, (size_t)capacity * sizeof(uint8_t)));
    gpuErrchk(cudaMemset(*d_locks_out, 0, (size_t)capacity * sizeof(uint8_t)));
}

void freeDeltaMemory(uint8_t*  d_graph,
                     uint8_t*  d_locks)
{
    if (d_graph)        cudaFree(d_graph);
    if (d_locks)        cudaFree(d_locks);
}

unsigned sealCurrentRW(UnifiedIndex* idx) {
    if (idx->numActiveTemps >= idx->maxTemps) {
        fprintf(stderr, "sealCurrentRW: already at max temps (%u)\n", idx->maxTemps);
        return UINT_MAX;
    }
    // Record the RW's medoid into the temp slot
    idx->tempMedoids[idx->numActiveTemps] = idx->rwMedoid;
    unsigned newTempIdx = idx->numActiveTemps;
    idx->numActiveTemps++;
    idx->rwBase += idx->tempCapacity;
    idx->rwCount = 0;
    idx->rwMedoid = UINT_MAX;            // unset until first insert into new RW
    return newTempIdx;
}

// ============================================================================
// DeleteListWrapper — unchanged from prior impl
// ============================================================================

void allocateDeleteList(DeleteListWrapper* dl, unsigned capacity) {
    unsigned numWords = (capacity + 31) / 32;
    gpuErrchk(cudaMalloc(&dl->d_bits, (size_t)numWords * sizeof(unsigned)));
    gpuErrchk(cudaMemset(dl->d_bits, 0, (size_t)numWords * sizeof(unsigned)));
    dl->capacity = capacity;
    dl->count    = 0;
}

void freeDeleteList(DeleteListWrapper* dl) {
    if (dl->d_bits) cudaFree(dl->d_bits);
    dl->d_bits = nullptr;
    dl->capacity = dl->count = 0;
}

void clearDeleteList(DeleteListWrapper* dl, cudaStream_t stream) {
    unsigned numWords = (dl->capacity + 31) / 32;
    gpuErrchk(cudaMemsetAsync(dl->d_bits, 0,
                              (size_t)numWords * sizeof(unsigned), stream));
    dl->count = 0;
}

void markDeleted(DeleteListWrapper* dl, unsigned id) {
    if (id >= dl->capacity) return;
    unsigned word = id / 32;
    unsigned mask = 1u << (id % 32);
    unsigned curr;
    gpuErrchk(cudaMemcpy(&curr, dl->d_bits + word, sizeof(unsigned),
                         cudaMemcpyDeviceToHost));
    if (!(curr & mask)) {
        curr |= mask;
        gpuErrchk(cudaMemcpy(dl->d_bits + word, &curr, sizeof(unsigned),
                             cudaMemcpyHostToDevice));
        dl->count++;
    }
}

__global__ void batchMarkDeletedKernel(unsigned* d_bits,
                                       const unsigned* d_ids,
                                       unsigned numIds, unsigned capacity)
{
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numIds) return;
    unsigned id = d_ids[i];
    if (id >= capacity) return;
    unsigned word = id / 32;
    unsigned mask = 1u << (id % 32);
    atomicOr(&d_bits[word], mask);
}

void batchMarkDeleted(DeleteListWrapper* dl,
                      const unsigned* d_ids, unsigned numIds,
                      cudaStream_t stream)
{
    if (numIds == 0) return;
    unsigned threads = 256;
    unsigned blocks  = (numIds + threads - 1) / threads;
    batchMarkDeletedKernel<<<blocks, threads, 0, stream>>>(
        dl->d_bits, d_ids, numIds, dl->capacity);
    dl->count += numIds;
}

} // namespace fda
