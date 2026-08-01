#include "deleteList.h"
#include <cuda_runtime.h>

// Helper macro for CUDA error checking
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// GPU kernel for batch marking points as deleted
__global__ void markDeletedKernel(unsigned int* d_deleted, unsigned* d_pointIds,
                                   unsigned count, unsigned* d_deleteCount) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < count) {
        unsigned pointId = d_pointIds[tid];

        // Atomically mark as deleted if not already deleted
        unsigned int oldVal = atomicExch(&d_deleted[pointId], 1);

        // If it wasn't deleted before, increment delete count
        if (oldVal == 0) {
            atomicAdd(d_deleteCount, 1);
        }
    }
}

// Constructor: Allocate GPU memory
DeleteList::DeleteList(unsigned maxPoints) : capacity(maxPoints) {
    // Allocate GPU memory for bitvector
    gpuErrchk(cudaMalloc(&d_deleted, maxPoints * sizeof(unsigned int)));
    gpuErrchk(cudaMemset(d_deleted, 0, maxPoints * sizeof(unsigned int))); // Initialize to 0

    // Allocate GPU memory for delete counter
    gpuErrchk(cudaMalloc(&d_deleteCount, sizeof(unsigned)));
    gpuErrchk(cudaMemset(d_deleteCount, 0, sizeof(unsigned))); // Initialize to 0

    printf("DeleteList initialized with capacity %u\n", maxPoints);
}

// Destructor: Free GPU memory
DeleteList::~DeleteList() {
    if (d_deleted) {
        cudaFree(d_deleted);
        d_deleted = nullptr;
    }
    if (d_deleteCount) {
        cudaFree(d_deleteCount);
        d_deleteCount = nullptr;
    }
}

// Mark a single point as deleted
void DeleteList::markDeleted(unsigned pointId) {
    if (pointId >= capacity) {
        fprintf(stderr, "Error: pointId %u exceeds capacity %u\n", pointId, capacity);
        return;
    }

    // Use batch operation with single point
    unsigned* d_pointId;
    gpuErrchk(cudaMalloc(&d_pointId, sizeof(unsigned)));
    gpuErrchk(cudaMemcpy(d_pointId, &pointId, sizeof(unsigned), cudaMemcpyHostToDevice));

    // Launch kernel with single thread
    markDeletedKernel<<<1, 1>>>(d_deleted, d_pointId, 1, d_deleteCount);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaFree(d_pointId));
}

// Mark multiple points as deleted in batch (optimized)
void DeleteList::batchMarkDeleted(unsigned* pointIds, unsigned count) {
    if (count == 0) return;

    // Allocate GPU memory for point IDs
    unsigned* d_pointIds;
    gpuErrchk(cudaMalloc(&d_pointIds, count * sizeof(unsigned)));
    gpuErrchk(cudaMemcpy(d_pointIds, pointIds, count * sizeof(unsigned),
                         cudaMemcpyHostToDevice));

    // Launch kernel with appropriate grid/block dimensions
    unsigned threadsPerBlock = 256;
    unsigned numBlocks = (count + threadsPerBlock - 1) / threadsPerBlock;

    markDeletedKernel<<<numBlocks, threadsPerBlock>>>(d_deleted, d_pointIds,
                                                       count, d_deleteCount);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaFree(d_pointIds));
}

// Check if a point is deleted (host version)
__host__ __device__ bool DeleteList::isDeleted(unsigned pointId) {
#ifdef __CUDA_ARCH__
    // Device code: direct access
    return d_deleted[pointId] != 0;
#else
    // Host code: need to copy from device
    unsigned int result;
    cudaMemcpy(&result, &d_deleted[pointId], sizeof(unsigned int), cudaMemcpyDeviceToHost);
    return result != 0;
#endif
}

// Get total number of deleted points
unsigned DeleteList::getDeleteCount() {
    unsigned count;
    gpuErrchk(cudaMemcpy(&count, d_deleteCount, sizeof(unsigned),
                         cudaMemcpyDeviceToHost));
    return count;
}

// Clear all deletions and reset counter
void DeleteList::clear() {
    gpuErrchk(cudaMemset(d_deleted, 0, capacity * sizeof(unsigned int)));
    gpuErrchk(cudaMemset(d_deleteCount, 0, sizeof(unsigned)));
}
