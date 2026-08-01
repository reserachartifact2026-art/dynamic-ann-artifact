#ifndef DELETE_LIST_H
#define DELETE_LIST_H

#include <cuda_runtime.h>
#include <stdio.h>

/**
 * DeleteList: Manages lazy deletion of points in FreshDiskANN
 *
 * Uses a GPU bitvector to track deleted points without immediately
 * modifying the graph structure. Points are marked as deleted and
 * filtered during search operations.
 *
 * Key Features:
 * - GPU-resident bitvector for O(1) deletion checks
 * - Atomic operations for concurrent marking
 * - Batch operations for efficiency
 * - Delete count tracking for consolidation threshold
 */
class DeleteList {
private:
    unsigned int* d_deleted;   // GPU bitvector: d_deleted[pointId] = 1 if deleted, 0 otherwise
    unsigned* d_deleteCount;   // GPU counter for total deleted points
    unsigned capacity;         // Maximum number of points (N)

public:
    /**
     * Constructor: Allocates GPU memory for maxPoints points
     * @param maxPoints Maximum number of points in the index
     */
    DeleteList(unsigned maxPoints);

    /**
     * Destructor: Frees GPU memory
     */
    ~DeleteList();

    /**
     * Mark a single point as deleted (thread-safe)
     * @param pointId Point to mark as deleted
     */
    void markDeleted(unsigned pointId);

    /**
     * Mark multiple points as deleted in batch (optimized)
     * @param pointIds Array of point IDs to delete
     * @param count Number of points to delete
     */
    void batchMarkDeleted(unsigned* pointIds, unsigned count);

    /**
     * Check if a point is deleted (GPU kernel callable)
     * @param pointId Point to check
     * @return true if point is deleted, false otherwise
     */
    __host__ __device__ bool isDeleted(unsigned pointId);

    /**
     * Get total number of deleted points
     * @return Number of deleted points
     */
    unsigned getDeleteCount();

    /**
     * Clear all deletions and reset counter
     */
    void clear();

    /**
     * Get device pointer for GPU kernel access
     * @return Raw pointer to d_deleted array
     */
    unsigned int* getDevicePointer() { return d_deleted; }

    /**
     * Get capacity (max number of points)
     * @return Capacity N
     */
    unsigned getCapacity() { return capacity; }
};

// GPU kernel for batch marking
__global__ void markDeletedKernel(unsigned int* d_deleted, unsigned* d_pointIds,
                                   unsigned count, unsigned* d_deleteCount);

// GPU kernel for checking deletion status
__device__ inline bool checkDeleted(unsigned int* d_deleted, unsigned pointId) {
    return d_deleted[pointId] != 0;
}

#endif // DELETE_LIST_H
