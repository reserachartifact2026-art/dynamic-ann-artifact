#include <time.h>
#include <stdlib.h>
#include <cub/cub.cuh>

#ifndef VAMANA_H
#include "vamana.h"
#endif

// Define WarpReduce for 8 threads (matching our thread-per-dimension pattern)
#define THREADS_PER_NEIGHBOR 8
typedef cub::WarpReduce<float, THREADS_PER_NEIGHBOR> WarpReduce8;

void generateRandomGraph(uint8_t *graph, unsigned batchStart, unsigned batchSize) {
    srand(0);
    for (int i = batchStart; i < batchSize; i++) {
        unsigned* neighbors = (unsigned*)(graph + (i * graphEntrySize + D*sizeof(GVEC))) + 1;
        for (int j = 0; j < R; j++) {
            neighbors[j] = (rand() % batchSize) + batchStart;
        }
    }
}

__global__ void computeDists(uint8_t *d_graph,
                             unsigned *d_nodes,
                             unsigned *d_nodeCount,
                             GVEC *d_queryVecs,
                             float *d_dists,
                             unsigned rowSize,
                             const unsigned *d_querySlots) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    // Query vector source: when d_querySlots is given, read it in place from the
    // graph entry of slot d_querySlots[queryID] (vector lives at offset 0). This
    // lets the DELETE path use the deleted nodes' own vectors with no gather/copy.
    // When null, the legacy contiguous layout d_queryVecs + D*queryID is used.
    GVEC *queryVec = d_querySlots
        ? (GVEC*)(d_graph + graphEntrySize * (size_t)d_querySlots[queryID])
        : d_queryVecs + D*queryID;                      // Pointer to query vector
    unsigned offset = rowSize * queryID;
    unsigned numNodes = d_nodeCount[queryID];

    // Shared memory for WarpReduce - one per possible neighbor
    __shared__ typename WarpReduce8::TempStorage temp_storage[R+1];

    // Initialize distances to zero
    for (unsigned i = tid; i < numNodes; i += blockDim.x) {
        d_dists[offset + i] = 0;
    }

    __syncthreads();

    // Assign 8 threads to each node - use WarpReduce instead of atomicAdd
    unsigned neighborIdx = tid / THREADS_PER_NEIGHBOR;
    unsigned dimOffset = tid % THREADS_PER_NEIGHBOR;

    for (unsigned j = neighborIdx; j < numNodes; j += blockDim.x / THREADS_PER_NEIGHBOR) {
        unsigned node = d_nodes[offset + j];
        GVEC *nodeVec = (GVEC*)(d_graph + graphEntrySize*node); // Pointer to node vector
        float sum = 0;

        // Each of 8 threads computes D/8 dimensions
        for (unsigned i = dimOffset; i < D; i += THREADS_PER_NEIGHBOR) {
            float diff = nodeVec[i] - queryVec[i];
            sum += diff * diff;
        }

        // Use CUB WarpReduce - much faster than atomicAdd (3 cycles vs 30 cycles)
        float totalDist = WarpReduce8(temp_storage[j % (R+1)]).Sum(sum);

        // Only thread 0 of each 8-thread group writes the result
        if (dimOffset == 0) {
            d_dists[offset + j] = totalDist;
        }
    }
}

__device__ unsigned lowerBound(float arr[], unsigned lo, unsigned hi, float target) {
    while (lo < hi) {
        unsigned mid = (lo + hi) / 2;
        if (target > arr[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__device__ unsigned upperBound(float arr[], unsigned lo, unsigned hi, float target) {
    while (lo < hi) {
        unsigned mid = (lo + hi) / 2;
        if (target >= arr[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__global__ void sortByDistance(unsigned *d_items,
                               unsigned *d_itemCount,
                               float *d_dists,
                               unsigned *d_itemsAux,
                               float *d_distsAux,
                               unsigned rowSize) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    unsigned numItems = d_itemCount[queryID];
    unsigned offset = queryID * rowSize;

    extern __shared__ unsigned sortedPositions[];

    for (unsigned subarraySize = 2; subarraySize < 2 * numItems; subarraySize *= 2) {
        unsigned subarrayID = tid / subarraySize;
        unsigned start = subarrayID * subarraySize;
        unsigned mid = min(start + subarraySize / 2, numItems);
        unsigned end = min(start + subarraySize, numItems);        

        unsigned before;

        if (tid >= start && tid < mid) {
            // If current thread corresponds to lower half, find the no. of elements before this element from the upper half
            before = lowerBound(&d_dists[offset + mid], 0, end-mid, d_dists[offset + tid]);
            sortedPositions[tid] = tid + before;
        } else if (tid >= mid && tid < end) {
            // If current thread corresponds to upper half, find the no. of elements before this element from the lower half
            before = upperBound(&d_dists[offset + start], 0, mid-start, d_dists[offset + tid]);
            sortedPositions[tid] = before + (tid - mid + start); 
        }
       
        __syncthreads();
        __threadfence_block();

        // Copy the neigbors to correct positions in auxiliary array
        for (int i = tid; i < numItems; i += blockDim.x) {
            d_itemsAux[offset + sortedPositions[i]] = d_items[offset + i];
            d_distsAux[offset + sortedPositions[i]] = d_dists[offset + i];
        }

        __syncthreads();
        __threadfence_block();

        // Copy from auxiliary array back into original array
        for (int i = tid; i < numItems; i += blockDim.x) {
            d_items[offset + i] = d_itemsAux[offset + i];
            d_dists[offset + i] = d_distsAux[offset + i];
        }

        __syncthreads();
        __threadfence_block();   
    }
}
