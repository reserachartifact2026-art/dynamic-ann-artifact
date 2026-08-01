#ifndef VAMANA_H
#include "vamana.h"
#endif

__device__ bool contains(unsigned *set, unsigned count, unsigned el) {
    for (int i = 0; i < count; i++) {
        if (set[i] == el) {
            return true;
        }
    }

    return false;
}

// Runtime-configurable medoid: __device__ global set by greedySearchVersionedPrealloc
// before each launch. Defaults to the compile-time MEDOID for backward compat.
__device__ unsigned g_runtimeMedoid = MEDOID;

__global__ void initializeParents(bool *d_hasParent, unsigned *d_parents) {
    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    if (tid == 0) {
        d_hasParent[queryID] = true;
        d_parents[queryID] = g_runtimeMedoid;
    }
}

__global__ void initializeWorklist(uint8_t *d_graph,
                                   GVEC *d_queryVecs,
                                   unsigned *d_worklist,
                                   unsigned *d_worklistCount,
                                   float *d_worklistDist,
                                   bool *d_worklistVisited,
                                   unsigned searchL) {
    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    unsigned worklistOffset = searchL * queryID;

    unsigned medoid = g_runtimeMedoid;
    GVEC *queryVec = d_queryVecs + D * queryID;
    GVEC *medoidVec = (GVEC*)(d_graph + graphEntrySize * medoid);

    if (tid == 0) {
        d_worklist[worklistOffset] = medoid;
        d_worklistCount[queryID] = 1;
        d_worklistVisited[worklistOffset] = true;

        float dist = 0;
        for (int i = 0; i < D; i++) {
            float diff = queryVec[i] - medoidVec[i];
            dist += diff * diff;
        }
        d_worklistDist[worklistOffset] = dist;
    }
}

// Host-side setter used by callers that want to override MEDOID at runtime.
void setRuntimeMedoid(unsigned medoid, cudaStream_t stream) {
    cudaMemcpyToSymbolAsync(g_runtimeMedoid, &medoid, sizeof(unsigned),
                            0, cudaMemcpyHostToDevice, stream);
}

/* Adds unvisited neighbours of nodes in d_parents to d_neighbors
 *
 * d_graph           - The graph
 * d_hasParent       - Whether there is a node to be visited (a parent) for a query
 * d_parents         - If the query has a parent, then the index of the parent
 * d_bloomFilters    - The bloom filters for checking if a node has been visited
 * d_neighbors       - Array for putting unvisited neighbors into
 * d_neighborsCount  - No. of unvisited neighbors for each query
 * d_visitedSet      - Array of visited points for each query
 * d_visitedSetCount - No. of visited points for each query
 */
__global__ void filterNeighbors(uint8_t *d_graph,
                                bool *d_hasParent,
                                unsigned *d_parents,
                                bool *d_bloomFilters,
                                unsigned *d_neighbors,
                                unsigned *d_neighborsCount,
                                unsigned *d_visitedSet,
                                unsigned *d_visitedSetCount,
                                unsigned int *d_deleted) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    if (!d_hasParent[queryID]) return;

    bool *bloomFilter = d_bloomFilters + (queryID * BF_MEMORY); // Get the bloom filter for this query
    unsigned parent = d_parents[queryID];                       // Get the parent for this query

    // Get the pointers to the degree and neighbors of the parent
    unsigned *degreePtr = (unsigned*)(d_graph + parent*graphEntrySize + D*sizeof(GVEC));
    unsigned *neighborPtr = degreePtr + 1;

    __syncthreads();

    // Initialize neighbor count to zero, reset hasParent
    if (tid == 0) {
        d_neighborsCount[queryID] = 0;
        d_hasParent[queryID] = false;

        unsigned visitedSetIdx = atomicAdd(&d_visitedSetCount[queryID], 1);
        if (visitedSetIdx < MAX_PARENTS_PERQUERY) {
            d_visitedSet[MAX_PARENTS_PERQUERY*queryID + visitedSetIdx] = parent;
        } else {
            printf("Limit hit for visited set: %d\n", queryID);
            atomicSub(&d_visitedSetCount[queryID], 1);
        }
    }

    __syncthreads();

    // Loop over each neighbor
    unsigned degree = *degreePtr;

    for (unsigned ii = tid; ii < degree; ii += blockDim.x) {
        unsigned neighbor = neighborPtr[ii];

        if (neighbor == queryID)
            continue;

        // FreshDiskANN: Skip deleted neighbors if deletion filtering is enabled
        if (d_deleted != nullptr && d_deleted[neighbor] != 0)
            continue;

        // Ensure the neighbor has not been visited yet
        if (!bf_check(bloomFilter, neighbor)) {
            bf_set(bloomFilter, neighbor);

            // Add the neighbor to d_neighbors
            unsigned neighborIdx = atomicAdd(&d_neighborsCount[queryID], 1);
            d_neighbors[(R+1)*queryID + neighborIdx] = neighbor;
        }
    }
}

/**
 * Version-based filterNeighbors kernel for concurrent safety
 *
 * Uses version numbers to ensure consistent reads of neighbor lists
 * even when other threads are modifying them concurrently.
 */
__global__ void filterNeighborsVersioned(uint8_t *d_graph,
                                          unsigned *d_versions,
                                          bool *d_hasParent,
                                          unsigned *d_parents,
                                          bool *d_bloomFilters,
                                          unsigned *d_neighbors,
                                          unsigned *d_neighborsCount,
                                          unsigned *d_visitedSet,
                                          unsigned *d_visitedSetCount,
                                          unsigned int *d_deleted) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    if (!d_hasParent[queryID]) return;

    bool *bloomFilter = d_bloomFilters + (queryID * BF_MEMORY);
    unsigned parent = d_parents[queryID];

    // Shared memory for neighbor snapshot
    __shared__ unsigned localNeighbors[R];
    __shared__ unsigned localDegree;

    __syncthreads();

    // Thread 0 takes the snapshot with version check
    if (tid == 0) {
        d_neighborsCount[queryID] = 0;
        d_hasParent[queryID] = false;

        unsigned visitedSetIdx = atomicAdd(&d_visitedSetCount[queryID], 1);
        if (visitedSetIdx < MAX_PARENTS_PERQUERY) {
            d_visitedSet[MAX_PARENTS_PERQUERY*queryID + visitedSetIdx] = parent;
        } else {
            printf("Limit hit for visited set: %d\n", queryID);
            atomicSub(&d_visitedSetCount[queryID], 1);
        }

        // Version-based read with retry
        unsigned retries = 0;
        const unsigned maxRetries = 5;

        while (retries < maxRetries) {
            unsigned v1 = d_versions[parent];
            if (v1 & 1) {
                // Write in progress, small backoff
                retries++;
                continue;
            }
            __threadfence();

            // Read degree and neighbors
            unsigned *degreePtr = (unsigned*)(d_graph + parent*graphEntrySize + D*sizeof(GVEC));
            unsigned *neighborPtr = degreePtr + 1;

            localDegree = *degreePtr;
            unsigned count = min(localDegree, (unsigned)R);

            for (unsigned i = 0; i < count; i++) {
                localNeighbors[i] = neighborPtr[i];
            }

            __threadfence();
            unsigned v2 = d_versions[parent];

            if (v1 == v2) {
                break;  // Consistent read
            }
            retries++;
        }

        // If max retries hit, use last read (bounded staleness)
        if (retries >= maxRetries) {
            unsigned *degreePtr = (unsigned*)(d_graph + parent*graphEntrySize + D*sizeof(GVEC));
            unsigned *neighborPtr = degreePtr + 1;
            localDegree = *degreePtr;
            for (unsigned i = 0; i < min(localDegree, (unsigned)R); i++) {
                localNeighbors[i] = neighborPtr[i];
            }
        }
    }

    __syncthreads();

    // All threads process from the snapshot
    unsigned degree = localDegree;

    for (unsigned ii = tid; ii < degree; ii += blockDim.x) {
        unsigned neighbor = localNeighbors[ii];

        if (neighbor == queryID)
            continue;

        // Skip deleted neighbors
        if (d_deleted != nullptr && d_deleted[neighbor] != 0)
            continue;

        // Ensure the neighbor has not been visited yet
        if (!bf_check(bloomFilter, neighbor)) {
            bf_set(bloomFilter, neighbor);

            // Add the neighbor to d_neighbors
            unsigned neighborIdx = atomicAdd(&d_neighborsCount[queryID], 1);
            d_neighbors[(R+1)*queryID + neighborIdx] = neighbor;
        }
    }
}


// ============================================================================
// filterNeighborsLocked — design-compliant adjacency read under per-node lock.
//
// Used by INSERT greedy. Per the design:
//   "Before reading the adjacency list of a node, use the lock."
//
// Thread 0 of each block acquires the binary lock on `parent` (atomicCAS),
// snapshots the degree + neighbor list into shared memory, releases the lock,
// then all threads process the snapshot. This is the lock-based equivalent of
// filterNeighborsVersioned's seqlock-style version check.
// ============================================================================
// uint8 byte-CAS for the per-node binary lock (CUDA atomicCAS is 32-bit so
// we operate on the containing word; see fda_locks.cuh for the same pattern).
__device__ __forceinline__
uint8_t bn_atomicCAS_u8(uint8_t* p, uint8_t cmp, uint8_t val) {
    uintptr_t addr   = (uintptr_t)p;
    uint32_t* word_p = (uint32_t*)(addr & ~uintptr_t(3));
    unsigned shift   = ((unsigned)(addr & 3)) << 3;
    uint32_t mask    = 0xffu << shift;
    // Force a fresh load each call — without this the compiler can hoist
    // *word_p out of an outer spinlock loop and the spinner reads stale
    // "lock=1" forever even after the holder releases.
    uint32_t old = atomicAdd(word_p, 0u);
    while (true) {
        uint8_t  old_byte = (uint8_t)((old >> shift) & 0xffu);
        if (old_byte != cmp) return old_byte;
        uint32_t new_word = (old & ~mask) | ((uint32_t)val << shift);
        uint32_t prev = atomicCAS(word_p, old, new_word);
        if (prev == old) return cmp;
        old = prev;
    }
}
__device__ __forceinline__
void bn_atomicExch_u8(uint8_t* p, uint8_t val) {
    uintptr_t addr   = (uintptr_t)p;
    uint32_t* word_p = (uint32_t*)(addr & ~uintptr_t(3));
    unsigned shift   = ((unsigned)(addr & 3)) << 3;
    uint32_t mask    = 0xffu << shift;
    uint32_t old = atomicAdd(word_p, 0u);   // force fresh load
    while (true) {
        uint32_t new_word = (old & ~mask) | ((uint32_t)val << shift);
        uint32_t prev = atomicCAS(word_p, old, new_word);
        if (prev == old) return;
        old = prev;
    }
}

// Lock-protected adjacency read (INSERT path; concurrent writers possible).
//
// Structure:
//   1. Tid 0: bookkeeping + acquire lock + read degree into shared.
//   2. All threads: parallel cooperative read of R neighbors into a shared
//      array (coalesced 256-byte transaction instead of 64 serial loads).
//   3. __syncthreads → all reads complete → tid 0 releases the lock.
//   4. All threads: fused bloom check + atomic emit, reading the snapshot
//      back from shared memory.
//
// Note: a pure BANG-style register-only version (no localNeighbors[] array,
// neighbor kept in a per-thread register across the lock release) was tried
// and benchmarked ~30% slower. Likely register pressure dropped occupancy
// enough to overcome the saved shared-mem hop. Keeping the shared-mem
// snapshot is the empirically faster shape for the locked variant.
__global__ void filterNeighborsLocked(uint8_t *d_graph,
                                      uint8_t *d_locks,
                                      bool *d_hasParent,
                                      unsigned *d_parents,
                                      bool *d_bloomFilters,
                                      unsigned *d_neighbors,
                                      unsigned *d_neighborsCount,
                                      unsigned *d_visitedSet,
                                      unsigned *d_visitedSetCount,
                                      unsigned int *d_deleted) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    if (!d_hasParent[queryID]) return;

    bool *bloomFilter = d_bloomFilters + (queryID * BF_MEMORY);
    unsigned parent = d_parents[queryID];

    __shared__ unsigned localNeighbors[R];
    __shared__ unsigned localDegree;

    unsigned *degreePtr  = (unsigned*)(d_graph + parent*graphEntrySize + D*sizeof(GVEC));
    unsigned *neighborPtr = degreePtr + 1;

    if (tid == 0) {
        d_neighborsCount[queryID] = 0;
        d_hasParent[queryID] = false;

        unsigned visitedSetIdx = atomicAdd(&d_visitedSetCount[queryID], 1);
        if (visitedSetIdx < MAX_PARENTS_PERQUERY) {
            d_visitedSet[MAX_PARENTS_PERQUERY*queryID + visitedSetIdx] = parent;
        } else {
            atomicSub(&d_visitedSetCount[queryID], 1);
        }

        while (bn_atomicCAS_u8(&d_locks[parent], 0u, 1u) != 0u) { /* spin */ }
        __threadfence();
        localDegree = *degreePtr;
    }
    __syncthreads();

    // Parallel cooperative read into shared mem (BANG-style stride pattern).
    unsigned count = min(localDegree, (unsigned)R);
    if (tid < count) {
        localNeighbors[tid] = neighborPtr[tid];
    }
    __syncthreads();

    if (tid == 0) {
        __threadfence();
        bn_atomicExch_u8(&d_locks[parent], 0u);
    }
    __syncthreads();

    unsigned degree = localDegree;
    for (unsigned ii = tid; ii < degree; ii += blockDim.x) {
        unsigned neighbor = localNeighbors[ii];
        if (neighbor == queryID) continue;
        if (d_deleted != nullptr && d_deleted[neighbor] != 0) continue;
        if (!bf_check(bloomFilter, neighbor)) {
            bf_set(bloomFilter, neighbor);
            unsigned neighborIdx = atomicAdd(&d_neighborsCount[queryID], 1);
            d_neighbors[(R+1)*queryID + neighborIdx] = neighbor;
        }
    }
}

// ============================================================================
// filterNeighborsNoSync — lock-free adjacency read for SEARCH.
//
// Per the design for Search Kernel:
//   "No need to take locks while reading the adjacency list."
//
// Thread 0 reads degree + neighbor list directly into shared memory, no
// lock/version check. Safe under the design assumption that searches don't
// race with concurrent writers on the same vertex (in our sequential
// workload this is trivially true).
// ============================================================================
__global__ void filterNeighborsNoSync(uint8_t *d_graph,
                                      bool *d_hasParent,
                                      unsigned *d_parents,
                                      bool *d_bloomFilters,
                                      unsigned *d_neighbors,
                                      unsigned *d_neighborsCount,
                                      unsigned *d_visitedSet,
                                      unsigned *d_visitedSetCount,
                                      unsigned int *d_deleted) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    if (!d_hasParent[queryID]) return;

    bool *bloomFilter = d_bloomFilters + (queryID * BF_MEMORY);
    unsigned parent = d_parents[queryID];

    __shared__ unsigned localNeighbors[R];
    __shared__ unsigned localDegree;

    __syncthreads();

    if (tid == 0) {
        d_neighborsCount[queryID] = 0;
        d_hasParent[queryID] = false;

        unsigned visitedSetIdx = atomicAdd(&d_visitedSetCount[queryID], 1);
        if (visitedSetIdx < MAX_PARENTS_PERQUERY) {
            d_visitedSet[MAX_PARENTS_PERQUERY*queryID + visitedSetIdx] = parent;
        } else {
            atomicSub(&d_visitedSetCount[queryID], 1);
        }

        // Direct read, no synchronization
        unsigned *degreePtr  = (unsigned*)(d_graph + parent*graphEntrySize + D*sizeof(GVEC));
        unsigned *neighborPtr = degreePtr + 1;
        localDegree = *degreePtr;
        unsigned count = min(localDegree, (unsigned)R);
        for (unsigned i = 0; i < count; i++) {
            localNeighbors[i] = neighborPtr[i];
        }
    }

    __syncthreads();

    unsigned degree = localDegree;
    for (unsigned ii = tid; ii < degree; ii += blockDim.x) {
        unsigned neighbor = localNeighbors[ii];
        if (neighbor == queryID) continue;
        if (d_deleted != nullptr && d_deleted[neighbor] != 0) continue;
        if (!bf_check(bloomFilter, neighbor)) {
            bf_set(bloomFilter, neighbor);
            unsigned neighborIdx = atomicAdd(&d_neighborsCount[queryID], 1);
            d_neighbors[(R+1)*queryID + neighborIdx] = neighbor;
        }
    }
}


__global__ void mergeIntoWorklist(unsigned *d_worklistCount,
                                  unsigned *d_worklist,
                                  float *d_worklistDist,
                                  bool *d_worklistVisited,

                                  unsigned *d_neighborsCount,
                                  unsigned *d_neighbors,
                                  float *d_neighborsDist,

                                  bool *d_hasParent,
                                  unsigned *d_parents,
                                  bool *d_nextIter,
                                  unsigned searchL) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    unsigned neighborsOffset = queryID * (R + 1);
    unsigned worklistOffset = queryID * searchL;

    unsigned numNeighbors = d_neighborsCount[queryID];
    unsigned worklistSize = d_worklistCount[queryID];
    unsigned newWorklistSize = min(numNeighbors + worklistSize, searchL);

    __shared__ unsigned sortedPositions[R + MAX_L + 1];

    unsigned id;
    float dist;
    bool visited;
    unsigned newPos = searchL;

    if (tid < worklistSize) {
        // First searchL threads find new position for worklist elements
        unsigned before = lowerBound(&d_neighborsDist[neighborsOffset], 0, numNeighbors, d_worklistDist[worklistOffset + tid]);
        id = d_worklist[worklistOffset + tid];
        dist = d_worklistDist[worklistOffset + tid];
        visited = d_worklistVisited[worklistOffset + tid];
        newPos = before + tid;
        sortedPositions[tid] = newPos;
    } else if (tid >= MAX_L && tid < MAX_L + numNeighbors) {
        // Next R + 1 threads find new position for neighbors
        unsigned idx = tid - MAX_L; // Index into the neighbors array
        unsigned before = upperBound(&d_worklistDist[worklistOffset], 0, worklistSize, d_neighborsDist[neighborsOffset + idx]);
        id = d_neighbors[neighborsOffset + idx];
        dist = d_neighborsDist[neighborsOffset + idx];
        visited = false;
        newPos = before + idx;
        sortedPositions[tid] = newPos;
    }
    
    __syncthreads();
    __threadfence_block();

    if (newPos < newWorklistSize) {
        d_worklist[worklistOffset + newPos] = id;
        d_worklistDist[worklistOffset + newPos] = dist;
        d_worklistVisited[worklistOffset + newPos] = visited;
    }

    __syncthreads();
    __threadfence_block();
    
    if (tid == 0) {
        d_worklistCount[queryID] = newWorklistSize;
        
        for (unsigned ii = 0; ii < newWorklistSize; ii++) {
            // Find the closest unvisited node, set it as the parent for the next iteration, and mark it as visited.
            if (!d_worklistVisited[worklistOffset + ii]) {
                *d_nextIter = true; 
                d_hasParent[queryID] = true;
                d_parents[queryID] = d_worklist[worklistOffset + ii];
                d_worklistVisited[worklistOffset + ii] = true;

                break;
            }
        }
    }   
}


// Performs greedy search and returns the visited sets
void greedySearch(uint8_t *d_graph,
                  GVEC *d_queryVecs,
                  unsigned *d_visitedSets,
                  unsigned *d_visitedSetCount,
                  unsigned batchStart,
                  unsigned batchSize,
                  unsigned searchL,
                  unsigned int *d_deleted) {
 
    bool *d_hasParent;
    unsigned *d_parents;
    bool *d_bloomFilters;

    gpuErrchk(cudaMalloc(&d_hasParent, batchSize * sizeof(bool)));
    gpuErrchk(cudaMalloc(&d_parents, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_bloomFilters, batchSize * BF_MEMORY * sizeof(bool)));
    gpuErrchk(cudaMemset(d_bloomFilters, 0, batchSize * BF_MEMORY * sizeof(bool)));

    unsigned *d_neighbors;
    unsigned *d_neighborsCount;
    float *d_neighborDists;
    unsigned *d_neighborsAux;
    float *d_neighborDistsAux;

    gpuErrchk(cudaMalloc(&d_neighbors, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborsCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborDists, batchSize * (R+1) * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_neighborsAux, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborDistsAux, batchSize * (R+1) * sizeof(float)));
    gpuErrchk(cudaMemset(d_neighbors, 0, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMemset(d_neighborsCount, 0, batchSize * sizeof(unsigned)));
    
    unsigned *d_worklist;
    unsigned *d_worklistCount;
    float *d_worklistDist;
    bool *d_worklistVisited;

    gpuErrchk(cudaMalloc(&d_worklist, batchSize * MAX_L * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_worklistCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_worklistDist, batchSize * MAX_L * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_worklistVisited, batchSize * MAX_L * sizeof(bool)));
    gpuErrchk(cudaMemset(d_worklistCount, 0, batchSize * sizeof(unsigned)));    

    bool nextIter;
    bool *d_nextIter;

    gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));

    initializeParents<<<batchSize, 1>>>(d_hasParent, d_parents);
    initializeWorklist<<<batchSize, 1>>>(d_graph,
                                         d_queryVecs,
                                         d_worklist,
                                         d_worklistCount,
                                         d_worklistDist,
                                         d_worklistVisited,
                                         searchL);
    // cudaDeviceSynchronize();

    unsigned *neighbors = (unsigned*)malloc((R+1) * sizeof(unsigned));
    float *neighborsDist = (float*)malloc((R+1) * sizeof(float));
    unsigned *worklist = (unsigned*)malloc(MAX_L * sizeof(unsigned));
    
    int iter = 0;
    do {    
        iter++;
        gpuErrchk(cudaMemset(d_nextIter, false, sizeof(bool)));
        
        filterNeighbors<<<batchSize, R>>>(d_graph,
                                            d_hasParent,
                                            d_parents,
                                            d_bloomFilters,
                                            d_neighbors,
                                            d_neighborsCount,
                                            d_visitedSets,
                                            d_visitedSetCount,
                                            d_deleted);

        // gpuErrchk(cudaDeviceSynchronize);

        computeDists<<<batchSize, R*8>>>(d_graph,
                                           d_neighbors,
                                           d_neighborsCount,
                                           d_queryVecs,
                                           d_neighborDists,
                                           (R+1));
        // gpuErrchk(cudaDeviceSynchronize);

        sortByDistance<<<batchSize, R, R*sizeof(unsigned)>>>(d_neighbors,
                                                                     d_neighborsCount,
                                                                     d_neighborDists,
                                                                     d_neighborsAux,
                                                                     d_neighborDistsAux,
                                                                     R+1);
        // gpuErrchk(cudaDeviceSynchronize);
       
        mergeIntoWorklist<<<batchSize, R+MAX_L>>>(d_worklistCount,
                                                d_worklist,
                                                d_worklistDist,
                                                d_worklistVisited,

                                                d_neighborsCount,
                                                d_neighbors,
                                                d_neighborDists,

                                                d_hasParent,
                                                d_parents,
                                                d_nextIter,
                                                searchL);
        gpuErrchk(cudaMemcpy(&nextIter, d_nextIter, sizeof(bool), cudaMemcpyDeviceToHost));
    }  while (nextIter);
        
    gpuErrchk(cudaFree(d_hasParent));
    gpuErrchk(cudaFree(d_parents));
    gpuErrchk(cudaFree(d_bloomFilters));

    gpuErrchk(cudaFree(d_neighbors));
    gpuErrchk(cudaFree(d_neighborsCount));
    gpuErrchk(cudaFree(d_neighborDists));
    gpuErrchk(cudaFree(d_neighborsAux));
    gpuErrchk(cudaFree(d_neighborDistsAux));

    gpuErrchk(cudaFree(d_worklist));
    gpuErrchk(cudaFree(d_worklistCount));
    gpuErrchk(cudaFree(d_worklistDist));
    gpuErrchk(cudaFree(d_worklistVisited));
    
    gpuErrchk(cudaFree(d_nextIter));

    printf("Greedy search finished in %d iterations.\n", iter);
}

/**
 * Version-based greedy search for concurrent safety
 *
 * Uses version numbers to ensure consistent reads of neighbor lists
 * even when other threads are modifying them concurrently.
 */
void greedySearchVersioned(uint8_t *d_graph,
                           unsigned *d_versions,
                           GVEC *d_queryVecs,
                           unsigned *d_visitedSets,
                           unsigned *d_visitedSetCount,
                           unsigned batchStart,
                           unsigned batchSize,
                           unsigned searchL,
                           unsigned int *d_deleted) {

    bool *d_hasParent;
    unsigned *d_parents;
    bool *d_bloomFilters;

    gpuErrchk(cudaMalloc(&d_hasParent, batchSize * sizeof(bool)));
    gpuErrchk(cudaMalloc(&d_parents, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_bloomFilters, batchSize * BF_MEMORY * sizeof(bool)));
    gpuErrchk(cudaMemset(d_bloomFilters, 0, batchSize * BF_MEMORY * sizeof(bool)));

    unsigned *d_neighbors;
    unsigned *d_neighborsCount;
    float *d_neighborDists;
    unsigned *d_neighborsAux;
    float *d_neighborDistsAux;

    gpuErrchk(cudaMalloc(&d_neighbors, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborsCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborDists, batchSize * (R+1) * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_neighborsAux, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborDistsAux, batchSize * (R+1) * sizeof(float)));
    gpuErrchk(cudaMemset(d_neighbors, 0, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMemset(d_neighborsCount, 0, batchSize * sizeof(unsigned)));

    unsigned *d_worklist;
    unsigned *d_worklistCount;
    float *d_worklistDist;
    bool *d_worklistVisited;

    gpuErrchk(cudaMalloc(&d_worklist, batchSize * MAX_L * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_worklistCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_worklistDist, batchSize * MAX_L * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_worklistVisited, batchSize * MAX_L * sizeof(bool)));
    gpuErrchk(cudaMemset(d_worklistCount, 0, batchSize * sizeof(unsigned)));

    bool nextIter;
    bool *d_nextIter;

    gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));

    initializeParents<<<batchSize, 1>>>(d_hasParent, d_parents);
    initializeWorklist<<<batchSize, 1>>>(d_graph,
                                         d_queryVecs,
                                         d_worklist,
                                         d_worklistCount,
                                         d_worklistDist,
                                         d_worklistVisited,
                                         searchL);

    int iter = 0;
    do {
        iter++;
        gpuErrchk(cudaMemset(d_nextIter, false, sizeof(bool)));

        // Use versioned filterNeighbors for concurrent safety
        filterNeighborsVersioned<<<batchSize, R>>>(d_graph,
                                                    d_versions,
                                                    d_hasParent,
                                                    d_parents,
                                                    d_bloomFilters,
                                                    d_neighbors,
                                                    d_neighborsCount,
                                                    d_visitedSets,
                                                    d_visitedSetCount,
                                                    d_deleted);

        computeDists<<<batchSize, R*8>>>(d_graph,
                                           d_neighbors,
                                           d_neighborsCount,
                                           d_queryVecs,
                                           d_neighborDists,
                                           (R+1));

        sortByDistance<<<batchSize, R, R*sizeof(unsigned)>>>(d_neighbors,
                                                                     d_neighborsCount,
                                                                     d_neighborDists,
                                                                     d_neighborsAux,
                                                                     d_neighborDistsAux,
                                                                     R+1);

        mergeIntoWorklist<<<batchSize, R+MAX_L>>>(d_worklistCount,
                                                d_worklist,
                                                d_worklistDist,
                                                d_worklistVisited,

                                                d_neighborsCount,
                                                d_neighbors,
                                                d_neighborDists,

                                                d_hasParent,
                                                d_parents,
                                                d_nextIter,
                                                searchL);
        gpuErrchk(cudaMemcpy(&nextIter, d_nextIter, sizeof(bool), cudaMemcpyDeviceToHost));
    }  while (nextIter);

    gpuErrchk(cudaFree(d_hasParent));
    gpuErrchk(cudaFree(d_parents));
    gpuErrchk(cudaFree(d_bloomFilters));

    gpuErrchk(cudaFree(d_neighbors));
    gpuErrchk(cudaFree(d_neighborsCount));
    gpuErrchk(cudaFree(d_neighborDists));
    gpuErrchk(cudaFree(d_neighborsAux));
    gpuErrchk(cudaFree(d_neighborDistsAux));

    gpuErrchk(cudaFree(d_worklist));
    gpuErrchk(cudaFree(d_worklistCount));
    gpuErrchk(cudaFree(d_worklistDist));
    gpuErrchk(cudaFree(d_worklistVisited));

    gpuErrchk(cudaFree(d_nextIter));
}

// Allocate pre-allocated buffers for greedySearchVersioned
void allocateGreedySearchBuffers(GreedySearchBuffers* buffers, unsigned batchSize) {
    buffers->batchSize = batchSize;

    gpuErrchk(cudaMalloc(&buffers->d_hasParent, batchSize * sizeof(bool)));
    gpuErrchk(cudaMalloc(&buffers->d_parents, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_bloomFilters, batchSize * BF_MEMORY * sizeof(bool)));

    gpuErrchk(cudaMalloc(&buffers->d_neighbors, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_neighborsCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_neighborDists, batchSize * (R+1) * sizeof(float)));
    gpuErrchk(cudaMalloc(&buffers->d_neighborsAux, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_neighborDistsAux, batchSize * (R+1) * sizeof(float)));

    gpuErrchk(cudaMalloc(&buffers->d_worklist, batchSize * MAX_L * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_worklistCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_worklistDist, batchSize * MAX_L * sizeof(float)));
    gpuErrchk(cudaMalloc(&buffers->d_worklistVisited, batchSize * MAX_L * sizeof(bool)));

    gpuErrchk(cudaMalloc(&buffers->d_nextIter, sizeof(bool)));
    gpuErrchk(cudaMallocHost(&buffers->h_nextIter, sizeof(bool)));  // Pinned memory
}

// Free pre-allocated buffers
void freeGreedySearchBuffers(GreedySearchBuffers* buffers) {
    gpuErrchk(cudaFree(buffers->d_hasParent));
    gpuErrchk(cudaFree(buffers->d_parents));
    gpuErrchk(cudaFree(buffers->d_bloomFilters));

    gpuErrchk(cudaFree(buffers->d_neighbors));
    gpuErrchk(cudaFree(buffers->d_neighborsCount));
    gpuErrchk(cudaFree(buffers->d_neighborDists));
    gpuErrchk(cudaFree(buffers->d_neighborsAux));
    gpuErrchk(cudaFree(buffers->d_neighborDistsAux));

    gpuErrchk(cudaFree(buffers->d_worklist));
    gpuErrchk(cudaFree(buffers->d_worklistCount));
    gpuErrchk(cudaFree(buffers->d_worklistDist));
    gpuErrchk(cudaFree(buffers->d_worklistVisited));

    gpuErrchk(cudaFree(buffers->d_nextIter));
    gpuErrchk(cudaFreeHost(buffers->h_nextIter));  // Free pinned memory
}

// Version using pre-allocated buffers (much faster)
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
                                    cudaStream_t stream) {

    // Reset buffers for this query (async on stream)
    gpuErrchk(cudaMemsetAsync(buffers->d_bloomFilters, 0, batchSize * BF_MEMORY * sizeof(bool), stream));
    gpuErrchk(cudaMemsetAsync(buffers->d_neighbors, 0, batchSize * (R+1) * sizeof(unsigned), stream));
    gpuErrchk(cudaMemsetAsync(buffers->d_neighborsCount, 0, batchSize * sizeof(unsigned), stream));
    gpuErrchk(cudaMemsetAsync(buffers->d_worklistCount, 0, batchSize * sizeof(unsigned), stream));

    initializeParents<<<batchSize, 1, 0, stream>>>(buffers->d_hasParent, buffers->d_parents);
    initializeWorklist<<<batchSize, 1, 0, stream>>>(d_graph,
                                         d_queryVecs,
                                         buffers->d_worklist,
                                         buffers->d_worklistCount,
                                         buffers->d_worklistDist,
                                         buffers->d_worklistVisited,
                                         searchL);

    // Use pinned memory for truly async copy
    bool* h_nextIter = buffers->h_nextIter;
    *h_nextIter = true;

    int iter = 0;
    const int CHECK_INTERVAL = 10;  // Check termination every N iterations

    while (*h_nextIter && iter < searchL * 2) {  // Upper bound to prevent infinite loops
        // Run multiple iterations before checking termination
        for (int i = 0; i < CHECK_INTERVAL && iter < searchL * 2; i++) {
            iter++;
            gpuErrchk(cudaMemsetAsync(buffers->d_nextIter, false, sizeof(bool), stream));

            filterNeighborsVersioned<<<batchSize, R, 0, stream>>>(d_graph,
                                                        d_versions,
                                                        buffers->d_hasParent,
                                                        buffers->d_parents,
                                                        buffers->d_bloomFilters,
                                                        buffers->d_neighbors,
                                                        buffers->d_neighborsCount,
                                                        d_visitedSets,
                                                        d_visitedSetCount,
                                                        d_deleted);

            computeDists<<<batchSize, R*8, 0, stream>>>(d_graph,
                                               buffers->d_neighbors,
                                               buffers->d_neighborsCount,
                                               d_queryVecs,
                                               buffers->d_neighborDists,
                                               (R+1));

            sortByDistance<<<batchSize, R, R*sizeof(unsigned), stream>>>(buffers->d_neighbors,
                                                                         buffers->d_neighborsCount,
                                                                         buffers->d_neighborDists,
                                                                         buffers->d_neighborsAux,
                                                                         buffers->d_neighborDistsAux,
                                                                         R+1);

            mergeIntoWorklist<<<batchSize, R+MAX_L, 0, stream>>>(buffers->d_worklistCount,
                                                    buffers->d_worklist,
                                                    buffers->d_worklistDist,
                                                    buffers->d_worklistVisited,

                                                    buffers->d_neighborsCount,
                                                    buffers->d_neighbors,
                                                    buffers->d_neighborDists,

                                                    buffers->d_hasParent,
                                                    buffers->d_parents,
                                                    buffers->d_nextIter,
                                                    searchL);
        }

        // Check termination condition (only once per CHECK_INTERVAL iterations)
        gpuErrchk(cudaMemcpyAsync(h_nextIter, buffers->d_nextIter, sizeof(bool), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
    }

    // printf("Greedy search finished in %d iterations.\n", iter);
}

// ============================================================================
// greedySearchLockedPrealloc — INSERT path.
// Same as Versioned but uses filterNeighborsLocked (per-node binary lock
// around adjacency read).
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
                                 cudaStream_t stream) {

    gpuErrchk(cudaMemsetAsync(buffers->d_bloomFilters, 0, batchSize * BF_MEMORY * sizeof(bool), stream));
    gpuErrchk(cudaMemsetAsync(buffers->d_neighbors, 0, batchSize * (R+1) * sizeof(unsigned), stream));
    gpuErrchk(cudaMemsetAsync(buffers->d_neighborsCount, 0, batchSize * sizeof(unsigned), stream));
    gpuErrchk(cudaMemsetAsync(buffers->d_worklistCount, 0, batchSize * sizeof(unsigned), stream));

    initializeParents<<<batchSize, 1, 0, stream>>>(buffers->d_hasParent, buffers->d_parents);
    initializeWorklist<<<batchSize, 1, 0, stream>>>(d_graph,
                                         d_queryVecs,
                                         buffers->d_worklist,
                                         buffers->d_worklistCount,
                                         buffers->d_worklistDist,
                                         buffers->d_worklistVisited,
                                         searchL);

    bool* h_nextIter = buffers->h_nextIter;
    *h_nextIter = true;
    int iter = 0;
    const int CHECK_INTERVAL = 10;

    while (*h_nextIter && iter < searchL * 2) {
        for (int i = 0; i < CHECK_INTERVAL && iter < searchL * 2; i++) {
            iter++;
            gpuErrchk(cudaMemsetAsync(buffers->d_nextIter, false, sizeof(bool), stream));

            filterNeighborsLocked<<<batchSize, R, 0, stream>>>(d_graph,
                                                        d_locks,
                                                        buffers->d_hasParent,
                                                        buffers->d_parents,
                                                        buffers->d_bloomFilters,
                                                        buffers->d_neighbors,
                                                        buffers->d_neighborsCount,
                                                        d_visitedSets,
                                                        d_visitedSetCount,
                                                        d_deleted);

            computeDists<<<batchSize, R*8, 0, stream>>>(d_graph,
                                               buffers->d_neighbors,
                                               buffers->d_neighborsCount,
                                               d_queryVecs,
                                               buffers->d_neighborDists,
                                               (R+1));

            sortByDistance<<<batchSize, R, R*sizeof(unsigned), stream>>>(buffers->d_neighbors,
                                                                         buffers->d_neighborsCount,
                                                                         buffers->d_neighborDists,
                                                                         buffers->d_neighborsAux,
                                                                         buffers->d_neighborDistsAux,
                                                                         R+1);

            mergeIntoWorklist<<<batchSize, R+MAX_L, 0, stream>>>(buffers->d_worklistCount,
                                                    buffers->d_worklist,
                                                    buffers->d_worklistDist,
                                                    buffers->d_worklistVisited,
                                                    buffers->d_neighborsCount,
                                                    buffers->d_neighbors,
                                                    buffers->d_neighborDists,
                                                    buffers->d_hasParent,
                                                    buffers->d_parents,
                                                    buffers->d_nextIter,
                                                    searchL);
        }
        gpuErrchk(cudaMemcpyAsync(h_nextIter, buffers->d_nextIter, sizeof(bool), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
    }
}

// ============================================================================
// greedySearchNoSyncPrealloc — SEARCH path.
// Same as Versioned but uses filterNeighborsNoSync (direct read, no locks
// or version checks). Note: "No need to take locks while reading the
// adjacency list" for search.
// ============================================================================
void greedySearchNoSyncPrealloc(uint8_t *d_graph,
                                 GVEC *d_queryVecs,
                                 unsigned *d_visitedSets,
                                 unsigned *d_visitedSetCount,
                                 unsigned batchStart,
                                 unsigned batchSize,
                                 unsigned searchL,
                                 unsigned int *d_deleted,
                                 GreedySearchBuffers* buffers,
                                 cudaStream_t stream) {

    gpuErrchk(cudaMemsetAsync(buffers->d_bloomFilters, 0, batchSize * BF_MEMORY * sizeof(bool), stream));
    gpuErrchk(cudaMemsetAsync(buffers->d_neighbors, 0, batchSize * (R+1) * sizeof(unsigned), stream));
    gpuErrchk(cudaMemsetAsync(buffers->d_neighborsCount, 0, batchSize * sizeof(unsigned), stream));
    gpuErrchk(cudaMemsetAsync(buffers->d_worklistCount, 0, batchSize * sizeof(unsigned), stream));

    initializeParents<<<batchSize, 1, 0, stream>>>(buffers->d_hasParent, buffers->d_parents);
    initializeWorklist<<<batchSize, 1, 0, stream>>>(d_graph,
                                         d_queryVecs,
                                         buffers->d_worklist,
                                         buffers->d_worklistCount,
                                         buffers->d_worklistDist,
                                         buffers->d_worklistVisited,
                                         searchL);

    bool* h_nextIter = buffers->h_nextIter;
    *h_nextIter = true;
    int iter = 0;
    const int CHECK_INTERVAL = 10;

    while (*h_nextIter && iter < searchL * 2) {
        for (int i = 0; i < CHECK_INTERVAL && iter < searchL * 2; i++) {
            iter++;
            gpuErrchk(cudaMemsetAsync(buffers->d_nextIter, false, sizeof(bool), stream));

            filterNeighborsNoSync<<<batchSize, R, 0, stream>>>(d_graph,
                                                        buffers->d_hasParent,
                                                        buffers->d_parents,
                                                        buffers->d_bloomFilters,
                                                        buffers->d_neighbors,
                                                        buffers->d_neighborsCount,
                                                        d_visitedSets,
                                                        d_visitedSetCount,
                                                        d_deleted);

            computeDists<<<batchSize, R*8, 0, stream>>>(d_graph,
                                               buffers->d_neighbors,
                                               buffers->d_neighborsCount,
                                               d_queryVecs,
                                               buffers->d_neighborDists,
                                               (R+1));

            sortByDistance<<<batchSize, R, R*sizeof(unsigned), stream>>>(buffers->d_neighbors,
                                                                         buffers->d_neighborsCount,
                                                                         buffers->d_neighborDists,
                                                                         buffers->d_neighborsAux,
                                                                         buffers->d_neighborDistsAux,
                                                                         R+1);

            mergeIntoWorklist<<<batchSize, R+MAX_L, 0, stream>>>(buffers->d_worklistCount,
                                                    buffers->d_worklist,
                                                    buffers->d_worklistDist,
                                                    buffers->d_worklistVisited,
                                                    buffers->d_neighborsCount,
                                                    buffers->d_neighbors,
                                                    buffers->d_neighborDists,
                                                    buffers->d_hasParent,
                                                    buffers->d_parents,
                                                    buffers->d_nextIter,
                                                    searchL);
        }
        gpuErrchk(cudaMemcpyAsync(h_nextIter, buffers->d_nextIter, sizeof(bool), cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
    }
}

// ============================================================================
// FUSED GREEDY SEARCH (BANG-style single-kernel) — ported from Tree B
//
// The non-fused path above launches 4 kernels per iter + sync every 10 iters,
// adding 4–6× the per-query overhead vs a single kernel. The fused kernel runs
// the entire loop in one launch with the worklist in shared memory.
//
// API: greedySearchFusedPrealloc — caller passes pre-allocated buffers.
// Output: d_visitedSets (parents explored), d_visitedSetCount, d_visitedSetDists.
// MVCC reads d_visitedSets[0..k] after a small post-sort for top-k.
// ============================================================================

template<typename Idx, typename Vec, typename Dist>
__device__ void filterNeighborsNew(uint8_t *d_graph,
                                   Idx parent,
                                   bool *bloomFilter,
                                   Idx *neighborCount,
                                   Idx *neighborIdxs,
                                   unsigned *d_deleted) {
    unsigned tid = threadIdx.x;
    Idx *degreePtr   = (Idx*)(d_graph + parent*graphEntrySzT<Idx, Vec> + D*sizeof(Vec));
    Idx *neighborPtr = degreePtr + 1;
    unsigned degree  = *degreePtr;
    for (unsigned ii = tid; ii < degree; ii += blockDim.x) {
        Idx neighbor = neighborPtr[ii];
        // Skip tombstoned vectors so they never enter the worklist.
        // Without this, deleted IDs can crowd out live candidates at high
        // delete churn and recall drops (~10pp on v10 sliding_window).
        if (d_deleted != nullptr && d_deleted[neighbor] != 0) continue;
        if (!bf_check(bloomFilter, neighbor)) {
            bf_set(bloomFilter, neighbor);
            Idx oldCount = atomicAdd(neighborCount, 1ULL);
            if (oldCount < R) neighborIdxs[oldCount] = neighbor;
        }
    }
}

template<typename Idx, typename Vec, typename Dist>
__device__ void computeDistsNew(uint8_t *d_graph,
                                Vec *queryVec,
                                Idx neighborCount,
                                Idx *neighborIdxs,
                                Dist *neighborDists) {
    unsigned tid = threadIdx.x;
    for (unsigned i = tid; i < neighborCount; i += blockDim.x) neighborDists[i] = 0;
    __syncthreads();
    unsigned warp = tid >> 5;
    unsigned lane = tid & 31;
    unsigned warpsPerBlock = blockDim.x >> 5;
    for (Idx j = warp; j < neighborCount; j += warpsPerBlock) {
        Idx neighbor = neighborIdxs[j];
        const Vec *nbrVec = reinterpret_cast<const Vec*>(d_graph + graphEntrySzT<Idx, Vec> * neighbor);
        Dist sum = 0;
        for (unsigned i = lane; i < D; i += 32) {
            Dist diff = Dist(nbrVec[i]) - Dist(queryVec[i]);
            sum += diff * diff;
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (lane == 0) neighborDists[j] = sum;
    }
    __syncthreads();
}

#ifndef FUSED_HELPERS_DEFINED
#define FUSED_HELPERS_DEFINED
__device__ constexpr int log2_floor(unsigned x) {
    return x <= 1 ? 0 : 1 + log2_floor(x >> 1);
}
template<int Size, typename Idx, typename Dist>
__device__ __forceinline__ void bitonicSortFused(unsigned length,
                                                 Idx *shm_idx,
                                                 Dist *shm_dist) {
    unsigned tid = threadIdx.x;
    unsigned blockSize = blockDim.x;
    for (unsigned i = length + tid; i < Size; i += blockSize) {
        shm_dist[i] = 3.402823466e+38f;
        shm_idx[i]  = 0xFFFFFFFF;
    }
    __syncthreads();
    unsigned num_stages = log2_floor(Size);
    #pragma unroll
    for (int sort_len = 0; sort_len < (int)num_stages; sort_len++) {
        for (int bitonic_len = sort_len; bitonic_len >= 0; bitonic_len--) {
            int bitonic_len_val  = 1 << bitonic_len;
            int bitonic_len_mask = ~(bitonic_len_val - 1);
            for (unsigned t = tid; t < Size / 2; t += blockSize) {
                int ind     = t + (t & bitonic_len_mask);
                int partner = ind + bitonic_len_val;
                bool descending = (t >> sort_len) & 0x1;
                bool shouldSwap = (shm_dist[ind] > shm_dist[partner]) != descending;
                if (shouldSwap) {
                    Dist tmpD = shm_dist[ind];
                    shm_dist[ind]     = shm_dist[partner];
                    shm_dist[partner] = tmpD;
                    Idx tmpI = shm_idx[ind];
                    shm_idx[ind]      = shm_idx[partner];
                    shm_idx[partner]  = tmpI;
                }
            }
            __syncthreads();
        }
    }
}
#endif // FUSED_HELPERS_DEFINED

template<typename Idx, typename Vec, typename Dist>
__device__ void mergeIntoWorklistNew(Idx *worklistCount,
                                     Idx *worklistIdxs,
                                     Dist *worklistDists,
                                     bool *worklistVisited,
                                     Idx *worklistIdxsAux,
                                     Dist *worklistDistsAux,
                                     bool *worklistVisitedAux,
                                     Idx neighborsCount,
                                     Idx *neighborIdxs,
                                     Dist *neighborDists,
                                     bool *hasParent,
                                     Idx *parent,
                                     Dist *parentDist) {
    unsigned tid = threadIdx.x;
    Idx worklistSize = *worklistCount;
    Idx numNeighbors = neighborsCount;
    Idx newWorklistSize = min(numNeighbors + worklistSize, (unsigned int)L);

    for (int ii = tid; ii < L + R; ii += blockDim.x) {
        if (ii < (int)worklistSize) {
            unsigned before = lowerBound(&neighborDists[0], 0, numNeighbors, worklistDists[ii]);
            unsigned newPos = before + ii;
            if (newPos < newWorklistSize) {
                worklistIdxsAux[newPos]     = worklistIdxs[ii];
                worklistDistsAux[newPos]    = worklistDists[ii];
                worklistVisitedAux[newPos]  = worklistVisited[ii];
            }
        } else if (ii >= L && ii < L + (int)numNeighbors) {
            unsigned idx = ii - L;
            unsigned before = upperBound(&worklistDists[0], 0, worklistSize, neighborDists[idx]);
            unsigned newPos = before + idx;
            if (newPos < newWorklistSize) {
                worklistIdxsAux[newPos]     = neighborIdxs[idx];
                worklistDistsAux[newPos]    = neighborDists[idx];
                worklistVisitedAux[newPos]  = false;
            }
        }
    }
    __syncthreads();
    for (int ii = tid; ii < (int)newWorklistSize; ii += blockDim.x) {
        worklistIdxs[ii]    = worklistIdxsAux[ii];
        worklistDists[ii]   = worklistDistsAux[ii];
        worklistVisited[ii] = worklistVisitedAux[ii];
    }
    __syncthreads();
    if (tid == 0) {
        *worklistCount = newWorklistSize;
        *hasParent = false;
        for (unsigned ii = 0; ii < newWorklistSize; ii++) {
            if (!worklistVisited[ii]) {
                worklistVisited[ii] = true;
                *hasParent = true;
                *parent     = worklistIdxs[ii];
                *parentDist = worklistDists[ii];
                break;
            }
        }
    }
}

// Fused single-kernel greedy search. Each block = one query.
// Output: d_visitedSetIdxs/d_visitedSetCount (parents explored, capped by
// MAX_PARENTS_PERQUERY), d_visitedSetDists (distance of each parent).
// d_deleted: optional tombstone filter — skip recording deleted parents to
// the visited set (still expand their neighbors so the search progresses).
template<typename Idx, typename Vec, typename Dist>
__global__ void greedySearchFusedKernel(uint8_t *d_graph,
                                        Vec    *d_queryVecs,
                                        bool   *d_bloomFilters,
                                        Idx    *d_visitedSetCount,
                                        Idx    *d_visitedSetIdxs,
                                        Dist   *d_visitedSetDists,
                                        unsigned *d_deleted,
                                        // Worklist output (sorted top-searchL by distance) so
                                        // MVCC's L0 augment can read it. Stride = searchL per
                                        // query — matches the existing non-fused convention
                                        // (d_worklist[q*searchL + i]).
                                        Idx    *d_worklistOut,
                                        Dist   *d_worklistDistOut,
                                        Idx    *d_worklistCountOut,
                                        unsigned searchL)
{
    __shared__ bool   hasParent;
    __shared__ Idx    parent;
    __shared__ Dist   parentDist;
    __shared__ Vec    queryVec[D];

    __shared__ Idx    neighborCount;
    __shared__ Idx    neighborIdxs[R];
    __shared__ Dist   neighborDists[R];

    __shared__ Idx    worklistCount;
    __shared__ Idx    worklistIdxs[L];
    __shared__ Dist   worklistDists[L];
    __shared__ bool   worklistVisited[L];
    __shared__ Idx    worklistIdxsAux[L];
    __shared__ Dist   worklistDistsAux[L];
    __shared__ bool   worklistVisitedAux[L];

    unsigned queryID = blockIdx.x;
    unsigned tid     = threadIdx.x;
    bool    *bloomFilter = d_bloomFilters + queryID*BF_MEMORY;

    for (int ii = tid; ii < D; ii += blockDim.x)
        queryVec[ii] = d_queryVecs[D*queryID + ii];
    __syncthreads();

    if (tid == 0) {
        hasParent = true;
        neighborCount = 0;
        Dist dist = 0;
        Vec *medoidVec = (Vec*)(d_graph + MEDOID*graphEntrySzT<Idx, Vec>);
        for (int i = 0; i < D; i++) {
            Dist diff = Dist(queryVec[i]) - Dist(medoidVec[i]);
            dist += diff * diff;
        }
        parent     = MEDOID;
        parentDist = dist;
        worklistCount    = 1;
        worklistIdxs[0]  = MEDOID;
        worklistDists[0] = dist;
        worklistVisited[0] = true;
        d_visitedSetCount[queryID] = 0;
    }
    __syncthreads();

    while (hasParent) {
        __syncthreads();
        if (tid == 0) {
            // Record parent to visited set (unless tombstoned)
            bool skipRecord = (d_deleted != nullptr) && (d_deleted[parent] != 0);
            if (!skipRecord) {
                Idx oldCount = d_visitedSetCount[queryID];
                if (oldCount < MAX_PARENTS_PERQUERY) {
                    d_visitedSetCount[queryID]++;
                    d_visitedSetIdxs[MAX_PARENTS_PERQUERY*queryID + oldCount] = parent;
                    d_visitedSetDists[MAX_PARENTS_PERQUERY*queryID + oldCount] = parentDist;
                }
            }
            hasParent = false;
            neighborCount = 0;
        }
        __syncthreads();

        filterNeighborsNew<Idx, Vec, Dist>(d_graph, parent, bloomFilter,
                                            &neighborCount, neighborIdxs,
                                            d_deleted);
        __syncthreads();
        computeDistsNew<Idx, Vec, Dist>(d_graph, queryVec, neighborCount,
                                         neighborIdxs, neighborDists);
        __syncthreads();
        if (!neighborCount) continue;
        bitonicSortFused<R, Idx, Dist>(neighborCount, neighborIdxs, neighborDists);
        __syncthreads();
        mergeIntoWorklistNew<Idx, Vec, Dist>(&worklistCount,
                                              worklistIdxs, worklistDists, worklistVisited,
                                              worklistIdxsAux, worklistDistsAux, worklistVisitedAux,
                                              neighborCount, neighborIdxs, neighborDists,
                                              &hasParent, &parent, &parentDist);
        __syncthreads();
    }

    // Spill worklist (top-searchL sorted) to global so callers (MVCC L0
    // augment) can read it. Stride = searchL per query — matches the
    // existing non-fused convention. Positions [outCount..searchL) get
    // sentinels so L0 augment can't pick them up.
    Idx wlCount = worklistCount;
    unsigned outCount = (wlCount < searchL) ? wlCount : searchL;
    if (tid == 0) d_worklistCountOut[queryID] = outCount;
    for (unsigned ii = tid; ii < searchL; ii += blockDim.x) {
        if (ii < outCount) {
            d_worklistOut[(size_t)queryID * searchL + ii]     = worklistIdxs[ii];
            d_worklistDistOut[(size_t)queryID * searchL + ii] = worklistDists[ii];
        } else {
            d_worklistOut[(size_t)queryID * searchL + ii]     = 0xFFFFFFFFu;
            d_worklistDistOut[(size_t)queryID * searchL + ii] = 3.402823466e+38f;
        }
    }
}

// Host-side pre-allocated wrapper used by MVCC's query/insert paths.
// Caller passes d_visitedSets/d_visitedSetCount/d_visitedSetDists as output
// (sized [batchSize × MAX_PARENTS_PERQUERY] / [batchSize] / [bs × MAX_PARENTS]).
// Resets bloom filters via buffers->d_bloomFilters. searchL is informational
// (the kernel runs internally up to L=150 which dominates).
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
                                cudaStream_t stream)
{
    (void)searchL;
    gpuErrchk(cudaMemsetAsync(buffers->d_bloomFilters, 0,
                              (size_t)batchSize * BF_MEMORY * sizeof(bool), stream));
    gpuErrchk(cudaMemsetAsync(d_visitedSetCount, 0,
                              batchSize * sizeof(unsigned), stream));

    GVEC *queryVecs = d_queryVecs + (size_t)D * batchStart;

    greedySearchFusedKernel<unsigned, GVEC, float><<<batchSize, 64, 0, stream>>>(
        d_graph, queryVecs, buffers->d_bloomFilters,
        d_visitedSetCount, d_visitedSets, d_visitedSetDists,
        d_deleted,
        buffers->d_worklist, buffers->d_worklistDist, buffers->d_worklistCount,
        searchL);
}

// ============================================================================
// FUSED LOCKED greedy for INSERT path (clean rewrite, v2)
// ============================================================================
// One block per query, block size = 32 (single warp). Each block:
//   1. Loads query vector into shared memory
//   2. Initializes worklist with entryPoint + its distance
//   3. Loop until no unvisited entry in worklist OR iterCap reached:
//      a. Find best (lowest-dist) unvisited worklist entry → next parent
//      b. Mark parent visited; emit to output visited set
//      c. Thread 0 acquires d_locks[parent], snapshots adjacency, releases
//      d. Warp computes L2 distance for each neighbor
//      e. Thread 0 merges neighbors into worklist (sorted ascending by dist,
//         dedup against existing entries, drop if worklist full and entry
//         would go to the end)
//
// Invariants:
//   - worklist sorted ascending by dist at all times
//   - worklist_count ≤ searchL ≤ MAX_L
//   - lock acquire/release strictly balanced inside thread 0 critical section
//   - termination: parentIdx == -1 OR iter == iterCap (hard cap)
//
// Block size 32 was picked deliberately: single warp → __syncthreads() is
// essentially free, no cross-warp coordination, and 32 threads cooperatively
// covering D=128 (each does 4 dimensions) is a clean fit for L2 dist.
//
// Differences from the previous (now-removed) fused kernel:
//   - Insertion sort instead of bitonic (smaller search depth, easier to verify)
//   - bool[] visited array instead of bitmap (simpler shifts on insert)
//   - No global bloom filter (worklist scan + worklist-visited bit suffices
//     for dedup at searchL ≤ 200)
//   - No separate aux arrays for merge — in-place shift
//   - Hard iteration cap; never spins on a worklist state that doesn't shrink
// ============================================================================

__device__ __forceinline__
static uint8_t flk2_atomicCAS_u8(uint8_t* p, uint8_t cmp, uint8_t val) {
    uintptr_t addr   = (uintptr_t)p;
    uint32_t* word_p = (uint32_t*)(addr & ~uintptr_t(3));
    unsigned shift   = ((unsigned)(addr & 3)) << 3;
    uint32_t mask    = 0xffu << shift;
    uint32_t old = *word_p;
    while (true) {
        uint8_t  old_byte = (uint8_t)((old >> shift) & 0xffu);
        if (old_byte != cmp) return old_byte;
        uint32_t new_word = (old & ~mask) | ((uint32_t)val << shift);
        uint32_t prev = atomicCAS(word_p, old, new_word);
        if (prev == old) return cmp;
        old = prev;
    }
}

__device__ __forceinline__
static void flk2_atomicExch_u8(uint8_t* p, uint8_t val) {
    uintptr_t addr   = (uintptr_t)p;
    uint32_t* word_p = (uint32_t*)(addr & ~uintptr_t(3));
    unsigned shift   = ((unsigned)(addr & 3)) << 3;
    uint32_t mask    = 0xffu << shift;
    uint32_t old = *word_p;
    while (true) {
        uint32_t new_word = (old & ~mask) | ((uint32_t)val << shift);
        uint32_t prev = atomicCAS(word_p, old, new_word);
        if (prev == old) return;
        old = prev;
    }
}

// Block-cooperative L2² distance between two D-dim float vectors.
// Block is one warp (32 threads). Returns the sum in lane 0; result is
// broadcast via shared memory before the caller uses it.
template<typename TA, typename TB>
__device__ __forceinline__
static float flk2_l2dist(const TA* a, const TB* b, float* smem_broadcast) {
    int tid = threadIdx.x;
    float partial = 0.0f;
    #pragma unroll
    for (int i = tid; i < D; i += 32) {
        float diff = a[i] - b[i];
        partial += diff * diff;
    }
    // Warp shuffle reduce
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        partial += __shfl_down_sync(0xffffffffu, partial, off);
    if (tid == 0) smem_broadcast[0] = partial;
    __syncthreads();
    return smem_broadcast[0];
}

__global__ void greedySearchFusedLockedKernel(
    uint8_t*  d_graph,
    uint8_t*  d_locks,
    GVEC*    d_queryVecs,
    unsigned  bs,
    unsigned* d_visitedSets,        // [bs × MAX_PARENTS_PERQUERY]
    unsigned* d_visitedCounts,      // [bs]
    unsigned  entryPoint,
    unsigned  searchL,
    unsigned  iterCap)
{
    unsigned q = blockIdx.x;
    if (q >= bs) return;
    unsigned tid = threadIdx.x;

    // ---- Shared memory -----------------------------------------------------
    __shared__ float    queryVec[D];                  // 128 × 4 = 512 B
    __shared__ unsigned worklist_id[MAX_L];           // 200 × 4 = 800 B
    __shared__ float    worklist_dist[MAX_L];         // 200 × 4 = 800 B
    __shared__ bool     worklist_visited[MAX_L];      // 200 × 1 = 200 B
    __shared__ unsigned worklist_count;

    __shared__ unsigned adj_count;
    __shared__ unsigned adj_ids[R];                   // 64 × 4 = 256 B
    __shared__ float    adj_dists[R];                 // 64 × 4 = 256 B

    __shared__ int      parentIdx;                    // selected worklist slot
    __shared__ float    smem_dist;                    // L2 broadcast scratch
    // Total per-block shared mem ~3 KB. Plenty of headroom on A100 (48-96 KB).

    // ---- 1. Load query vector ---------------------------------------------
    const GVEC* qVecSrc = d_queryVecs + (size_t)q * D;
    for (int i = tid; i < D; i += 32) queryVec[i] = qVecSrc[i];
    __syncthreads();

    // ---- 2. Initialize worklist with entryPoint ---------------------------
    const GVEC* entryVec = (const GVEC*)(d_graph + (size_t)entryPoint * graphEntrySize);
    float entryDist = flk2_l2dist(queryVec, entryVec, &smem_dist);
    if (tid == 0) {
        worklist_id[0]      = entryPoint;
        worklist_dist[0]    = entryDist;
        worklist_visited[0] = false;
        worklist_count      = 1;
        d_visitedCounts[q]  = 0;
    }
    __syncthreads();

    // ---- 3. Greedy loop ----------------------------------------------------
    unsigned iter = 0;
    while (iter < iterCap) {
        iter++;

        // 3a. Pick lowest-dist unvisited entry as parent (worklist is sorted
        // ascending, so first unvisited entry is the best).
        if (tid == 0) {
            parentIdx = -1;
            for (unsigned i = 0; i < worklist_count; i++) {
                if (!worklist_visited[i]) { parentIdx = (int)i; break; }
            }
        }
        __syncthreads();

        if (parentIdx == -1) break;   // converged — no more unvisited

        unsigned parent = worklist_id[parentIdx];

        // 3b. Snapshot adjacency under lock (thread 0).
        if (tid == 0) {
            worklist_visited[parentIdx] = true;

            // Emit to output visited set (used by downstream RobustPrune).
            unsigned outIdx = d_visitedCounts[q];
            if (outIdx < MAX_PARENTS_PERQUERY) {
                d_visitedSets[(size_t)q * MAX_PARENTS_PERQUERY + outIdx] = parent;
                d_visitedCounts[q] = outIdx + 1;
            }

            // Fine-grained, short-term lock — acquire, snapshot, release.
            uint8_t* lock = &d_locks[parent];
            while (flk2_atomicCAS_u8(lock, 0u, 1u) != 0u) { /* spin */ }
            __threadfence();

            unsigned* degPtr = (unsigned*)(d_graph + (size_t)parent * graphEntrySize
                                           + D*sizeof(GVEC));
            unsigned* nbrPtr = degPtr + 1;
            unsigned deg = *degPtr;
            if (deg > R) deg = R;
            adj_count = deg;
            for (unsigned i = 0; i < deg; i++) adj_ids[i] = nbrPtr[i];

            __threadfence();
            flk2_atomicExch_u8(lock, 0u);
        }
        __syncthreads();

        unsigned nNbrs = adj_count;
        if (nNbrs == 0) continue;     // empty adjacency; pick next parent

        // 3c. Compute L2 distance to each neighbor (warp-cooperative).
        for (unsigned i = 0; i < nNbrs; i++) {
            unsigned nbr = adj_ids[i];
            const GVEC* nbrVec = (const GVEC*)(d_graph + (size_t)nbr * graphEntrySize);
            float d = flk2_l2dist(queryVec, nbrVec, &smem_dist);
            if (tid == 0) adj_dists[i] = d;
        }
        // (smem_dist + adj_dists already syncthreaded inside flk2_l2dist)

        // 3d. Merge sorted-by-dist neighbors into worklist (thread 0).
        // Worklist stays sorted ascending; dedup against existing IDs; drop if
        // worklist is full and the entry would go past the end.
        if (tid == 0) {
            for (unsigned i = 0; i < nNbrs; i++) {
                unsigned nbr     = adj_ids[i];
                float    nbrDist = adj_dists[i];

                // Dedup
                bool already = false;
                for (unsigned j = 0; j < worklist_count; j++) {
                    if (worklist_id[j] == nbr) { already = true; break; }
                }
                if (already) continue;

                // Insertion position
                unsigned pos = 0;
                while (pos < worklist_count && worklist_dist[pos] <= nbrDist) pos++;

                // Worklist full and entry would land at the end → drop it.
                if (worklist_count >= searchL && pos >= searchL) continue;

                // Shift right from `end` down to pos+1 to open `pos`.
                unsigned end = (worklist_count < searchL) ? worklist_count
                                                          : searchL - 1;
                for (unsigned j = end; j > pos; j--) {
                    worklist_id[j]      = worklist_id[j-1];
                    worklist_dist[j]    = worklist_dist[j-1];
                    worklist_visited[j] = worklist_visited[j-1];
                }

                worklist_id[pos]      = nbr;
                worklist_dist[pos]    = nbrDist;
                worklist_visited[pos] = false;
                if (worklist_count < searchL) worklist_count++;
            }
        }
        __syncthreads();
    }
}

// Host wrapper. Pre-allocated buffer slots that aren't used by v2:
// d_bloomFilters, d_worklist*, d_visitedSetDists. Kept in the signature so
// callers don't need to know which fields are touched. The unused buffers
// are simply ignored.
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
                                      cudaStream_t stream)
{
    (void)d_visitedSetDists;   // not used by v2
    (void)buffers;             // not used by v2

    gpuErrchk(cudaMemsetAsync(d_visitedSetCount, 0,
                              (size_t)batchSize * sizeof(unsigned), stream));

    GVEC* queryVecs = d_queryVecs + (size_t)D * batchStart;
    unsigned iterCap = searchL * 2;

    greedySearchFusedLockedKernel<<<batchSize, 32, 0, stream>>>(
        d_graph, d_locks, queryVecs, batchSize,
        d_visitedSets, d_visitedSetCount,
        entryPoint, searchL, iterCap);
    gpuErrchk(cudaGetLastError());
}
