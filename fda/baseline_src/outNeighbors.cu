#include <cuda/std/limits>

#ifndef VAMANA_H
#include "vamana.h"
#endif

__global__ void getNeighbors(uint8_t *d_graph,
                             unsigned batchStart,
                             unsigned *d_neighbors,
                             unsigned *d_neighborsCount) {

    unsigned queryID = blockIdx.x;
    unsigned extendedQueryID = batchStart + queryID;
    unsigned tid = threadIdx.x;
 
    unsigned *degreePtr = (unsigned*)(d_graph + extendedQueryID*graphEntrySize + D*sizeof(GVEC));
    unsigned *neighborPtr = degreePtr + 1;
    
    unsigned degree = *degreePtr;

    if (tid == 0) {
        d_neighborsCount[queryID] = degree;
    }

    // TODO: Test necessity
    __syncthreads();

    // Loop over each neighbor
    for (int ii = tid; ii < degree; ii += blockDim.x) {
        d_neighbors[(R+1)*queryID + ii] = neighborPtr[ii];
    }
}

// Could be merged with mergeIntoWorklist (with a dummy array for d_visited))
__global__ void mergeIntoVisitedSet(unsigned *d_visitedSetCount,
                                    unsigned *d_visitedSet,
                                    float *d_visitedSetDists,
                                    unsigned *d_neighborsCount,
                                    unsigned *d_neighbors,
                                    float *d_neighborsDist) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    unsigned visitedSetOffset = queryID * MAX_PARENTS_PERQUERY;
    unsigned neighborsOffset = queryID * (R + 1);

    unsigned numNeighbors = d_neighborsCount[queryID];
    unsigned visitedSetSize = d_visitedSetCount[queryID];

    unsigned newVisitedSetSize = min(numNeighbors + visitedSetSize, MAX_PARENTS_PERQUERY);

    unsigned id;
    float dist;
    unsigned newPos = MAX_PARENTS_PERQUERY;
    
    if (tid < visitedSetSize) {
        unsigned before = lowerBound(&d_neighborsDist[neighborsOffset], 0, numNeighbors, d_visitedSetDists[visitedSetOffset + tid]);
        id = d_visitedSet[visitedSetOffset + tid];
        dist = d_visitedSetDists[visitedSetOffset + tid];
        newPos = before + tid;
    } else if (tid >= MAX_PARENTS_PERQUERY && tid < MAX_PARENTS_PERQUERY + numNeighbors) {
        unsigned idx = tid - MAX_PARENTS_PERQUERY;
        unsigned before = upperBound(&d_visitedSetDists[visitedSetOffset], 0, visitedSetSize, d_neighborsDist[neighborsOffset + idx]);
        id = d_neighbors[neighborsOffset + idx];
        dist = d_neighborsDist[neighborsOffset + idx];
        newPos = before + idx;
    }

    __syncthreads();

    if (newPos < newVisitedSetSize) {
        d_visitedSet[visitedSetOffset + newPos] = id;
        d_visitedSetDists[visitedSetOffset + newPos] = dist;
    }

    __syncthreads();
    
    if (tid == 0) {
        d_visitedSetCount[queryID] = newVisitedSetSize;
    }
}

// Robust Prune
__global__ void pruneOutNeighbors(uint8_t *d_graph,
                                  unsigned batchStart,
                                  unsigned *d_visitedSet,
                                  unsigned *d_visitedSetCount,
                                  float *d_visitedSetDists,
                                  NodeState *d_visitedSetStatus,  

                                  GVEC *d_queryVecs,
                                  uint8_t *d_reverseEdgeIndex,
                                  float alpha) {

    for (unsigned iter = 1; ; iter++) {
        unsigned queryID = blockIdx.x;
        unsigned extendedQueryID = batchStart + queryID;
        unsigned tid = threadIdx.x;

        unsigned numNodes = d_visitedSetCount[queryID];
        unsigned visitedSetOffset = queryID * MAX_PARENTS_PERQUERY;

        unsigned *degreePtr = (unsigned*)(d_graph + extendedQueryID*graphEntrySize + D*sizeof(GVEC));
        unsigned *neighborPtr = degreePtr + 1;


        // Initialization
        if (tid == 0 && iter == 1) {
            // Mark all candidate nodes as INIT
            for (unsigned i = 0; i < numNodes; i++) {
                d_visitedSetStatus[visitedSetOffset + i] = INIT;
            }

            // Set degree to zero
            *degreePtr = 0;
        }

        __syncthreads();

        // Don't add more neighbors if we already have R neighbors
        if (*degreePtr >= R) return;

        __shared__ unsigned pStarShared[1];
        *pStarShared = cuda::std::numeric_limits<unsigned>::max();

        // Find p_star
        if (tid == 0) {
            for (unsigned i = 0; i < numNodes; i++) {
                // Find the closest 'INIT' node (p_star)
                if (d_visitedSetStatus[visitedSetOffset + i] != INIT) continue;
                
                *pStarShared = d_visitedSet[visitedSetOffset + i];
                
                // Add an edge from the query to p_star
                unsigned oldDegree = atomicAdd(degreePtr, 1);
                neighborPtr[oldDegree] = *pStarShared;

                // Set it to neighbor
                d_visitedSetStatus[visitedSetOffset + i] = NEIGHBOR;

                // We need to add a reverse edge from p_star to query.
                // Skip when d_reverseEdgeIndex is null — the caller is collecting
                // reverse edges via a separate mechanism (e.g. d_revEdgeSrc/Dst
                // pairs), so this dense per-vertex index isn't needed.
                if (d_reverseEdgeIndex != nullptr) {
                    unsigned *entryPtr = (unsigned*)&d_reverseEdgeIndex[*pStarShared * reverseIndexEntrySize];
                    unsigned oldLen = atomicAdd(entryPtr, 1);
                    if (oldLen < MAX_REVERSE_INDEX_ENTRIES) {
                        entryPtr[1 + oldLen] = extendedQueryID;
                    } else {
                        // printf("Reverse index limit hit: %d\n", queryID);
                        atomicSub(entryPtr, 1);
                    }
                }

                // *d_nextIter = true;
                break;
            }
        }

        __syncthreads();

        int pStar = *pStarShared;
        if (pStar == cuda::std::numeric_limits<unsigned>::max()) {
            return;
        }

        // Copy p_star into shared memory
        __shared__ float pStarVec[D];
        GVEC *vecPtr = (GVEC*)(d_graph + pStar*graphEntrySize); // Pointer to query vector 
        for (unsigned ii = tid; ii < D; ii += blockDim.x) {
            pStarVec[ii] = vecPtr[ii];
        }

        __syncthreads();


        unsigned laneId = threadIdx.x & 31;
        unsigned warpId = threadIdx.x >> 5;          // warp index within block
        unsigned warpsPerBlock = blockDim.x >> 5;

        for (unsigned ii = warpId; ii < numNodes; ii += warpsPerBlock) {
            if (d_visitedSetStatus[visitedSetOffset + ii] != INIT) continue;

            unsigned p = d_visitedSet[visitedSetOffset + ii];
            const GVEC *pVec = reinterpret_cast<const GVEC*>(d_graph + p*graphEntrySize);

            // cooperative distance computation
            float partial = 0.0f;
            for (unsigned j = laneId; j < D; j += 32) {
                float diff = pVec[j] - pStarVec[j];
                partial = fmaf(diff, diff, partial);
            }

            // warp reduce sum
            for (int offset = 16; offset > 0; offset >>= 1)
                partial += __shfl_down_sync(0xffffffff, partial, offset);

            if (laneId == 0) {
                float queryDist = d_visitedSetDists[visitedSetOffset + ii];
                if (partial * alpha <= queryDist) {
                    d_visitedSetStatus[visitedSetOffset + ii] = PRUNED;
                }
            }
        }
    }
}


void computeOutNeighbors(uint8_t *d_graph,
                         GVEC *d_queryVecs,
                         unsigned *d_visitedSets,
                         unsigned *d_visitedSetCount,
                         float alpha,
                         uint8_t *d_reverseEdgeIndex,
                         unsigned batchStart,
                         unsigned batchSize) {
        
    bool log = false;

    cudaStream_t stream = 0; // default stream
    GPUTimer gputimer(stream, !log);
    // printf("%d\n", batchSize);
                           
    float *d_visitedSetDists;
    unsigned *d_visitedSetAux;
    float *d_visitedSetDistsAux;
    NodeState *d_visitedSetStatus;

    gpuErrchk(cudaMalloc(&d_visitedSetDists, batchSize * MAX_PARENTS_PERQUERY * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_visitedSetAux, batchSize * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_visitedSetDistsAux, batchSize * MAX_PARENTS_PERQUERY * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_visitedSetStatus, batchSize * MAX_PARENTS_PERQUERY * sizeof(NodeState)));

    unsigned *d_neighbors;
    unsigned *d_neighborsCount;
    float *d_neighborsDists;
    unsigned *d_neighborsAux;
    float *d_neighborsDistsAux;

    gpuErrchk(cudaMalloc(&d_neighbors, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborsCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborsDists, batchSize * (R+1) * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_neighborsAux, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborsDistsAux, batchSize * (R+1) * sizeof(float)));

    // bool nextIter;
    // bool *d_nextIter;
    // gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));

    gputimer.Start();
    getNeighbors<<<batchSize, R>>>(d_graph,
                                     batchStart,
                                     d_neighbors,
                                     d_neighborsCount);
    gputimer.Stop();
    gpuErrchk(cudaDeviceSynchronize());
    // printf("getNeighbors GPU time: %f ms\n", gputimer.Elapsed());


    gputimer.Start();
    computeDists<<<batchSize, R*8>>>(d_graph,
                                       d_neighbors,
                                       d_neighborsCount,
                                       d_queryVecs,
                                       d_neighborsDists,
                                       (R+1));
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("computeDists GPU time: %f ms\n", gputimer.Elapsed());


    gputimer.Start();
    sortByDistance<<<batchSize, R+1, (R+1)*sizeof(unsigned)>>>(d_neighbors,
                                                                 d_neighborsCount,
                                                                 d_neighborsDists,
                                                                 d_neighborsAux,
                                                                 d_neighborsDistsAux,
                                                                 R+1);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("sortByDistance GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    computeDists<<<batchSize, 1024>>>(d_graph,
                                      d_visitedSets,
                                      d_visitedSetCount,
                                      d_queryVecs,
                                      d_visitedSetDists,
                                      MAX_PARENTS_PERQUERY);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("computeDists GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    sortByDistance<<<batchSize, MAX_PARENTS_PERQUERY,
                     MAX_PARENTS_PERQUERY*sizeof(unsigned)>>>(d_visitedSets,
                                                              d_visitedSetCount,
                                                              d_visitedSetDists,
                                                              d_visitedSetAux,
                                                              d_visitedSetDistsAux,
                                                              MAX_PARENTS_PERQUERY);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("sortByDistance GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    mergeIntoVisitedSet<<<batchSize, MAX_PARENTS_PERQUERY + R>>>(d_visitedSetCount,
                                                                 d_visitedSets,
                                                                 d_visitedSetDists,
                                                                 d_neighborsCount,
                                                                 d_neighbors,
                                                                 d_neighborsDists);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("mergeIntoVisitedSet GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    pruneOutNeighbors<<<batchSize, 32>>>(d_graph,
                                            batchStart,
                                            d_visitedSets,
                                            d_visitedSetCount,
                                            d_visitedSetDists,
                                            d_visitedSetStatus,
                                            
                                            d_queryVecs,
                                            d_reverseEdgeIndex,
                                            alpha);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("pruneOutNeighbors GPU time: %f ms\n", gputimer.Elapsed());

    gpuErrchk(cudaFree(d_visitedSetDists));
    gpuErrchk(cudaFree(d_visitedSetAux));
    gpuErrchk(cudaFree(d_visitedSetDistsAux));
    gpuErrchk(cudaFree(d_visitedSetStatus));

    gpuErrchk(cudaFree(d_neighbors));
    gpuErrchk(cudaFree(d_neighborsCount));
    gpuErrchk(cudaFree(d_neighborsDists));
    gpuErrchk(cudaFree(d_neighborsAux));
    gpuErrchk(cudaFree(d_neighborsDistsAux));
    // gpuErrchk(cudaFree(d_nextIter));

    // printf("Out neighbor pruning finished in %d iterations.\n", iter);
}

// ============== Pre-allocated OutNeighbors Functions ==============

void allocateOutNeighborsBuffers(OutNeighborsBuffers* buffers, unsigned batchSize) {
    buffers->batchSize = batchSize;

    gpuErrchk(cudaMalloc(&buffers->d_visitedSetDists, batchSize * MAX_PARENTS_PERQUERY * sizeof(float)));
    gpuErrchk(cudaMalloc(&buffers->d_visitedSetAux, batchSize * MAX_PARENTS_PERQUERY * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_visitedSetDistsAux, batchSize * MAX_PARENTS_PERQUERY * sizeof(float)));
    gpuErrchk(cudaMalloc(&buffers->d_visitedSetStatus, batchSize * MAX_PARENTS_PERQUERY * sizeof(NodeState)));

    gpuErrchk(cudaMalloc(&buffers->d_neighbors, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_neighborsCount, batchSize * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_neighborsDists, batchSize * (R+1) * sizeof(float)));
    gpuErrchk(cudaMalloc(&buffers->d_neighborsAux, batchSize * (R+1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&buffers->d_neighborsDistsAux, batchSize * (R+1) * sizeof(float)));
}

void freeOutNeighborsBuffers(OutNeighborsBuffers* buffers) {
    gpuErrchk(cudaFree(buffers->d_visitedSetDists));
    gpuErrchk(cudaFree(buffers->d_visitedSetAux));
    gpuErrchk(cudaFree(buffers->d_visitedSetDistsAux));
    gpuErrchk(cudaFree(buffers->d_visitedSetStatus));

    gpuErrchk(cudaFree(buffers->d_neighbors));
    gpuErrchk(cudaFree(buffers->d_neighborsCount));
    gpuErrchk(cudaFree(buffers->d_neighborsDists));
    gpuErrchk(cudaFree(buffers->d_neighborsAux));
    gpuErrchk(cudaFree(buffers->d_neighborsDistsAux));
}

void computeOutNeighborsPrealloc(uint8_t *d_graph,
                                  GVEC *d_queryVecs,
                                  unsigned *d_visitedSets,
                                  unsigned *d_visitedSetCount,
                                  float alpha,
                                  uint8_t *d_reverseEdgeIndex,
                                  unsigned batchStart,
                                  unsigned batchSize,
                                  OutNeighborsBuffers* buffers,
                                  cudaStream_t stream) {

    // Use pre-allocated buffers (no cudaMalloc overhead!)
    float *d_visitedSetDists = buffers->d_visitedSetDists;
    unsigned *d_visitedSetAux = buffers->d_visitedSetAux;
    float *d_visitedSetDistsAux = buffers->d_visitedSetDistsAux;
    NodeState *d_visitedSetStatus = buffers->d_visitedSetStatus;
    unsigned *d_neighbors = buffers->d_neighbors;
    unsigned *d_neighborsCount = buffers->d_neighborsCount;
    float *d_neighborsDists = buffers->d_neighborsDists;
    unsigned *d_neighborsAux = buffers->d_neighborsAux;
    float *d_neighborsDistsAux = buffers->d_neighborsDistsAux;

    getNeighbors<<<batchSize, R, 0, stream>>>(d_graph,
                                              batchStart,
                                              d_neighbors,
                                              d_neighborsCount);

    computeDists<<<batchSize, R*8, 0, stream>>>(d_graph,
                                                 d_neighbors,
                                                 d_neighborsCount,
                                                 d_queryVecs,
                                                 d_neighborsDists,
                                                 (R+1));

    sortByDistance<<<batchSize, R+1, (R+1)*sizeof(unsigned), stream>>>(d_neighbors,
                                                                        d_neighborsCount,
                                                                        d_neighborsDists,
                                                                        d_neighborsAux,
                                                                        d_neighborsDistsAux,
                                                                        R+1);

    computeDists<<<batchSize, 1024, 0, stream>>>(d_graph,
                                                  d_visitedSets,
                                                  d_visitedSetCount,
                                                  d_queryVecs,
                                                  d_visitedSetDists,
                                                  MAX_PARENTS_PERQUERY);

    sortByDistance<<<batchSize, MAX_PARENTS_PERQUERY,
                     MAX_PARENTS_PERQUERY*sizeof(unsigned), stream>>>(d_visitedSets,
                                                                       d_visitedSetCount,
                                                                       d_visitedSetDists,
                                                                       d_visitedSetAux,
                                                                       d_visitedSetDistsAux,
                                                                       MAX_PARENTS_PERQUERY);

    mergeIntoVisitedSet<<<batchSize, MAX_PARENTS_PERQUERY + R, 0, stream>>>(d_visitedSetCount,
                                                                             d_visitedSets,
                                                                             d_visitedSetDists,
                                                                             d_neighborsCount,
                                                                             d_neighbors,
                                                                             d_neighborsDists);

    pruneOutNeighbors<<<batchSize, MAX_PARENTS_PERQUERY, 0, stream>>>(d_graph,
                                                                       batchStart,
                                                                       d_visitedSets,
                                                                       d_visitedSetCount,
                                                                       d_visitedSetDists,
                                                                       d_visitedSetStatus,
                                                                       d_queryVecs,
                                                                       d_reverseEdgeIndex,
                                                                       alpha);
}
