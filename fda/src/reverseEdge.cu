#include <cuda/std/limits>

#ifndef VAMANA_H
#include "vamana.h"
#endif

__global__ void loadQueryVecs(uint8_t *d_graph,
                              GVEC *d_queryVecs) {
    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    GVEC *queryVec = (GVEC*)(d_graph + queryID*graphEntrySize);

    for (int i = tid; i < D; i += blockDim.x) {
        d_queryVecs[queryID*D + i] = queryVec[i];
    }
}


// Convert from the byte-array representation into the usual array-and-length representation
__global__ void parseReverseIndex(uint8_t *d_reverseEdgeIndex,
                                  unsigned *d_reverseEdges,
                                  unsigned *d_reverseEdgeCount,
                                  unsigned *degreeCounts) {

    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    unsigned *entryPtr = (unsigned*)(d_reverseEdgeIndex + queryID*reverseIndexEntrySize);
    unsigned numReverseEdges = *entryPtr;

    if (tid == 0) {
        d_reverseEdgeCount[queryID] = numReverseEdges;
        atomicAdd(&degreeCounts[numReverseEdges], 1);
    }


    for (unsigned ii = tid; ii < numReverseEdges; ii += blockDim.x) {
        d_reverseEdges[queryID*MAX_REVERSE_INDEX_ENTRIES + ii] = entryPtr[1 + ii];
    }
}

__global__ void getPrunableQueryIDs(unsigned *d_reverseEdgeCount, unsigned *d_queryIDs, unsigned *d_queryCount) {
    int tid = threadIdx.x;
    int lane = tid % 32;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned flag = (i < N) && (d_reverseEdgeCount[i] != 0);
    if (i >= N) return;

    // unsigned flag = (d_reverseEdgeCount[i] != 0);
    unsigned mask = __ballot_sync(0xffffffff, flag);
    int warpActive = __popc(mask);

    unsigned warpBase = 0;
    if (lane == 0) {
        warpBase = atomicAdd(d_queryCount, warpActive);
    }
    warpBase = __shfl_sync(0xffffffff, warpBase, 0);

    int posInWarp = __popc(mask & ((1u << lane) - 1));
    if (flag)
        d_queryIDs[warpBase + posInWarp] = i;
}

// Could be merged with mergeIntoVisitedSets
__global__ void mergeIntoReverseEdges(unsigned *d_reverseEdgeCount,
                                      unsigned *d_reverseEdges,
                                      float *d_reverseEdgeDists,
                                      unsigned *d_neighborsCount,
                                      unsigned *d_neighbors,
                                      float *d_neighborsDist) {
    
    unsigned queryID = blockIdx.x;
    unsigned tid = threadIdx.x;

    unsigned reverseEdgeOffset = queryID * MAX_REVERSE_INDEX_ENTRIES;
    unsigned neighborsOffset = queryID * (R + 1);

    unsigned numNeighbors = d_neighborsCount[queryID];
    unsigned reverseEdgeCount = d_reverseEdgeCount[queryID];

    unsigned newReverseEdgeCount = min(numNeighbors + reverseEdgeCount, MAX_REVERSE_INDEX_ENTRIES);

    unsigned id;
    float dist;
    unsigned newPos = MAX_REVERSE_INDEX_ENTRIES;

    if (tid < reverseEdgeCount) {
        unsigned before = lowerBound(&d_neighborsDist[neighborsOffset], 0, numNeighbors, d_reverseEdgeDists[reverseEdgeOffset + tid]);
        id = d_reverseEdges[reverseEdgeOffset + tid];
        dist = d_reverseEdgeDists[reverseEdgeOffset + tid];
        newPos = before + tid;
    } else if (tid >= MAX_REVERSE_INDEX_ENTRIES && tid < MAX_REVERSE_INDEX_ENTRIES + numNeighbors) {
        unsigned idx = tid - MAX_REVERSE_INDEX_ENTRIES;
        unsigned before = upperBound(&d_reverseEdgeDists[reverseEdgeOffset], 0, reverseEdgeCount, d_neighborsDist[neighborsOffset + idx]);
        id = d_neighbors[neighborsOffset + idx];
        dist = d_neighborsDist[neighborsOffset + idx];
        newPos = before + idx;
    }

    __syncthreads();

    if (newPos < newReverseEdgeCount) {
        d_reverseEdges[reverseEdgeOffset + newPos] = id;
        d_reverseEdgeDists[reverseEdgeOffset + newPos] = dist;
    }

    __syncthreads();

    if (tid == 0) {
        d_reverseEdgeCount[queryID] = newReverseEdgeCount;
    }
}

__global__ void pruneReverseEdges(uint8_t *d_graph,
                                  unsigned *d_queryIDs,   
                                  unsigned *d_reverseEdges,
                                  unsigned *d_reverseEdgeCount,
                                  float *d_reverseEdgeDists,
                                  NodeState *d_reverseEdgeStatus,
                                  
                                  GVEC *d_queryVecs,
                                  float alpha) {

    for (unsigned iter=1; ; iter++) {
        // unsigned bid = blockIdx.x;
        // unsigned queryID = d_queryIDs[bid];
        unsigned queryID = blockIdx.x;
        unsigned tid = threadIdx.x;

        unsigned numNodes = d_reverseEdgeCount[queryID];
        unsigned reverseEdgeOffset = queryID * MAX_REVERSE_INDEX_ENTRIES;

        unsigned *degreePtr = (unsigned*)(d_graph + queryID*graphEntrySize + D*sizeof(GVEC));
        unsigned *neighborPtr = degreePtr + 1;

        // printf("%d\n", queryID);


        if (tid == 0 && iter == 1) {
            // Mark all candidate nodes as INIT
            for (unsigned i = 0; i < numNodes; i++) {
                d_reverseEdgeStatus[reverseEdgeOffset + i] = INIT;
            }

            // Set degree to zero
            *degreePtr = 0;
        }
        
        __syncthreads();
        
        // Don't add more neighbors if we already have R neighbors
        if (*degreePtr >= R)
            return;

        __shared__ unsigned pStarShared[1];
        *pStarShared = cuda::std::numeric_limits<unsigned>::max();

        // Find p_star
        if (tid == 0) {
            for (unsigned i = 0; i < numNodes; i++) {
                // Find the closest 'INIT' node (p_star)
                if (d_reverseEdgeStatus[reverseEdgeOffset + i] != INIT) continue;
                
                *pStarShared = d_reverseEdges[reverseEdgeOffset + i];
                
                // Add an edge from the query to p_star
                // TODO: Are we sure that oldDegree is always less than R
                unsigned oldDegree = atomicAdd(degreePtr, 1);
                neighborPtr[oldDegree] = *pStarShared;
                
                if (oldDegree > R) printf("Backward degree exceeded R: %d\n", queryID);

                // Set it to neighbor
                d_reverseEdgeStatus[reverseEdgeOffset + i] = NEIGHBOR;
                // *d_nextIter = true;
                break;
            }
        }

        __syncthreads();

        unsigned pStar = *pStarShared;
        if (pStar == cuda::std::numeric_limits<unsigned>::max())
            return;


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
            if (d_reverseEdgeStatus[reverseEdgeOffset + ii] != INIT) continue;

            unsigned p = d_reverseEdges[reverseEdgeOffset + ii];
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
                float queryDist = d_reverseEdgeDists[reverseEdgeOffset + ii];
                if (partial * alpha <= queryDist) {
                    d_reverseEdgeStatus[reverseEdgeOffset + ii] = PRUNED;
                }
            }
        }
    }
}

void computeReverseEdges(uint8_t *d_graph,
                         uint8_t *d_reverseEdgeIndex,
                         float alpha) {

    // TODO: Replace queryVecs with something better. It seems like a good idea to do this pruning in batches
    GVEC *d_queryVecs;

    gpuErrchk(cudaMalloc(&d_queryVecs, N * D * sizeof(GVEC)));
  
    unsigned *d_reverseEdges;
    unsigned *d_reverseEdgeCount;
    float *d_reverseEdgeDists;
    unsigned *d_reverseEdgesAux;
    float *d_reverseEdgeDistsAux;
    NodeState *d_reverseEdgeStatus;
 
    gpuErrchk(cudaMalloc(&d_reverseEdges, N * MAX_REVERSE_INDEX_ENTRIES * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeCount, N * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeDists, N * MAX_REVERSE_INDEX_ENTRIES * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_reverseEdgesAux, N * MAX_REVERSE_INDEX_ENTRIES * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeDistsAux, N * MAX_REVERSE_INDEX_ENTRIES * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeStatus, N * MAX_REVERSE_INDEX_ENTRIES * sizeof(NodeState)));

    unsigned *d_neighbors;
    unsigned *d_neighborsCount;
    float *d_neighborDists;
    unsigned *d_neighborsAux;
    float *d_neighborDistsAux;

    gpuErrchk(cudaMalloc(&d_neighbors, N * (R + 1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborsCount, N * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborDists, N * (R + 1) * sizeof(float)));
    gpuErrchk(cudaMalloc(&d_neighborsAux, N * (R + 1) * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_neighborDistsAux, N * (R + 1) * sizeof(float)));
    
    // bool nextIter;
    // bool *d_nextIter;

    // gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));

    loadQueryVecs<<<N, D>>>(d_graph, d_queryVecs);

    unsigned *degreeSum;
    gpuErrchk(cudaMalloc(&degreeSum, (MAX_REVERSE_INDEX_ENTRIES + 1) * sizeof(unsigned)));
    gpuErrchk(cudaMemset(degreeSum, 0, (MAX_REVERSE_INDEX_ENTRIES + 1) * sizeof(unsigned)));

    parseReverseIndex<<<N, 1024>>>(d_reverseEdgeIndex,
                                    d_reverseEdges,
                                    d_reverseEdgeCount,
                                    degreeSum);        

    // unsigned h_degreeCounts[MAX_REVERSE_INDEX_ENTRIES + 1];
    // cudaMemcpy(&h_degreeCounts, degreeSum, (MAX_REVERSE_INDEX_ENTRIES + 1) * sizeof(unsigned), cudaMemcpyDeviceToHost);

    // for (int i = 0; i <= MAX_REVERSE_INDEX_ENTRIES; i++) {
    //     if (h_degreeCounts[i] != 0)
    //         printf("%d:%d ", i, h_degreeCounts[i]);
    // }
    // printf("\n");

    unsigned h_queryCount;
    unsigned *d_queryIDs, *d_queryCount;
    gpuErrchk(cudaMalloc(&d_queryIDs, N * sizeof(unsigned)));
    gpuErrchk(cudaMalloc(&d_queryCount, sizeof(unsigned)));
    const int numThreads = 32;
    getPrunableQueryIDs<<<(N + numThreads - 1) / numThreads, numThreads>>>(d_reverseEdgeCount, d_queryIDs, d_queryCount);
    cudaMemcpy(&h_queryCount, d_queryCount, sizeof(unsigned), cudaMemcpyDeviceToHost);

    // printf("%d\n", h_queryCount)

    // Can't use 8*MAX_REVERSE_INDEX_ENTRIES because it exceeds block size limit 
    computeDists<<<N, 1024>>>(d_graph,
                              d_reverseEdges,
                              d_reverseEdgeCount,
                              d_queryVecs,
                              d_reverseEdgeDists,
                              MAX_REVERSE_INDEX_ENTRIES);

    //sortByDistance<<<N, MAX_REVERSE_INDEX_ENTRIES,
    sortByDistance<<<N, 1024,
                     MAX_REVERSE_INDEX_ENTRIES*sizeof(unsigned)>>>(d_reverseEdges,
                                                                   d_reverseEdgeCount,
                                                                   d_reverseEdgeDists,
                                                                   d_reverseEdgesAux,
                                                                   d_reverseEdgeDistsAux,
                                                                   MAX_REVERSE_INDEX_ENTRIES);

    getNeighbors<<<N, R>>>(d_graph,
                           0,
                           d_neighbors,
                           d_neighborsCount);

    computeDists<<<N, R*8>>>(d_graph,
                             d_neighbors,
                             d_neighborsCount,
                             d_queryVecs,
                             d_neighborDists,
                             (R+1));

    sortByDistance<<<N, R+1, (R+1)*sizeof(unsigned)>>>(d_neighbors,
                                                       d_neighborsCount,
                                                       d_neighborDists,
                                                       d_neighborsAux,
                                                       d_neighborDistsAux,
                                                       R+1);

    mergeIntoReverseEdges<<<N, 1024>>>(d_reverseEdgeCount,
    //mergeIntoReverseEdges<<<N, R+MAX_REVERSE_INDEX_ENTRIES>>>(d_reverseEdgeCount,
                                                              d_reverseEdges,
                                                              d_reverseEdgeDists,
                                                              d_neighborsCount,
                                                              d_neighbors,
                                                              d_neighborDists);

    // unsigned iter = 0;


    h_queryCount = N;
    pruneReverseEdges<<<h_queryCount, 32>>>(d_graph,
                                  d_queryIDs,
                                  d_reverseEdges,
                                  d_reverseEdgeCount,
                                  d_reverseEdgeDists,
                                  d_reverseEdgeStatus,

                                  d_queryVecs,
                                  alpha);

    gpuErrchk(cudaFree(d_queryVecs));

    gpuErrchk(cudaFree(d_reverseEdges));
    gpuErrchk(cudaFree(d_reverseEdgeCount));
    gpuErrchk(cudaFree(d_reverseEdgeDists));
    gpuErrchk(cudaFree(d_reverseEdgesAux));
    gpuErrchk(cudaFree(d_reverseEdgeDistsAux));
    gpuErrchk(cudaFree(d_reverseEdgeStatus));

    gpuErrchk(cudaFree(d_neighbors));
    gpuErrchk(cudaFree(d_neighborsCount));
    gpuErrchk(cudaFree(d_neighborDists));
    gpuErrchk(cudaFree(d_neighborsAux));
    gpuErrchk(cudaFree(d_neighborDistsAux));

    // printf("Reverse edge pruning finished in %d iterations.\n", iter);
}
