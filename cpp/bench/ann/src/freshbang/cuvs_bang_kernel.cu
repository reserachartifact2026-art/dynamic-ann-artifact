/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../cuvs/cuvs_cagra_wrapper.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <raft/util/cudart_utils.hpp>

#include <cstdio>
#include <cstdint>
#include <memory>



namespace cuvs::bench::detail {
  typedef struct _DeleteKernelParams {
    uint64_t* row_num; // rows to be deleted
    size_t num_ids;
    uint32_t* graph;
    int64_t graph_rows;
    int64_t graph_cols;
    uint8_t* deleted_rows;
  } DeleteKernelParams;

    typedef struct _InsertKernelParams {
    uint64_t* row_num; // rows to be inserted
    size_t num_ids; // number of rows to be inserted i.e size of row_num
    uint32_t* graph;
    int64_t graph_rows;
    int64_t graph_cols;
    uint8_t* deleted_rows;
    float* d_distance_threshold1; // to be used for insertions
    float* d_distance_threshold2; // to be used for insertions  
  } InsertKernelParams;

__device__ __forceinline__ bool is_row_in_insert_batch(const InsertKernelParams& params,
                                                        uint64_t row)
{
  for (size_t i = 0; i < params.num_ids; ++i) {
    if (params.row_num[i] == row) { return true; }
  }
  return false;
}

__device__ __forceinline__ bool is_invalid_or_deleted_neighbor(const InsertKernelParams& params,
                                                               uint32_t neighbor)
{
  if (static_cast<int64_t>(neighbor) >= params.graph_rows) { return true; }
  return params.deleted_rows[neighbor] == 1;
}

template<typename T>
__global__ void cuvs_bang_insert_kernel1(InsertKernelParams params,
                                         T* d_insert_vectors,
                                         T* d_dataset_vectors,
                                         int vector_dim,
                                         int recall_at_k,
                                         const algo_base::index_type* d_search_neighbors,
                                         const float* d_search_distances);

template<typename T>
__global__ void cuvs_bang_insert_kernel2(InsertKernelParams params);

 template<typename T>
__global__ void cuvs_bang_insert_kernel1(InsertKernelParams params, 
                                        T* d_insert_vectors,
                                        T* d_dataset_vectors, 
                                        int vector_dim,
                                        int recall_at_k,
                                        const algo_base::index_type* d_search_neighbors,
                                        const float* d_search_distances
                                      )
{
// for debugging, in thread 0/block 0 print the graph dimensions, recall_at_k, search neighbors and search_distances for the first row to be inserted 
#if 0
if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[bang_insert_kernel1] graph_rows=%lld graph_cols=%lld num_ids=%llu recall_at_k=%d first_row=%llu\n",
           (long long)params.graph_rows, (long long)params.graph_cols, (unsigned long long)params.num_ids, recall_at_k,
           (unsigned long long)params.row_num[0]);
    printf("[bang_insert_kernel1] search_neighbors for first row: ");
    for (int i = 0; i < recall_at_k; i++) {
        printf("%lld ", (long long)d_search_neighbors[i]);
    }
    printf("\n");
    printf("[bang_insert_kernel1] search_distances for first row: ");
    for (int i = 0; i < recall_at_k; i++) {
        printf("%f ", d_search_distances[i]);
    }
    printf("\n"); 
}
#endif

  auto row_idx = blockIdx.x;
  auto col_idx = threadIdx.x;
  if (row_idx >= params.num_ids || col_idx >= params.graph_cols) { return; }

  // identify the current row to be processed by this thread
  auto cur_row = params.row_num[row_idx];

  if (cur_row >= static_cast<uint64_t>(params.graph_rows)) { return; }
  if (col_idx >= static_cast<uint64_t>(params.graph_cols)) { return; }
  // Mark this row as live (not deleted) now that it is being inserted.
  // Only one thread per block needs to do this write.
 

  // First Step: copy d_insert_vectors to the appropriate location in the dataset 
  uint32_t dims_per_thread =  (vector_dim + blockDim.x - 1) / blockDim.x; // ceiling division to cover all dimensions

  // Each thread will copy dims_per_thread dimensions of the vector. Calculate the start and end dimension index for this thread.
  for (uint32_t cur_dim = threadIdx.x * dims_per_thread; cur_dim < vector_dim && cur_dim < (threadIdx.x + 1) * dims_per_thread; cur_dim++) {
    d_dataset_vectors[cur_row * vector_dim + cur_dim] = d_insert_vectors[row_idx * vector_dim + cur_dim];
  }

  // Second Step: Update the adjacency list for the current row with the search neighbors returned by the search kernel and mark the distance thresholds for this row based on the search distances returned by the search kernel.
  {
    auto nbr_val = d_search_neighbors[(row_idx * recall_at_k) + col_idx];
    // Only write valid neighbor IDs; skip CAGRA sentinels (UINT32_MAX) and out-of-range values.
    if (nbr_val >= 0 && nbr_val < params.graph_rows) {
      params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + col_idx] = 
       static_cast<uint32_t>(nbr_val);
    }
    else
    {
      params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + col_idx] = 0; // or some other default value indicating no neighbor
    }
  }
  __syncthreads();
  #if 0
  // Third Step: for each neighbor identify the neighbors
  // Mental mode of graph: row#|neighbor1 neighbor2 neighbor3 ...
  for (int64_t col = static_cast<int64_t>(col_idx); col < params.graph_cols/2;
       col += static_cast<int64_t>(blockDim.x)) {
    auto cur_row_cur_neighbor = params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + col];
    // Guard against CAGRA sentinel values (e.g. UINT32_MAX) that can appear when
    // the search cannot fill all k slots (graph degradation after concurrent inserts).
    if (static_cast<int64_t>(cur_row_cur_neighbor) >= params.graph_rows) { continue; }
    // Do not update rows that are also being inserted in this launch.
    if (is_row_in_insert_batch(params, static_cast<uint64_t>(cur_row_cur_neighbor))) { continue; }
    bool reverse_edge_added = false;
    for (int64_t col2 = 0; col2 < params.graph_cols; col2++) {
      auto neighbor_of_cur_row_cur_neighbor =
        params.graph[(cur_row_cur_neighbor * static_cast<uint64_t>(params.graph_cols)) + col2];
      // scan the neighbours of cur_row
      for (int64_t entry = 0; entry < params.graph_cols; entry++) {
        if (params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + entry] == 
        neighbor_of_cur_row_cur_neighbor) {
          auto target_offset = (cur_row_cur_neighbor * static_cast<uint64_t>(params.graph_cols)) + col2;
          //auto old_val       = params.graph[target_offset];
          // Preserve existing valid edges; only backfill empty/stale slots.
          //if (is_invalid_or_deleted_neighbor(params, old_val)) 
          {
            params.graph[target_offset] =
              cur_row;  // add cur_row as a neighbor to the neighbors of cur_row's neighbors
            reverse_edge_added = true;
            break;
          }
        }
      }
      if (reverse_edge_added) { break; } // no need to continue scanning cur_row_cur_neighbor if reverse edge is added
    }
  }
  #endif
  //__syncthreads();
   if (col_idx == 0) { params.deleted_rows[cur_row] = 0; }
  //__syncthreads();
  
}  
 template<typename T>
__global__ void cuvs_bang_insert_kernel2(InsertKernelParams params)
{
  auto row_idx = blockIdx.x;
  auto col_idx = threadIdx.x;

  if (row_idx >= params.num_ids || col_idx >= params.graph_cols) { return; }

  // identify the current row to be processed by this thread
  auto cur_row = params.row_num[row_idx];

  if (cur_row >= static_cast<uint64_t>(params.graph_rows)) { return; }
  
  // Third Step: for each neighbor identify the neighbors
  // Mental mode of graph: row#|neighbor1 neighbor2 neighbor3 ...
  for (int64_t col = static_cast<int64_t>(col_idx); col < params.graph_cols/2;
       col += static_cast<int64_t>(blockDim.x)) {
    auto cur_row_cur_neighbor = params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + col];
    // Guard against CAGRA sentinel values (e.g. UINT32_MAX) that can appear when
    // the search cannot fill all k slots (graph degradation after concurrent inserts).
    if (static_cast<int64_t>(cur_row_cur_neighbor) >= params.graph_rows) { continue; }
    // Do not update rows that are also being inserted in this launch.
    if (is_row_in_insert_batch(params, static_cast<uint64_t>(cur_row_cur_neighbor))) { continue; }
    bool reverse_edge_added = false;
    for (int64_t col2 = 0; col2 < params.graph_cols; col2++) {
      auto neighbor_of_cur_row_cur_neighbor =
        params.graph[(cur_row_cur_neighbor * static_cast<uint64_t>(params.graph_cols)) + col2];
      // scan the neighbours of cur_row
      for (int64_t entry = 0; entry < params.graph_cols; entry++) {
        if (params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + entry] == 
        neighbor_of_cur_row_cur_neighbor) {
          auto target_offset = (cur_row_cur_neighbor * static_cast<uint64_t>(params.graph_cols)) + col2;
          //auto old_val       = params.graph[target_offset];
          // Preserve existing valid edges; only backfill empty/stale slots.
          //if (is_invalid_or_deleted_neighbor(params, old_val)) 
          {
            params.graph[target_offset] =
              cur_row;  // add cur_row as a neighbor to the neighbors of cur_row's neighbors
            reverse_edge_added = true;
            break;
          }
        }
      }
      if (reverse_edge_added) { break; } // no need to continue scanning cur_row_cur_neighbor if reverse edge is added
    }
  }
  
}

template<typename T>
void launch_cuvs_bang_insert_kernel(const uint64_t* ids,
                                    size_t num_ids,
                                    const uint32_t* graph_ptr,
                                    int64_t graph_rows,
                                    int64_t graph_cols,
                                    const std::shared_ptr<rmm::device_buffer>& d_deleted_rows,
                                    const std::shared_ptr<rmm::device_buffer>& d_distance_threshold1,
                                    const std::shared_ptr<rmm::device_buffer>& d_distance_threshold2,
                                    const T* h_insert_vectors,
                                    T* d_dataset_vectors,
                                    int vector_dim,
                                    const algo_base::index_type* d_search_neighbors,
                                    const float* d_search_distances,
                                    int recall_at_k)
{
  if (num_ids == 0) { return; }
  if (graph_rows <= 0 || graph_cols <= 0) { return; }
  
  auto* mutable_graph_ptr = const_cast<uint32_t*>(graph_ptr);
  InsertKernelParams params{nullptr, num_ids, mutable_graph_ptr, graph_rows, graph_cols,
                            static_cast<uint8_t*>(d_deleted_rows->data()),
                            static_cast<float*>(d_distance_threshold1->data()),
                            static_cast<float*>(d_distance_threshold2->data())};

  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&params.row_num), num_ids * sizeof(uint64_t)));
  RAFT_CUDA_TRY(cudaMemcpy(params.row_num, ids, num_ids * sizeof(uint64_t), cudaMemcpyHostToDevice));

  uint32_t threads = static_cast<uint32_t>(graph_cols > 1024 ? 1024 : graph_cols); // The kernel expect this to be graph degree
  auto blocks      = static_cast<uint32_t>(num_ids);

  // copy the h_insert_vectrs to device
  #if 1
  T* d_insert_vectors;
  
  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_insert_vectors), num_ids * vector_dim * sizeof(T)));
  RAFT_CUDA_TRY(cudaMemcpy(d_insert_vectors, h_insert_vectors, num_ids * vector_dim * sizeof(T), cudaMemcpyHostToDevice));

  cuvs_bang_insert_kernel1<<<blocks, threads>>>(params, d_insert_vectors,d_dataset_vectors, 
    vector_dim, recall_at_k, d_search_neighbors, d_search_distances );
  
  cuvs_bang_insert_kernel2<T><<<blocks, graph_cols/2>>>(params);

  RAFT_CUDA_TRY(cudaPeekAtLastError());

  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  
  RAFT_CUDA_TRY(cudaFree(params.row_num));  
  RAFT_CUDA_TRY(cudaFree(d_insert_vectors));
  #endif

  std::printf("[bang] insert kernel processed=%zu ids\n", num_ids);
}

__global__ void cuvs_bang_delete_kernel1(DeleteKernelParams params)
{
  // Print graph dimensions once from thread 0 and also first and last row_num to be deleted
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[bang_delete_kernel] graph_rows=%lld graph_cols=%lld num_ids=%llu first_row=%llu last_row=%llu\n",
           (long long)params.graph_rows, (long long)params.graph_cols, (unsigned long long)params.num_ids,
           (unsigned long long)params.row_num[0], (unsigned long long)params.row_num[params.num_ids - 1]);
  }

  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= params.num_ids) { return; }

  // identify the current row to be processed by this thread
  auto cur_row = params.row_num[idx];

  if (cur_row >= static_cast<uint64_t>(params.graph_rows)) { return; }

  // mark the current row as deleted 
  params.deleted_rows[cur_row] = 1;
  __syncthreads();
}  



__global__ void cuvs_bang_delete_kernel2_BFS_shm(DeleteKernelParams params,
                                                 uint32_t* frontier_curr_all,
                                                 uint32_t* frontier_next_all,
                                                 int* frontier_size_all,
                                                 int* next_size_all,
                                                 int64_t frontier_capacity)
{
  // Print graph dimensions once from thread 0 and also first and last row_num to be deleted
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[bang_kernel] graph_rows=%lld graph_cols=%lld num_ids=%llu first_row=%llu last_row=%llu\n",
           (long long)params.graph_rows, (long long)params.graph_cols, (unsigned long long)params.num_ids,
           (unsigned long long)params.row_num[0], (unsigned long long)params.row_num[params.num_ids - 1]);
  }

  auto row_idx = blockIdx.x;
  auto col_idx = threadIdx.x;
  if (row_idx >= params.num_ids || col_idx >= params.graph_cols) { return; }

  // identify the current row to be deleted by this thread-block
  auto cur_row = params.row_num[row_idx];

  if (cur_row >= static_cast<uint64_t>(params.graph_rows)) { return; }

  if (params.deleted_rows[cur_row] != 1) { return; } // skip if not marked deleted

  auto block_offset = static_cast<uint64_t>(row_idx) * static_cast<uint64_t>(frontier_capacity);
  auto* frontier_curr = frontier_curr_all + block_offset;
  auto* frontier_next = frontier_next_all + block_offset;
  auto* frontier_size = frontier_size_all + row_idx;
  auto* next_size     = next_size_all + row_idx;

  // Diagnostics: count IN-edge replacements discovered by BFS.
  __shared__ unsigned long long bfs_in_edges_count;
  if (col_idx == 0) {
    bfs_in_edges_count    = 0;
  }
  __syncthreads();

  // find a guaranteed non-deleted neighbour of cur_row to replace deleted-row references.
  // Only thread 0 scans the adjacency list; all others wait at __syncthreads().
  // Using uint32_t to match the graph element type; UINT32_MAX is the "not found" sentinel.
  __shared__ uint32_t backup_replacement_neighbor ;
  if (col_idx == 0) {
    backup_replacement_neighbor = UINT32_MAX; // sentinel: no valid backup found yet
    for (int64_t c = 0; c < params.graph_cols; c++) {
      uint32_t nb = params.graph[cur_row * static_cast<uint64_t>(params.graph_cols) + c];
      if (static_cast<int64_t>(nb) < params.graph_rows && params.deleted_rows[nb] != 1) {
        backup_replacement_neighbor = nb;
        break;
      }
    }
    if (backup_replacement_neighbor == UINT32_MAX) {
      printf("[bang_kernel] WARNING: No non-deleted neighbor found for deleted row %llu\n",
             (unsigned long long)cur_row);
    }
  }

  __syncthreads();
#if 0
  // Ground truth: linear scan over the full graph for incoming references to cur_row.
  for (int64_t col = col_idx; col < params.graph_cols; col += blockDim.x) {
    unsigned long long local_hits = 0;
    for (int64_t row = 0; row < params.graph_rows; ++row) {
      auto neighbor = params.graph[row * static_cast<uint64_t>(params.graph_cols) + col];
      if (neighbor == cur_row) { ++local_hits; }
    }
    if (local_hits > 0) { atomicAdd(&linear_in_edges_count, local_hits); }
  }
  #endif
  __syncthreads();

  // Bounded BFS over candidate rows likely to contain incoming references to cur_row.
  // This avoids a full O(graph_rows * graph_cols) scan per deleted row.
  constexpr int kBfsLevels      = 2;

  if (col_idx == 0) {
    *frontier_size = 0;
    *next_size     = 0;
    for (int64_t c = 0; c < params.graph_cols && *frontier_size < frontier_capacity; ++c) {
      uint32_t seed = params.graph[cur_row * static_cast<uint64_t>(params.graph_cols) + c];
      if (static_cast<int64_t>(seed) >= params.graph_rows) { continue; }
      if (seed == cur_row || params.deleted_rows[seed] == 1) { continue; }
      frontier_curr[(*frontier_size)++] = seed;
    }
  }
  __syncthreads();

  for (int level = 0; level < kBfsLevels; ++level) {
    int curr_size = *frontier_size;
    if (curr_size == 0) { break; }

    for (int node_pos = col_idx; node_pos < curr_size; node_pos += blockDim.x) {
      uint32_t node = frontier_curr[node_pos];
      if (static_cast<int64_t>(node) >= params.graph_rows || params.deleted_rows[node] == 1) {
        // log this as a warning, but continue processing other nodes in the frontier
       /* printf("[bang_kernel][WARNING] BFS frontier node %llu is invalid or deleted; skipping.\n",
               (unsigned long long)node);*/
        continue;
      }

      // Each thread owns a distinct frontier node and scans its full adjacency list.
      unsigned long long local_bfs_hits = 0;
      for (int64_t col = 0; col < params.graph_cols; ++col) {
        auto off      = node * static_cast<uint64_t>(params.graph_cols) + col;
        auto neighbor = params.graph[off];

        if (static_cast<int64_t>(neighbor) < params.graph_rows && neighbor != cur_row &&
            params.deleted_rows[neighbor] != 1) {
          int pos = atomicAdd(next_size, 1);
          if (pos < frontier_capacity) 
          { 
            frontier_next[pos] = neighbor; 
          }
          else
          {
            /* printf("[bang_kernel][WARNING] BFS frontier capacity exceeded for deleted row %llu; some neighbors may not be processed.\n",
                   (unsigned long long)cur_row);*/
          }
        }

        if (neighbor == cur_row) {
          ++local_bfs_hits;
          uint32_t candidate = params.graph[cur_row * static_cast<uint64_t>(params.graph_cols) + col];
          if (static_cast<int64_t>(candidate) < params.graph_rows && params.deleted_rows[candidate] != 1) {
            params.graph[off] = candidate;
          } else if (backup_replacement_neighbor != UINT32_MAX) {
            params.graph[off] = backup_replacement_neighbor;
          }
        }
      }
      if (local_bfs_hits > 0) { atomicAdd(&bfs_in_edges_count, local_bfs_hits); }
    } // end for for current frontier

    __syncthreads();

    // Swap next frontier into current frontier for the next iteration.
    int capped_size = *next_size > frontier_capacity ? static_cast<int>(frontier_capacity)
                                                     : *next_size;
    for (int i = col_idx; i < capped_size; i += blockDim.x) {
      frontier_curr[i] = frontier_next[i];
    }
    __syncthreads();

    if (col_idx == 0) {
      *frontier_size = capped_size;
      *next_size     = 0;
    }
    __syncthreads();
  } // end for BFS levels
#if 0
  if (col_idx == 0 && bfs_in_edges_count != linear_in_edges_count) {
    printf("\n[bang_kernel][WARNING][IN_EDGE_MISMATCH] ***************************************\n");
    printf("[bang_kernel][WARNING][IN_EDGE_MISMATCH] deleted_row=%llu linear_in_edges=%llu bfs_discovered_in_edges=%llu\n",
           (unsigned long long)cur_row,
           (unsigned long long)linear_in_edges_count,
           (unsigned long long)bfs_in_edges_count);
    printf("[bang_kernel][WARNING][IN_EDGE_MISMATCH] BFS did not match linear scan.\n");
    printf("[bang_kernel][WARNING][IN_EDGE_MISMATCH] ***************************************\n\n");
  }

#endif
}




__global__ void cuvs_bang_delete_kernel2_BFS_shm_bkp(DeleteKernelParams params,
                                                 uint32_t* frontier_curr_all,
                                                 uint32_t* frontier_next_all,
                                                 int* frontier_size_all,
                                                 int* next_size_all,
                                                 int64_t frontier_capacity)
{
  // Print graph dimensions once from thread 0 and also first and last row_num to be deleted
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[bang_kernel] graph_rows=%lld graph_cols=%lld num_ids=%llu first_row=%llu last_row=%llu\n",
           (long long)params.graph_rows, (long long)params.graph_cols, (unsigned long long)params.num_ids,
           (unsigned long long)params.row_num[0], (unsigned long long)params.row_num[params.num_ids - 1]);
  }

  auto row_idx = blockIdx.x;
  auto col_idx = threadIdx.x;
  if (row_idx >= params.num_ids || col_idx >= params.graph_cols) { return; }

  // identify the current row to be deleted by this thread-block
  auto cur_row = params.row_num[row_idx];

  if (cur_row >= static_cast<uint64_t>(params.graph_rows)) { return; }

  if (params.deleted_rows[cur_row] != 1) { return; } // skip if not marked deleted

  auto block_offset = static_cast<uint64_t>(row_idx) * static_cast<uint64_t>(frontier_capacity);
  auto* frontier_curr = frontier_curr_all + block_offset;
  auto* frontier_next = frontier_next_all + block_offset;
  auto* frontier_size = frontier_size_all + row_idx;
  auto* next_size     = next_size_all + row_idx;

  // find a guaranteed non-deleted neighbour of cur_row to replace deleted-row references.
  // Only thread 0 scans the adjacency list; all others wait at __syncthreads().
  // Using uint32_t to match the graph element type; UINT32_MAX is the "not found" sentinel.
  __shared__ uint32_t backup_replacement_neighbor ;
  if (col_idx == 0) {
    backup_replacement_neighbor = UINT32_MAX; // sentinel: no valid backup found yet
    for (int64_t c = 0; c < params.graph_cols; c++) {
      uint32_t nb = params.graph[cur_row * static_cast<uint64_t>(params.graph_cols) + c];
      if (static_cast<int64_t>(nb) < params.graph_rows && params.deleted_rows[nb] != 1) {
        backup_replacement_neighbor = nb;
        break;
      }
    }
    if (backup_replacement_neighbor == UINT32_MAX) {
      printf("[bang_kernel] WARNING: No non-deleted neighbor found for deleted row %llu\n",
             (unsigned long long)cur_row);
    }
  }

  __syncthreads();

  // Bounded BFS over candidate rows likely to contain incoming references to cur_row.
  // This avoids a full O(graph_rows * graph_cols) scan per deleted row.
  constexpr int kBfsLevels      = 8;

  if (col_idx == 0) {
    *frontier_size = 0;
    *next_size     = 0;
    for (int64_t c = 0; c < params.graph_cols && *frontier_size < frontier_capacity; ++c) {
      uint32_t seed = params.graph[cur_row * static_cast<uint64_t>(params.graph_cols) + c];
      if (static_cast<int64_t>(seed) >= params.graph_rows) { continue; }
      if (seed == cur_row || params.deleted_rows[seed] == 1) { continue; }
      frontier_curr[(*frontier_size)++] = seed;
    }
  }
  __syncthreads();

  for (int level = 0; level < kBfsLevels; ++level) {
    int curr_size = *frontier_size;
    if (curr_size == 0) { break; }

    for (int node_pos = 0; node_pos < curr_size; ++node_pos) {
      uint32_t node = frontier_curr[node_pos];
      if (static_cast<int64_t>(node) >= params.graph_rows || params.deleted_rows[node] == 1) {
        continue;
      }

      // Parallel-in-column replacement for this candidate row.
      for (int64_t col = col_idx; col < params.graph_cols; col += blockDim.x) {
        auto off      = node * static_cast<uint64_t>(params.graph_cols) + col;
        auto neighbor = params.graph[off];
        if (neighbor == cur_row) {
          uint32_t candidate = params.graph[cur_row * static_cast<uint64_t>(params.graph_cols) + col];
          if (static_cast<int64_t>(candidate) < params.graph_rows && params.deleted_rows[candidate] != 1) {
            params.graph[off] = candidate;
          } else if (backup_replacement_neighbor != UINT32_MAX) {
            params.graph[off] = backup_replacement_neighbor;
          }
        }
      }
      __syncthreads();

      // Expand BFS frontier from this node.
      for (int64_t col = col_idx; col < params.graph_cols; col += blockDim.x) {
        uint32_t nb = params.graph[node * static_cast<uint64_t>(params.graph_cols) + col];
        if (static_cast<int64_t>(nb) >= params.graph_rows) { continue; }
        if (nb == cur_row || params.deleted_rows[nb] == 1) { continue; }
        int pos = atomicAdd(next_size, 1);
        if (pos < frontier_capacity) { frontier_next[pos] = nb; }
      }
      __syncthreads();
    }

    // Swap next frontier into current frontier for the next iteration.
    int capped_size = *next_size > frontier_capacity ? static_cast<int>(frontier_capacity)
                                                     : *next_size;
    for (int i = col_idx; i < capped_size; i += blockDim.x) {
      frontier_curr[i] = frontier_next[i];
    }
    __syncthreads();

    if (col_idx == 0) {
      *frontier_size = capped_size;
      *next_size     = 0;
    }
    __syncthreads();
  } // end for BFS levels


}




__global__ void cuvs_bang_delete_kernel2_naive(DeleteKernelParams params)
{
  // Print graph dimensions once from thread 0 and also first and last row_num to be deleted
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[bang_kernel] graph_rows=%lld graph_cols=%lld num_ids=%llu first_row=%llu last_row=%llu\n",
           (long long)params.graph_rows, (long long)params.graph_cols, (unsigned long long)params.num_ids,
           (unsigned long long)params.row_num[0], (unsigned long long)params.row_num[params.num_ids - 1]);
  }

  auto row_idx = blockIdx.x;
  auto col_idx = threadIdx.x;
  if (row_idx >= params.num_ids || col_idx >= params.graph_cols) { return; }

  // identify the current row to be deleted by this thread-block
  auto cur_row = params.row_num[row_idx];

  if (cur_row >= static_cast<uint64_t>(params.graph_rows)) { return; }

  if (params.deleted_rows[cur_row] != 1) { return; } // skip if not marked deleted

  // find a guaranteed non-deleted neighbour of cur_row to replace deleted-row references.
  // Only thread 0 scans the adjacency list; all others wait at __syncthreads().
  // Using uint32_t to match the graph element type; UINT32_MAX is the "not found" sentinel.
  __shared__ uint32_t backup_replacement_neighbor ;
  if (col_idx == 0) {
    backup_replacement_neighbor = UINT32_MAX; // sentinel: no valid backup found yet
    for (int64_t c = 0; c < params.graph_cols; c++) {
      uint32_t nb = params.graph[cur_row * static_cast<uint64_t>(params.graph_cols) + c];
      if (static_cast<int64_t>(nb) < params.graph_rows && params.deleted_rows[nb] != 1) {
        backup_replacement_neighbor = nb;
        break;
      }
    }
    if (backup_replacement_neighbor == UINT32_MAX) {
      printf("[bang_kernel] WARNING: No non-deleted neighbor found for deleted row %llu\n",
             (unsigned long long)cur_row);
    }
  }

  __syncthreads();

  //uint32_t circular_counter = 0;
  // scan the entire graph to find the occurrence of cur_row
  for (int64_t col = col_idx; col < params.graph_cols; col += blockDim.x) {
  for (int64_t row = 0; row < params.graph_rows; row++) {
    if (params.deleted_rows[row] == 1)  { continue; } // skip deleted rows

    
      auto neighbor = params.graph[row * static_cast<uint64_t>(params.graph_cols) + col];
      if (neighbor == cur_row) {
        
        
        //uint32_t cur_row_neighbour = params.graph[cur_row * static_cast<uint64_t>(params.graph_cols) + circular_counter];
        //if (params.deleted_rows[cur_row_neighbour] != 1){
          // Replace the deleted-node reference with a live neighbor from the deleted row.
          {
            uint32_t candidate = params.graph[cur_row * static_cast<uint64_t>(params.graph_cols) + col];
            if (static_cast<int64_t>(candidate) < params.graph_rows && params.deleted_rows[candidate] != 1) {
              params.graph[row * static_cast<uint64_t>(params.graph_cols) + col] = candidate;
            } else if (backup_replacement_neighbor != UINT32_MAX) {
              params.graph[row * static_cast<uint64_t>(params.graph_cols) + col] = backup_replacement_neighbor;
            }
            // else: no live node available; leave the slot as-is (graph is severely degraded)
          }


          // NOTE: no break here — multiple rows can have cur_row at the same column
          // position, so we must continue scanning all rows for this column.
          //cur_row_neighbour;
          //if (++circular_counter >= params.graph_cols ) circular_counter = 0;
        //  break; // go to next neighbor
        //} else {
        //    if (++circular_counter >= params.graph_cols ) circular_counter = 0;
        //}
      
      } // end if neighbor match
    }  // end for cols  
  } // end for rows

    #if 0
      uint32_t temp_iter = 0;
      while (temp_iter++ < params.graph_cols-1) {
        uint32_t cur_neighbour = params.graph[cur_row * static_cast<uint64_t>(params.graph_cols) + temp_iter];
        if (params.deleted_rows[cur_neighbour] != 1) {
          // Copy adjacency list of a live neighbor into the deleted row.
          for (int64_t col = 0; col < params.graph_cols; col++) {
            uint32_t candidate = params.graph[cur_neighbour * static_cast<uint64_t>(params.graph_cols) + col];
            if ((params.deleted_rows[candidate] != 1)) {
              params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + col] = candidate;
            } else {
              params.graph[cur_row * static_cast<uint64_t>(params.graph_cols) + col] = 1;
            }
          }
          break;
        }
      } // end while
    #endif
}




void cuvs_bang_kernel_init()
{

}


void launch_cuvs_bang_delete_kernel(const uint64_t* ids,
                                    size_t num_ids,
                                    const uint32_t* graph_ptr,
                                    int64_t graph_rows,
                                    int64_t graph_cols,
                                    const std::shared_ptr<rmm::device_buffer>& d_deleted_rows,
                                    uint32_t* d_frontier_curr,
                                    uint32_t* d_frontier_next,
                                    int* d_frontier_size,
                                    int* d_next_size,
                                    int64_t frontier_capacity,
                                    size_t max_workspace_ids)
{
  if (num_ids == 0) { return; }
  if (graph_rows <= 0 || graph_cols <= 0) { return; }
  if (!d_frontier_curr || !d_frontier_next || !d_frontier_size || !d_next_size) {
    std::printf("[bang] delete kernel missing workspace buffers\n");
    return;
  }
  if (num_ids > max_workspace_ids) {
    std::printf("[bang] delete kernel workspace too small: num_ids=%zu max_ids=%zu\n",
                num_ids,
                max_workspace_ids);
    return;
  }
  if (frontier_capacity <= 0) {
    std::printf("[bang] delete kernel invalid frontier_capacity=%lld\n",
                static_cast<long long>(frontier_capacity));
    return;
  }
  
  auto* mutable_graph_ptr = const_cast<uint32_t*>(graph_ptr);
  DeleteKernelParams params{nullptr, num_ids, mutable_graph_ptr, graph_rows, graph_cols,
                            static_cast<uint8_t*>(d_deleted_rows->data())};

  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&params.row_num), num_ids * sizeof(uint64_t)));
  RAFT_CUDA_TRY(cudaMemcpy(params.row_num, ids, num_ids * sizeof(uint64_t), cudaMemcpyHostToDevice));

  constexpr uint32_t threads = 1024;
  auto blocks                = static_cast<uint32_t>((num_ids + threads - 1) / threads);

  cuvs_bang_delete_kernel1<<<blocks, threads>>>(params);

  RAFT_CUDA_TRY(cudaMemset(d_frontier_size, 0, num_ids * sizeof(int)));
  RAFT_CUDA_TRY(cudaMemset(d_next_size, 0, num_ids * sizeof(int)));

  auto bfs_threads = static_cast<uint32_t>(graph_cols > 1024 ? 1024 : graph_cols);
  cuvs_bang_delete_kernel2_BFS_shm<<<num_ids, bfs_threads>>>(params,
                                                              d_frontier_curr,
                                                              d_frontier_next,
                                                              d_frontier_size,
                                                              d_next_size,
                                                              frontier_capacity);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  std::printf("[bang] kernel processed=%zu ids\n", num_ids);

  RAFT_CUDA_TRY(cudaFree(params.row_num));
}
// Explicit template instantiations for supported types
template void launch_cuvs_bang_insert_kernel<float>(const uint64_t* ids,
                                                     size_t num_ids,
                                                     const uint32_t* graph_ptr,
                                                     int64_t graph_rows,
                                                     int64_t graph_cols,
                                                     const std::shared_ptr<rmm::device_buffer>&
                                                       d_deleted_rows,
                                                     const std::shared_ptr<rmm::device_buffer>&
                                                       d_distance_threshold1,
                                                     const std::shared_ptr<rmm::device_buffer>&
                                                       d_distance_threshold2,
                                                     const float* h_insert_vectors,
                                                     float* d_dataset_vectors,
                                                     int vector_dim,
                                                     const algo_base::index_type* d_search_neighbors,
                                                     const float* d_search_distances,
                                                     int recall_at_k);

template void launch_cuvs_bang_insert_kernel<half>(const uint64_t* ids,
                                                    size_t num_ids,
                                                    const uint32_t* graph_ptr,
                                                    int64_t graph_rows,
                                                    int64_t graph_cols,
                                                    const std::shared_ptr<rmm::device_buffer>&
                                                      d_deleted_rows,
                                                    const std::shared_ptr<rmm::device_buffer>&
                                                      d_distance_threshold1,
                                                    const std::shared_ptr<rmm::device_buffer>&
                                                      d_distance_threshold2,
                                                    const half* h_insert_vectors,
                                                    half* d_dataset_vectors,
                                                    int vector_dim,
                                                    const algo_base::index_type* d_search_neighbors,
                                                    const float* d_search_distances,
                                                    int recall_at_k);

template void launch_cuvs_bang_insert_kernel<int8_t>(const uint64_t* ids,
                                                      size_t num_ids,
                                                      const uint32_t* graph_ptr,
                                                      int64_t graph_rows,
                                                      int64_t graph_cols,
                                                      const std::shared_ptr<rmm::device_buffer>&
                                                        d_deleted_rows,
                                                      const std::shared_ptr<rmm::device_buffer>&
                                                        d_distance_threshold1,
                                                      const std::shared_ptr<rmm::device_buffer>&
                                                        d_distance_threshold2,
                                                      const int8_t* h_insert_vectors,
                                                      int8_t* d_dataset_vectors,
                                                      int vector_dim,
                                                      const algo_base::index_type* d_search_neighbors,
                                                      const float* d_search_distances,
                                                      int recall_at_k);

template void launch_cuvs_bang_insert_kernel<uint8_t>(const uint64_t* ids,
                                                       size_t num_ids,
                                                       const uint32_t* graph_ptr,
                                                       int64_t graph_rows,
                                                       int64_t graph_cols,
                                                       const std::shared_ptr<rmm::device_buffer>&
                                                         d_deleted_rows,
                                                       const std::shared_ptr<rmm::device_buffer>&
                                                         d_distance_threshold1,
                                                       const std::shared_ptr<rmm::device_buffer>&
                                                         d_distance_threshold2,
                                                       const uint8_t* h_insert_vectors,
                                                       uint8_t* d_dataset_vectors,
                                                       int vector_dim,
                                                       const algo_base::index_type* d_search_neighbors,
                                                       const float* d_search_distances,
                                                       int recall_at_k);


}  // namespace cuvs::bench::detail