/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../cuvs/cuvs_cagra_wrapper.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <raft/util/cudart_utils.hpp>
#include <cub/cub.cuh>

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
    uint32_t* reverse_graph;
    int64_t reverse_graph_cols;
    uint32_t* reverse_row_ptr;
    uint32_t* reverse_counts;
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
    uint32_t* reverse_graph;
    int64_t reverse_graph_cols;
    uint32_t* reverse_counts;
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

__global__ void cuvs_bang_count_reverse_graph_kernel(const uint32_t* graph,
                                                     int64_t graph_rows,
                                                     int64_t graph_cols,
                                                     const uint8_t* deleted_rows,
                                                     uint32_t* reverse_counts,
                                                     uint32_t* skipped_invalid_dst_count,
                                                     uint32_t* skipped_deleted_src_count,
                                                     uint32_t* skipped_deleted_dst_count)
{
  auto col_idx = static_cast<int64_t>(threadIdx.x);
  auto row_idx = static_cast<int64_t>(blockIdx.x);

  // Each block starts from row=blockIdx.x and then processes row+gridDim.x, row+2*gridDim.x, ...
  for (int64_t src_row = row_idx; src_row < graph_rows; src_row += static_cast<int64_t>(gridDim.x)) {
    if (deleted_rows && deleted_rows[src_row] == 1) {
      if (col_idx == 0) { atomicAdd(skipped_deleted_src_count, static_cast<uint32_t>(graph_cols)); }
      continue;
    }

    for (int64_t col = col_idx; col < graph_cols; col += static_cast<int64_t>(blockDim.x)) {
      auto off = static_cast<uint64_t>(src_row) * static_cast<uint64_t>(graph_cols) +
                 static_cast<uint64_t>(col);
      auto dst = graph[off];
      if (static_cast<int64_t>(dst) >= graph_rows) {
        atomicAdd(skipped_invalid_dst_count, 1U);
        continue;
      }
      if (deleted_rows && deleted_rows[dst] == 1) {
        atomicAdd(skipped_deleted_dst_count, 1U);
        continue;
      }

      auto pos = atomicAdd(reverse_counts + dst, 1U);
      (void)pos;
    }
  }
}

__global__ void cuvs_bang_fill_reverse_graph_kernel(const uint32_t* graph,
                                                    int64_t graph_rows,
                                                    int64_t graph_cols,
                                                    const uint8_t* deleted_rows,
                                                    uint32_t* reverse_graph,
                                                    uint32_t* reverse_write_offsets)
{
  auto col_idx = static_cast<int64_t>(threadIdx.x);
  auto row_idx = static_cast<int64_t>(blockIdx.x);

  for (int64_t src_row = row_idx; src_row < graph_rows; src_row += static_cast<int64_t>(gridDim.x)) {
    if (deleted_rows && deleted_rows[src_row] == 1) { continue; }

    for (int64_t col = col_idx; col < graph_cols; col += static_cast<int64_t>(blockDim.x)) {
      auto off = static_cast<uint64_t>(src_row) * static_cast<uint64_t>(graph_cols) +
                 static_cast<uint64_t>(col);
      auto dst = graph[off];
      if (static_cast<int64_t>(dst) >= graph_rows) { continue; }
      if (deleted_rows && deleted_rows[dst] == 1) { continue; }

      auto pos = atomicAdd(reverse_write_offsets + dst, 1U);
      reverse_graph[pos] = static_cast<uint32_t>(src_row);
    }
  }
}

__global__ void cuvs_bang_verify_no_deleted_neighbors_kernel(const uint32_t* graph,
                                                             int64_t graph_rows,
                                                             int64_t graph_cols,
                                                             const uint8_t* deleted_rows,
                                                             uint64_t* deleted_edge_count,
                                                             uint64_t* sample_src,
                                                             uint64_t* sample_dst,
                                                             uint32_t max_samples)
{
  auto col_idx = static_cast<int64_t>(threadIdx.x);
  auto row_idx = static_cast<int64_t>(blockIdx.x);

  for (int64_t src_row = row_idx; src_row < graph_rows; src_row += static_cast<int64_t>(gridDim.x)) {
    if (deleted_rows && deleted_rows[src_row] == 1) { continue; }

    for (int64_t col = col_idx; col < graph_cols; col += static_cast<int64_t>(blockDim.x)) {
      auto off = static_cast<uint64_t>(src_row) * static_cast<uint64_t>(graph_cols) +
                 static_cast<uint64_t>(col);
      auto dst = graph[off];
      if (static_cast<int64_t>(dst) >= graph_rows) { continue; }
      if (!(deleted_rows && deleted_rows[dst] == 1)) { continue; }

      auto hit_idx = static_cast<uint64_t>(atomicAdd(
        reinterpret_cast<unsigned long long*>(deleted_edge_count),
        static_cast<unsigned long long>(1)));
      if (hit_idx < static_cast<uint64_t>(max_samples)) {
        sample_src[hit_idx] = static_cast<uint64_t>(src_row);
        sample_dst[hit_idx] = static_cast<uint64_t>(dst);
      }
    }
  }
}

__global__ void cuvs_bang_sanitize_deleted_neighbors_kernel(uint32_t* graph,
                                                            int64_t graph_rows,
                                                            int64_t graph_cols,
                                                            const uint8_t* deleted_rows,
                                                            uint64_t* replaced_edge_count,
                                                            uint64_t* no_fallback_row_count)
{
  auto row_idx = static_cast<int64_t>(blockIdx.x);
  auto col_idx = static_cast<int64_t>(threadIdx.x);
  if (row_idx >= graph_rows || col_idx >= graph_cols) { return; }
  if (deleted_rows && deleted_rows[row_idx] == 1) { return; }

  __shared__ uint32_t s_fallback_neighbor;
  if (col_idx == 0) {
    s_fallback_neighbor = UINT32_MAX;
    for (int64_t c = 0; c < graph_cols; ++c) {
      auto nb = graph[static_cast<uint64_t>(row_idx) * static_cast<uint64_t>(graph_cols) +
                      static_cast<uint64_t>(c)];
      if (static_cast<int64_t>(nb) < graph_rows && (!deleted_rows || deleted_rows[nb] != 1)) {
        s_fallback_neighbor = nb;
        break;
      }
    }
    if (s_fallback_neighbor == UINT32_MAX) {
      atomicAdd(reinterpret_cast<unsigned long long*>(no_fallback_row_count),
                static_cast<unsigned long long>(1));
    }
  }
  __syncthreads();

  auto off = static_cast<uint64_t>(row_idx) * static_cast<uint64_t>(graph_cols) +
             static_cast<uint64_t>(col_idx);
  auto dst = graph[off];
  if (static_cast<int64_t>(dst) >= graph_rows) { return; }
  if (!(deleted_rows && deleted_rows[dst] == 1)) { return; }
  if (s_fallback_neighbor == UINT32_MAX) { return; }

  graph[off] = s_fallback_neighbor;
  atomicAdd(reinterpret_cast<unsigned long long*>(replaced_edge_count),
            static_cast<unsigned long long>(1));
}

void cuvs_bang_sanitize_deleted_neighbors(uint32_t* graph,
                                          int64_t graph_rows,
                                          int64_t graph_cols,
                                          const uint8_t* deleted_rows)
{
  if (!graph || !deleted_rows || graph_rows <= 0 || graph_cols <= 0) { return; }

  uint64_t* d_replaced_edge_count = nullptr;
  uint64_t* d_no_fallback_row_count = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_replaced_edge_count), sizeof(uint64_t)));
  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_no_fallback_row_count), sizeof(uint64_t)));
  RAFT_CUDA_TRY(cudaMemset(d_replaced_edge_count, 0, sizeof(uint64_t)));
  RAFT_CUDA_TRY(cudaMemset(d_no_fallback_row_count, 0, sizeof(uint64_t)));

  uint32_t threads = static_cast<uint32_t>(graph_cols > 1024 ? 1024 : graph_cols);
  if (threads == 0) {
    RAFT_CUDA_TRY(cudaFree(d_replaced_edge_count));
    RAFT_CUDA_TRY(cudaFree(d_no_fallback_row_count));
    return;
  }

  auto blocks = static_cast<uint32_t>(graph_rows);
  cuvs_bang_sanitize_deleted_neighbors_kernel<<<blocks, threads>>>(graph,
                                                                    graph_rows,
                                                                    graph_cols,
                                                                    deleted_rows,
                                                                    d_replaced_edge_count,
                                                                    d_no_fallback_row_count);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  uint64_t h_replaced_edge_count = 0;
  uint64_t h_no_fallback_row_count = 0;
  RAFT_CUDA_TRY(cudaMemcpy(&h_replaced_edge_count,
                           d_replaced_edge_count,
                           sizeof(uint64_t),
                           cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(&h_no_fallback_row_count,
                           d_no_fallback_row_count,
                           sizeof(uint64_t),
                           cudaMemcpyDeviceToHost));

  if (h_replaced_edge_count > 0 || h_no_fallback_row_count > 0) {
    std::printf("[bang][sanitize_delete] replaced_deleted_edges=%llu rows_without_fallback=%llu\n",
                static_cast<unsigned long long>(h_replaced_edge_count),
                static_cast<unsigned long long>(h_no_fallback_row_count));
  }

  RAFT_CUDA_TRY(cudaFree(d_replaced_edge_count));
  RAFT_CUDA_TRY(cudaFree(d_no_fallback_row_count));
}

void cuvs_bang_verify_no_deleted_neighbors(const uint32_t* graph,
                                           int64_t graph_rows,
                                           int64_t graph_cols,
                                           const uint8_t* deleted_rows)
{
  if (!graph || !deleted_rows || graph_rows <= 0 || graph_cols <= 0) { return; }

  constexpr uint32_t kMaxSamples = 32;
  uint64_t* d_deleted_edge_count = nullptr;
  uint64_t* d_sample_src         = nullptr;
  uint64_t* d_sample_dst         = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_deleted_edge_count), sizeof(uint64_t)));
  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_sample_src), kMaxSamples * sizeof(uint64_t)));
  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_sample_dst), kMaxSamples * sizeof(uint64_t)));
  RAFT_CUDA_TRY(cudaMemset(d_deleted_edge_count, 0, sizeof(uint64_t)));

  uint32_t threads = static_cast<uint32_t>(graph_cols > 1024 ? 1024 : graph_cols);
  if (threads == 0) {
    RAFT_CUDA_TRY(cudaFree(d_deleted_edge_count));
    RAFT_CUDA_TRY(cudaFree(d_sample_src));
    RAFT_CUDA_TRY(cudaFree(d_sample_dst));
    return;
  }
  constexpr uint32_t kMaxBlocks = 10000;
  auto blocks = static_cast<uint32_t>(graph_rows < static_cast<int64_t>(kMaxBlocks)
                                        ? graph_rows
                                        : static_cast<int64_t>(kMaxBlocks));
  if (blocks == 0) {
    RAFT_CUDA_TRY(cudaFree(d_deleted_edge_count));
    RAFT_CUDA_TRY(cudaFree(d_sample_src));
    RAFT_CUDA_TRY(cudaFree(d_sample_dst));
    return;
  }

  cuvs_bang_verify_no_deleted_neighbors_kernel<<<blocks, threads>>>(graph,
                                                                     graph_rows,
                                                                     graph_cols,
                                                                     deleted_rows,
                                                                     d_deleted_edge_count,
                                                                     d_sample_src,
                                                                     d_sample_dst,
                                                                     kMaxSamples);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  uint64_t h_deleted_edge_count = 0;
  RAFT_CUDA_TRY(cudaMemcpy(&h_deleted_edge_count,
                           d_deleted_edge_count,
                           sizeof(uint64_t),
                           cudaMemcpyDeviceToHost));
  if (h_deleted_edge_count > 0) {
    auto sample_count = static_cast<uint32_t>(h_deleted_edge_count < kMaxSamples
                                                 ? h_deleted_edge_count
                                                 : kMaxSamples);
    std::vector<uint64_t> h_sample_src(sample_count);
    std::vector<uint64_t> h_sample_dst(sample_count);
    RAFT_CUDA_TRY(cudaMemcpy(h_sample_src.data(),
                             d_sample_src,
                             static_cast<size_t>(sample_count) * sizeof(uint64_t),
                             cudaMemcpyDeviceToHost));
    RAFT_CUDA_TRY(cudaMemcpy(h_sample_dst.data(),
                             d_sample_dst,
                             static_cast<size_t>(sample_count) * sizeof(uint64_t),
                             cudaMemcpyDeviceToHost));
    std::printf("[bang][verify_delete] ERROR: live rows still have deleted neighbors; total=%llu (showing %u)\n",
                static_cast<unsigned long long>(h_deleted_edge_count),
                sample_count);
    for (uint32_t i = 0; i < sample_count; ++i) {
      std::printf("[bang][verify_delete] sample %u: %llu -> %llu\n",
                  i,
                  static_cast<unsigned long long>(h_sample_src[i]),
                  static_cast<unsigned long long>(h_sample_dst[i]));
    }
  } else {
    std::printf("[bang][verify_delete] OK: no deleted neighbors found in live rows\n");
  }

  RAFT_CUDA_TRY(cudaFree(d_deleted_edge_count));
  RAFT_CUDA_TRY(cudaFree(d_sample_src));
  RAFT_CUDA_TRY(cudaFree(d_sample_dst));
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
  
  // Pre-compute a fallback neighbor to pad invalid/out-of-range slots so the graph
  // stays regular. Thread 0 scans the search results for the first valid live node
  // and broadcasts it through shared memory to all threads in the block.
  __shared__ uint32_t s_fallback_neighbor;
  if (col_idx == 0) {
   
   if( params.deleted_rows[cur_row] == 0)
   {
    // This row is already live (not deleted) — no need to search for a fallback neighbor.
    printf("[bang_insert_kernel1] WARNING: row %llu is already live (not deleted) before insertion; skipping fallback neighbor search.\n",
           static_cast<unsigned long long>(cur_row));
           return;
   }

   s_fallback_neighbor = UINT32_MAX; // sentinel: no valid fallback found yet
    for (int k = 0; k < recall_at_k; ++k) {
      auto nb = d_search_neighbors[row_idx * recall_at_k + k];
      if (nb >= 0 && nb < params.graph_rows && params.deleted_rows[nb] != 1) {
        s_fallback_neighbor = static_cast<uint32_t>(nb);
        break;
      }
    }
  }

  // First Step: copy d_insert_vectors to the appropriate location in the dataset 
  uint32_t dims_per_thread =  (vector_dim + blockDim.x - 1) / blockDim.x; // ceiling division to cover all dimensions

  // Each thread will copy dims_per_thread dimensions of the vector. Calculate the start and end dimension index for this thread.
  for (uint32_t cur_dim = threadIdx.x * dims_per_thread; cur_dim < vector_dim && cur_dim < (threadIdx.x + 1) * dims_per_thread; cur_dim++) {
    d_dataset_vectors[cur_row * vector_dim + cur_dim] = d_insert_vectors[row_idx * vector_dim + cur_dim];
  }
  __syncthreads();

  // Second Step: Update the adjacency list for the current row with the search neighbors
  // returned by the search kernel.
  // - col_idx < recall_at_k : use the actual search result; pad with fallback if invalid/deleted.
  // - col_idx >= recall_at_k: no search result exists (OOB); pad with fallback to keep graph regular.
  // Reverse graph is rebuilt globally before delete, so we avoid incremental reverse updates here.
  if (static_cast<int64_t>(col_idx) < static_cast<int64_t>(recall_at_k)) {
    auto nbr_val = d_search_neighbors[(row_idx * recall_at_k) + col_idx];
    if (nbr_val >= 0 && nbr_val < params.graph_rows && params.deleted_rows[nbr_val] != 1) {
      // Valid neighbor — write into forward graph.
      params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + col_idx] =
        static_cast<uint32_t>(nbr_val);
    } else if (s_fallback_neighbor != UINT32_MAX) {
      // Invalid or deleted result — pad with fallback.
      params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + col_idx] = s_fallback_neighbor;
    }
    else {
      // Invalid or deleted result and no fallback neighbor found — leave the slot as-is.
       printf("[bang_insert_kernel1] WARNING: no valid neighbor found for inserted row %llu at col_idx %lld; leaving slot as-is.\n",
              static_cast<unsigned long long>(cur_row), static_cast<long long>(col_idx));
    }
    // else: no valid neighbor found at all — leave the slot as-is.
  } else if (s_fallback_neighbor != UINT32_MAX) {
    // Slot has no corresponding search result (graph_cols > recall_at_k) — pad to keep graph regular.
    params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + col_idx] = s_fallback_neighbor;
  }
  else
  {
    // Slot has no corresponding search result and no fallback neighbor found — leave the slot as-is.
    printf("[bang_insert_kernel1] WARNING: no valid neighbor found for inserted row %llu at col_idx %lld; leaving slot as-is.\n",
           static_cast<unsigned long long>(cur_row), static_cast<long long>(col_idx));
  }
  __syncthreads();

  //__syncthreads();
  // if (col_idx == 0) { params.deleted_rows[cur_row] = 0; }
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
    // Skip deleted neighbors — their adjacency lists must not be modified.
    if (params.deleted_rows[cur_row_cur_neighbor] == 1) { continue; }
    // Do not update rows that are also being inserted in this launch.
    if (is_row_in_insert_batch(params, static_cast<uint64_t>(cur_row_cur_neighbor))) 
    {
            // print warning
      printf("[bang_insert_kernel2] WARNING: cur_row %llu has neighbor %u that is also being inserted; skipping reverse-edge update for this neighbor.\n",
             static_cast<unsigned long long>(cur_row), static_cast<unsigned int>(cur_row_cur_neighbor)); 
      continue; 
    }
    bool reverse_edge_added = false;
    for (int64_t col2 = 0; col2 < params.graph_cols; col2++) {
      auto neighbor_of_cur_row_cur_neighbor =
        params.graph[(cur_row_cur_neighbor * static_cast<uint64_t>(params.graph_cols)) + col2];
      // Skip deleted or out-of-range shared-neighbor candidates — matching against them
      // would fire the replacement on a dead node.
      if (static_cast<int64_t>(neighbor_of_cur_row_cur_neighbor) >= params.graph_rows) { continue; }
      if (params.deleted_rows[neighbor_of_cur_row_cur_neighbor] == 1) { continue; }
      // scan the neighbours of cur_row
      for (int64_t entry = 0; entry < params.graph_cols; entry++) {

        if (params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + entry] == 
        neighbor_of_cur_row_cur_neighbor) {
          auto target_offset = (cur_row_cur_neighbor * static_cast<uint64_t>(params.graph_cols)) + col2;
          auto target_ptr = params.graph + target_offset;

          {
            auto previous_neighbor = atomicExch(target_ptr, static_cast<uint32_t>(cur_row));
            if (previous_neighbor != static_cast<uint32_t>(cur_row)) {
              reverse_edge_added = true;
            }

          break;
          }
        } // if (params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + entry] == neighbor_of_cur_row_cur_neighbor)
      }
      if (reverse_edge_added) { break; } // no need to continue scanning cur_row_cur_neighbor if reverse edge is added
    }
  }
  if (col_idx == 0) { params.deleted_rows[cur_row] = 0; }
}
 template<typename T>
__global__ void cuvs_bang_insert_kernel3(InsertKernelParams params)
{
// Compact the reverse graph: remove UINT32_MAX sentinels that were introduced
  // when edges were logically removed (e.g. in kernel2's reverse-edge cleanup).
  // Each block handles one or more rows in a row-stride pattern.
  // Single-thread compaction per row — the array width (reverse_graph_cols) is
  // small enough that sequential work per row is cheaper than a parallel scan.
  auto row_idx = static_cast<int64_t>(blockIdx.x);

  for (int64_t row = row_idx; row < params.graph_rows; row += static_cast<int64_t>(gridDim.x)) {
    if (params.deleted_rows[row] == 1) { continue; }

    // Cap at reverse_graph_cols: reverse_counts can overflow due to atomicAdd past
    // the allocated slot limit (dropped edges during build / incremental updates).
    int64_t count = min(static_cast<int64_t>(params.reverse_counts[row]),
                        params.reverse_graph_cols);
    if (count == 0) { continue; }

    auto* row_ptr = params.reverse_graph +
                    row * static_cast<uint64_t>(params.reverse_graph_cols);

    // In-place compaction: slide valid entries to the front.
    int64_t write_pos = 0;
    for (int64_t read_pos = 0; read_pos < count; ++read_pos) {
      if (row_ptr[read_pos] != UINT32_MAX) {
        row_ptr[write_pos++] = row_ptr[read_pos];
      }
    }
    // Fill the vacated tail with UINT32_MAX to keep the region clean.
    for (int64_t p = write_pos; p < count; ++p) {
      row_ptr[p] = UINT32_MAX;
    }
    // Update reverse_counts to reflect only valid entries.
    params.reverse_counts[row] = static_cast<uint32_t>(write_pos);
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
                                    int recall_at_k,
                                    uint32_t* reverse_graph,
                                    int64_t reverse_graph_cols,
                                    uint32_t* reverse_counts)
{
  if (num_ids == 0) { return; }
  if (graph_rows <= 0 || graph_cols <= 0) { return; }
  
  auto* mutable_graph_ptr = const_cast<uint32_t*>(graph_ptr);
  InsertKernelParams params{nullptr, num_ids, mutable_graph_ptr, graph_rows, graph_cols,
                            static_cast<uint8_t*>(d_deleted_rows->data()),
                            static_cast<float*>(d_distance_threshold1->data()),
                            static_cast<float*>(d_distance_threshold2->data()),
                            reverse_graph,
                            reverse_graph_cols,
                            reverse_counts};

  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&params.row_num), num_ids * sizeof(uint64_t)));
  RAFT_CUDA_TRY(cudaMemcpy(params.row_num, ids, num_ids * sizeof(uint64_t), cudaMemcpyHostToDevice));

  uint32_t threads = static_cast<uint32_t>(graph_cols > 1024 ? 1024 : graph_cols); // The kernel expect this to be graph degree
  auto blocks      = static_cast<uint32_t>(num_ids);

  // copy the h_insert_vectrs to device
  
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
}  

__global__ void cuvs_bang_delete_kernel2_reversegraph(DeleteKernelParams params)
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
  if (!params.reverse_row_ptr) { return; }

  // find a guaranteed non-deleted neighbour of cur_row to replace deleted-row references.
  // Only thread 0 scans the adjacency list; all others wait at __syncthreads().
  // Using uint32_t to match the graph element type; UINT32_MAX is the "not found" sentinel.
  __shared__ uint32_t backup_replacement_neighbor;
  __shared__ uint32_t s_row_start;
  __shared__ int64_t  s_in_count;
  if (col_idx == 0) {
    s_row_start = params.reverse_row_ptr[cur_row];
    auto row_end = params.reverse_row_ptr[cur_row + 1];
    s_in_count = static_cast<int64_t>(row_end) - static_cast<int64_t>(s_row_start);

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

  // Scan CSR reverse row [row_start, row_end) to find IN nodes of cur_row.
  for (int64_t col = col_idx; col < s_in_count; col += blockDim.x) {
    uint32_t in_node = params.reverse_graph[s_row_start + static_cast<uint32_t>(col)];
    if (static_cast<int64_t>(in_node) >= params.graph_rows || params.deleted_rows[in_node] == 1) { continue; } // skip invalid or deleted in_nodes
    // scan the adjacency list of in_node to find cur_row and replace it
    for (int64_t c = 0; c < params.graph_cols; ++c) {
      auto off      = in_node * static_cast<uint64_t>(params.graph_cols) + c;
      auto neighbor = params.graph[off];
      if (neighbor == cur_row) {
        // Determine the replacement node and write it into the forward graph.
        // Priority 1: cur_row's neighbor at the same column (spatially close).
        // Priority 2: backup_replacement_neighbor (first live node in cur_row's adj list).
        // Priority 3: in_node's own live neighbors — used when the entire deleted
        //             row's adj list is within the same deletion batch (spatial cluster).
        uint32_t replacement = UINT32_MAX;
        uint32_t candidate = params.graph[cur_row * static_cast<uint64_t>(params.graph_cols) + c];
        if (static_cast<int64_t>(candidate) < params.graph_rows && params.deleted_rows[candidate] != 1) {
          replacement = candidate;
        } else if (backup_replacement_neighbor != UINT32_MAX) {
          replacement = backup_replacement_neighbor;
        } else {
          // Both primary fallbacks are deleted — scan in_node's own adj list for any live node.
          for (int64_t c_fb = 0; c_fb < params.graph_cols; ++c_fb) {
            uint32_t fb = params.graph[in_node * static_cast<uint64_t>(params.graph_cols) + c_fb];
            if (static_cast<int64_t>(fb) < params.graph_rows && fb != cur_row &&
                params.deleted_rows[fb] != 1) {
              replacement = fb;
              break;
            }
          }
          if (replacement == UINT32_MAX) {
            printf("[bang_kernel] WARNING: No live replacement found for in_node %llu -> cur_row %llu; edge left unchanged.\n",
                   (unsigned long long)in_node, (unsigned long long)cur_row);
          }
        }

        if (replacement != UINT32_MAX) {
          // forward graph update
          params.graph[off] = replacement;
        }
        else
        {
          printf("[bang_kernel] WARNING: No live replacement found for in_node %llu -> cur_row %llu; edge left unchanged.\n",
                 (unsigned long long)in_node, (unsigned long long)cur_row);
        }
    }
  }
}
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

void launch_cuvs_bang_build_reverse_graph_kernel(const uint32_t* graph,
                                                 int64_t graph_rows,
                                                 int64_t graph_cols,
                                                 uint32_t* reverse_graph,
                                                 int64_t reverse_graph_cols,
                                                 const uint8_t* deleted_rows,
                                                 uint32_t* reverse_row_ptr,
                                                 uint32_t* reverse_counts)
{
  if (!graph || !reverse_graph || !reverse_row_ptr || !reverse_counts) { return; }
  if (graph_rows <= 0 || graph_cols <= 0 || reverse_graph_cols <= 0) { return; }

  auto reverse_capacity = static_cast<uint64_t>(graph_rows) * static_cast<uint64_t>(reverse_graph_cols);
  RAFT_CUDA_TRY(cudaMemset(reverse_graph, 0xFF, reverse_capacity * sizeof(uint32_t)));

  uint32_t* d_skipped_invalid_dst = nullptr;
  uint32_t* d_skipped_deleted_src = nullptr;
  uint32_t* d_skipped_deleted_dst = nullptr;
  RAFT_CUDA_TRY(cudaMemset(reverse_counts, 0, static_cast<size_t>(graph_rows) * sizeof(uint32_t)));
  RAFT_CUDA_TRY(cudaMemset(reverse_row_ptr, 0, static_cast<size_t>(graph_rows + 1) * sizeof(uint32_t)));

  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_skipped_invalid_dst), sizeof(uint32_t)));
  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_skipped_deleted_src), sizeof(uint32_t)));
  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_skipped_deleted_dst), sizeof(uint32_t)));
  RAFT_CUDA_TRY(cudaMemset(d_skipped_invalid_dst, 0, sizeof(uint32_t)));
  RAFT_CUDA_TRY(cudaMemset(d_skipped_deleted_src, 0, sizeof(uint32_t)));
  RAFT_CUDA_TRY(cudaMemset(d_skipped_deleted_dst, 0, sizeof(uint32_t)));

  uint32_t threads = static_cast<uint32_t>(graph_cols > 1024 ? 1024 : graph_cols);
  if (threads == 0) {
    RAFT_CUDA_TRY(cudaFree(d_skipped_invalid_dst));
    RAFT_CUDA_TRY(cudaFree(d_skipped_deleted_src));
    RAFT_CUDA_TRY(cudaFree(d_skipped_deleted_dst));
    return;
  }

  constexpr uint32_t kMaxBlocks = 10000;
  auto blocks = static_cast<uint32_t>(graph_rows < static_cast<int64_t>(kMaxBlocks)
                                        ? graph_rows
                                        : static_cast<int64_t>(kMaxBlocks));
  if (blocks == 0) {
    RAFT_CUDA_TRY(cudaFree(d_skipped_invalid_dst));
    RAFT_CUDA_TRY(cudaFree(d_skipped_deleted_src));
    RAFT_CUDA_TRY(cudaFree(d_skipped_deleted_dst));
    return;
  }

  cuvs_bang_count_reverse_graph_kernel<<<blocks, threads>>>(graph,
                                                            graph_rows,
                                                            graph_cols,
                                                            deleted_rows,
                                                            reverse_counts,
                                                            d_skipped_invalid_dst,
                                                            d_skipped_deleted_src,
                                                            d_skipped_deleted_dst);

  // Build canonical CSR row pointers:
  //   row_ptr[0] = 0
  //   row_ptr[i+1] = sum(counts[0..i])
  // We use InclusiveSum into row_ptr+1 to directly produce the inclusive prefix.
  void* d_scan_tmp = nullptr;
  size_t scan_tmp_bytes = 0;
  RAFT_CUDA_TRY(cub::DeviceScan::InclusiveSum(
    d_scan_tmp, scan_tmp_bytes, reverse_counts, reverse_row_ptr + 1, static_cast<int>(graph_rows)));
  RAFT_CUDA_TRY(cudaMalloc(&d_scan_tmp, scan_tmp_bytes));
  RAFT_CUDA_TRY(cub::DeviceScan::InclusiveSum(
    d_scan_tmp, scan_tmp_bytes, reverse_counts, reverse_row_ptr + 1, static_cast<int>(graph_rows)));
  RAFT_CUDA_TRY(cudaFree(d_scan_tmp));

 #if 0
  // Debug: print total reverse edges from CSR tail pointer.
  uint32_t h_total_reverse_edges = 0;
  RAFT_CUDA_TRY(cudaMemcpy(&h_total_reverse_edges,
                           reverse_row_ptr + graph_rows,
                           sizeof(uint32_t),
                           cudaMemcpyDeviceToHost));  
  std::printf("[bang] total_reverse_edges=%u\n", h_total_reverse_edges);
#endif
  uint32_t* d_reverse_write_offsets = nullptr;
  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_reverse_write_offsets),
                           static_cast<size_t>(graph_rows) * sizeof(uint32_t)));
  RAFT_CUDA_TRY(cudaMemcpy(d_reverse_write_offsets,
                           reverse_row_ptr,
                           static_cast<size_t>(graph_rows) * sizeof(uint32_t),
                           cudaMemcpyDeviceToDevice));

  cuvs_bang_fill_reverse_graph_kernel<<<blocks, threads>>>(graph,
                                                           graph_rows,
                                                           graph_cols,
                                                           deleted_rows,
                                                           reverse_graph,
                                                           d_reverse_write_offsets);
  RAFT_CUDA_TRY(cudaFree(d_reverse_write_offsets));

  RAFT_CUDA_TRY(cudaPeekAtLastError());
  RAFT_CUDA_TRY(cudaDeviceSynchronize());
#if 0
  uint32_t h_skipped_invalid_dst = 0;
  uint32_t h_skipped_deleted_src = 0;
  uint32_t h_skipped_deleted_dst = 0;
  RAFT_CUDA_TRY(cudaMemcpy(
    &h_skipped_invalid_dst, d_skipped_invalid_dst, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(
    &h_skipped_deleted_src, d_skipped_deleted_src, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  RAFT_CUDA_TRY(cudaMemcpy(
    &h_skipped_deleted_dst, d_skipped_deleted_dst, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  // Debug consistency check:
  // counted reverse edges must match total valid forward edges consumed by count kernel.
  auto total_forward_slots = static_cast<uint64_t>(graph_rows) * static_cast<uint64_t>(graph_cols);
  auto skipped_total       = static_cast<uint64_t>(h_skipped_deleted_src) +
                       static_cast<uint64_t>(h_skipped_invalid_dst) +
                       static_cast<uint64_t>(h_skipped_deleted_dst);
  auto expected_reverse_edges = total_forward_slots >= skipped_total
                                  ? (total_forward_slots - skipped_total)
                                  : 0ULL;
  if (expected_reverse_edges != static_cast<uint64_t>(h_total_reverse_edges)) {
    std::printf(
      "[bang][WARN] reverse edge mismatch: csr_total=%u expected_valid_forward_edges=%llu (rows=%lld cols=%lld skipped_deleted_src=%u skipped_invalid_dst=%u skipped_deleted_dst=%u)\n",
      h_total_reverse_edges,
      static_cast<unsigned long long>(expected_reverse_edges),
      static_cast<long long>(graph_rows),
      static_cast<long long>(graph_cols),
      h_skipped_deleted_src,
      h_skipped_invalid_dst,
      h_skipped_deleted_dst);
  }

  // Consolidated post-build diagnostic: check reverse edge totals against
  // allocated reverse-graph capacity (in edge slots).
  auto reverse_capacity_edges = reverse_capacity;
  if (static_cast<uint64_t>(h_total_reverse_edges) > reverse_capacity_edges) {
    std::printf(
      "[bang][WARN] reverse capacity overflow risk: csr_total=%u capacity_edges=%llu (rows=%lld rev_cols=%lld)\n",
      h_total_reverse_edges,
      static_cast<unsigned long long>(reverse_capacity_edges),
      static_cast<long long>(graph_rows),
      static_cast<long long>(reverse_graph_cols));
  }
  if (expected_reverse_edges > reverse_capacity_edges) {
    std::printf(
      "[bang][WARN] reverse capacity too small for expected edges: expected_valid_forward_edges=%llu capacity_edges=%llu (rows=%lld cols=%lld rev_cols=%lld)\n",
      static_cast<unsigned long long>(expected_reverse_edges),
      static_cast<unsigned long long>(reverse_capacity_edges),
      static_cast<long long>(graph_rows),
      static_cast<long long>(graph_cols),
      static_cast<long long>(reverse_graph_cols));
  }
  std::printf(
    "[bang][post_diag] reverse edges: csr_total=%u expected_valid_forward_edges=%llu capacity_edges=%llu\n",
    h_total_reverse_edges,
    static_cast<unsigned long long>(expected_reverse_edges),
    static_cast<unsigned long long>(reverse_capacity_edges));

  if (h_skipped_invalid_dst > 0 || h_skipped_deleted_src > 0 || h_skipped_deleted_dst > 0) {
    std::printf(
      "[bang] reverse_graph build skipped invalid_dst=%u deleted_src=%u deleted_dst=%u (rows=%lld fwd_cols=%lld rev_cols=%lld)\n",
      h_skipped_invalid_dst,
      h_skipped_deleted_src,
      h_skipped_deleted_dst,
      static_cast<long long>(graph_rows),
      static_cast<long long>(graph_cols),
      static_cast<long long>(reverse_graph_cols));
  }

  RAFT_CUDA_TRY(cudaFree(d_skipped_invalid_dst));
  RAFT_CUDA_TRY(cudaFree(d_skipped_deleted_src));
  RAFT_CUDA_TRY(cudaFree(d_skipped_deleted_dst));
#endif
}


void launch_cuvs_bang_delete_kernel(const uint64_t* ids,
                                    size_t num_ids,
                                    const uint32_t* graph_ptr,
                                    int64_t graph_rows,
                                    int64_t graph_cols,
                                    const std::shared_ptr<rmm::device_buffer>& d_deleted_rows,
                                    uint32_t* reverse_graph,
                                    int64_t reverse_graph_cols,
                                    uint32_t* reverse_row_ptr,
                                    uint32_t* reverse_counts)
{
  if (num_ids == 0) { return; }
  if (graph_rows <= 0 || graph_cols <= 0) { return; }
  auto* mutable_graph_ptr = const_cast<uint32_t*>(graph_ptr);
  DeleteKernelParams params{nullptr, num_ids, mutable_graph_ptr, graph_rows, graph_cols,
                            static_cast<uint8_t*>(d_deleted_rows->data()),
                            reverse_graph,
                            reverse_graph_cols,
                            reverse_row_ptr,
                            reverse_counts};

  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&params.row_num), num_ids * sizeof(uint64_t)));
  RAFT_CUDA_TRY(cudaMemcpy(params.row_num, ids, num_ids * sizeof(uint64_t), cudaMemcpyHostToDevice));

  // Reverse graph is only consumed by delete kernel2. Rebuild it from the
  // current forward graph immediately before delete processing so kernel2 sees
  // a fresh, consistent in-edge view.
  launch_cuvs_bang_build_reverse_graph_kernel(mutable_graph_ptr,
                                              graph_rows,
                                              graph_cols,
                                              reverse_graph,
                                              reverse_graph_cols,
                                              static_cast<uint8_t*>(d_deleted_rows->data()),
                                              reverse_row_ptr,
                                              reverse_counts);

  constexpr uint32_t threads = 1024;
  auto blocks                = static_cast<uint32_t>((num_ids + threads - 1) / threads);

  cuvs_bang_delete_kernel1<<<blocks, threads>>>(params);

  auto bfs_threads = static_cast<uint32_t>(graph_cols > 1024 ? 1024 : graph_cols);

  cuvs_bang_delete_kernel2_reversegraph<<<num_ids, bfs_threads>>>(params);
  
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  RAFT_CUDA_TRY(cudaDeviceSynchronize());
#if 0
  // Global cleanup to prevent stale references to older deleted rows from
  // accumulating across batched delete iterations.
  cuvs_bang_sanitize_deleted_neighbors(mutable_graph_ptr,
                                       graph_rows,
                                       graph_cols,
                                       static_cast<uint8_t*>(d_deleted_rows->data()));

  // Debug verifier: detect surviving edges from live rows to deleted destinations.
  cuvs_bang_verify_no_deleted_neighbors(mutable_graph_ptr,
                                        graph_rows,
                                        graph_cols,
                                        static_cast<uint8_t*>(d_deleted_rows->data()));
#endif
  std::printf("[bang] kernel processed=%zu ids (SKIPPED SANITIZE)\n", num_ids);

  RAFT_CUDA_TRY(cudaFree(params.row_num));
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
                                                     int recall_at_k,
                                                     uint32_t* reverse_graph,
                                                     int64_t reverse_graph_cols,
                                                     uint32_t* reverse_counts);

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
                                                    int recall_at_k,
                                                    uint32_t* reverse_graph,
                                                    int64_t reverse_graph_cols,
                                                    uint32_t* reverse_counts);

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
                                                      int recall_at_k,
                                                      uint32_t* reverse_graph,
                                                      int64_t reverse_graph_cols,
                                                      uint32_t* reverse_counts);

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
                                                       int recall_at_k,
                                                       uint32_t* reverse_graph,
                                                       int64_t reverse_graph_cols,
                                                       uint32_t* reverse_counts);


}  // namespace cuvs::bench::detail