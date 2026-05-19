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

  // First Step: copy d_insert_vectors to the appropriate location in the dataset 
  uint32_t dims_per_thread =  (vector_dim + blockDim.x - 1) / blockDim.x; // ceiling division to cover all dimensions

  // Each thread will copy dims_per_thread dimensions of the vector. Calculate the start and end dimension index for this thread.
  for (uint32_t cur_dim = threadIdx.x * dims_per_thread; cur_dim < vector_dim && cur_dim < (threadIdx.x + 1) * dims_per_thread; cur_dim++) {
    d_dataset_vectors[cur_row * vector_dim + cur_dim] = d_insert_vectors[row_idx * vector_dim + cur_dim];
  }

  // Second Step: Update the adjacency list for the current row with the search neighbors returned by the search kernel and mark the distance thresholds for this row based on the search distances returned by the search kernel.
  params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + col_idx] =
                         d_search_neighbors[(row_idx * recall_at_k) + col_idx]; // update adjacency list with search neighbors
  
  
  // Third Step: for each neighbor identify the neighbors
  // Mental mode of graph: row#|neighbor1 neighbor2 neighbor3 ...
  auto cur_row_cur_neighbor = params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + col_idx];
  for (int64_t col = 0; col < params.graph_cols; col++) {
    auto neighbor_of_cur_row_cur_neighbor = params.graph[(cur_row_cur_neighbor * static_cast<uint64_t>(params.graph_cols)) + col];
    // scan the neighbours of cur_row
    for (int64_t entry = 0; entry < params.graph_cols; entry++) {
      if (params.graph[(cur_row * static_cast<uint64_t>(params.graph_cols)) + entry] == neighbor_of_cur_row_cur_neighbor) {
        params.graph[(cur_row_cur_neighbor * static_cast<uint64_t>(params.graph_cols)) + col] = cur_row; // add cur_row as a neighbor to the neighbors of cur_row's neighbors
        return; 
      }
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
  T* d_insert_vectors;
  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&d_insert_vectors), num_ids * vector_dim * sizeof(T)));
  RAFT_CUDA_TRY(cudaMemcpy(d_insert_vectors, h_insert_vectors, num_ids * vector_dim * sizeof(T), cudaMemcpyHostToDevice));

  cuvs_bang_insert_kernel1<<<blocks, threads>>>(params, d_insert_vectors,d_dataset_vectors, 
    vector_dim, recall_at_k, d_search_neighbors, d_search_distances );
  //cuvs_bang_insert_kernel2<<<blocks, threads>>>(d_insert_vectors, params);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  std::printf("[bang] insert kernel processed=%zu ids\n", num_ids);

  RAFT_CUDA_TRY(cudaFree(params.row_num));  
  RAFT_CUDA_TRY(cudaFree(d_insert_vectors));
}

__global__ void cuvs_bang_delete_kernel1(DeleteKernelParams params)
{
  // Print graph dimensions once from thread 0 and also first and last row_num to be deleted
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[bang_kernel] graph_rows=%lld graph_cols=%lld num_ids=%llu first_row=%llu last_row=%llu\n",
           (long long)params.graph_rows, (long long)params.graph_cols, (unsigned long long)params.num_ids,
           (unsigned long long)params.row_num[0], (unsigned long long)params.row_num[params.num_ids - 1]);
  }

  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= params.num_ids) { return; }

  // identify the current row to be processed by this thread
  auto cur_row = params.row_num[idx];

  if (cur_row >= static_cast<uint64_t>(params.graph_rows)) { return; }

  // mark the current row as deleted 
  // ToDo: do this SET'ing in a separate kernel before launching this kernel for global barrier
  params.deleted_rows[cur_row] = 1;
}  

__global__ void cuvs_bang_delete_kernel2(DeleteKernelParams params)
{
  // Print graph dimensions once from thread 0 and also first and last row_num to be deleted
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[bang_kernel] graph_rows=%lld graph_cols=%lld num_ids=%llu first_row=%llu last_row=%llu\n",
           (long long)params.graph_rows, (long long)params.graph_cols, (unsigned long long)params.num_ids,
           (unsigned long long)params.row_num[0], (unsigned long long)params.row_num[params.num_ids - 1]);
  }

  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= params.num_ids) { return; }

  // identify the current row to be processed by this thread
  auto cur_row = params.row_num[idx];

  if (cur_row >= static_cast<uint64_t>(params.graph_rows)) { return; }

  if (params.deleted_rows[cur_row] != 1) { return; } // skip if not marked deleted

  uint32_t circular_counter = 0;
  // scan the entire graph to find the occurrence of cur_row
  for (int64_t row = 0; row < params.graph_rows; row++) {
    if (params.deleted_rows[row] == 1)  { continue; } // skip deleted rows

    for (int64_t col = 0; col < params.graph_cols; col++) {
      auto neighbor = params.graph[row * static_cast<uint64_t>(params.graph_cols) + col];
      if (neighbor == cur_row) {
        for (uint32_t temp_iter = 0; temp_iter < params.graph_cols; temp_iter++){
        uint32_t cur_row_neighbour = params.graph[cur_row * static_cast<uint64_t>(params.graph_cols) + circular_counter];
        if (params.deleted_rows[cur_row_neighbour] != 1){
          // Replace the deleted-node reference with a live neighbor from the deleted row.
          params.graph[(row * static_cast<uint64_t>(params.graph_cols)) + col] = cur_row_neighbour;
          if (++circular_counter >= params.graph_cols ) circular_counter = 0;
          break; // go to next neighbor
        } else {
            if (++circular_counter >= params.graph_cols ) circular_counter = 0;
        }
      }
       
      
      } // end if neighbor match
    }  // end for cols  
  } // end for rows

    
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
    
}




void cuvs_bang_kernel_init()
{

}


void launch_cuvs_bang_delete_kernel(const uint64_t* ids,
                                    size_t num_ids,
                                    const uint32_t* graph_ptr,
                                    int64_t graph_rows,
                                    int64_t graph_cols,
                                    const std::shared_ptr<rmm::device_buffer>& d_deleted_rows)
{
  if (num_ids == 0) { return; }
  if (graph_rows <= 0 || graph_cols <= 0) { return; }
  
  auto* mutable_graph_ptr = const_cast<uint32_t*>(graph_ptr);
  DeleteKernelParams params{nullptr, num_ids, mutable_graph_ptr, graph_rows, graph_cols,
                            static_cast<uint8_t*>(d_deleted_rows->data())};

  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&params.row_num), num_ids * sizeof(uint64_t)));
  RAFT_CUDA_TRY(cudaMemcpy(params.row_num, ids, num_ids * sizeof(uint64_t), cudaMemcpyHostToDevice));

  constexpr uint32_t threads = 1024;
  auto blocks                = static_cast<uint32_t>((num_ids + threads - 1) / threads);

  cuvs_bang_delete_kernel1<<<blocks, threads>>>(params);
  cuvs_bang_delete_kernel2<<<blocks, threads>>>(params);
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