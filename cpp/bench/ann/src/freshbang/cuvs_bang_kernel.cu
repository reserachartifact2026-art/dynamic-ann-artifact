/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../cuvs/cuvs_cagra_wrapper.h"

#include <cuda_runtime.h>
#include <raft/util/cudart_utils.hpp>

#include <cstdio>
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
        if (cur_row_neighbour >= static_cast<uint64_t>(params.graph_rows)) {
          if (++circular_counter >= params.graph_cols ) circular_counter = 0;
          continue;
        }
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
        if (cur_neighbour >= static_cast<uint64_t>(params.graph_rows)) { continue; }
        if (params.deleted_rows[cur_neighbour] != 1) {
          // Copy adjacency list of a live neighbor into the deleted row.
          for (int64_t col = 0; col < params.graph_cols; col++) {
            uint32_t candidate = params.graph[cur_neighbour * static_cast<uint64_t>(params.graph_cols) + col];
            if (candidate >= static_cast<uint64_t>(params.graph_rows)) {
              params.graph[cur_row * static_cast<uint64_t>(params.graph_cols) + col] = 1;
              continue;
            }
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
                                    std::shared_ptr<rmm::device_buffer> d_deleted_rows)
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
}  // namespace cuvs::bench::detail