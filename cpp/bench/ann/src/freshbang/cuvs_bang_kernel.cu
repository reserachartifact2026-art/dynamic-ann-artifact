/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "../cuvs/cuvs_cagra_wrapper.h"

#include <cuda_runtime.h>
#include <raft/util/cudart_utils.hpp>

#include <cstdio>

namespace cuvs::bench::detail {
  typedef struct _DeleteKernelParams {
    uint64_t* row_num; // rows to be deleted
    size_t num_ids;
    uint32_t* graph;
    int64_t graph_rows;
    int64_t graph_cols;
    bool* is_deleted_row;
  } DeleteKernelParams;
namespace {
__global__ void cuvs_bang_delete_kernel(DeleteKernelParams params)
{
  // Print graph dimensions once from thread 0
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf("[bang_kernel] graph_rows=%lld graph_cols=%lld num_ids=%llu\n",
           (long long)params.graph_rows, (long long)params.graph_cols, (unsigned long long)params.num_ids);
  }

  auto idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= params.num_ids) { return; }

  auto row = params.row_num[idx];
  if (row >= static_cast<uint64_t>(params.graph_rows)) { return; }

  params.is_deleted_row[row] = true;
  //for debugging, lets write the row id itself as the adjacency value
  const auto graph_cols_u64 = static_cast<uint64_t>(params.graph_cols);
  for (int64_t loop_iter = 0; loop_iter < params.graph_cols; loop_iter++) {
    params.graph[row * graph_cols_u64 + static_cast<uint64_t>(loop_iter)] = static_cast<uint32_t>(row);
  }
}
}  // namespace

void launch_cuvs_bang_delete_kernel(const uint64_t* ids,
                                    size_t num_ids,
                                    const uint32_t* graph_ptr,
                                    int64_t graph_rows,
                                    int64_t graph_cols)
{
  if (num_ids == 0) { return; }
  if (graph_rows <= 0 || graph_cols <= 0) { return; }
  
  auto* mutable_graph_ptr = const_cast<uint32_t*>(graph_ptr);
  DeleteKernelParams params{nullptr, num_ids, mutable_graph_ptr, graph_rows, graph_cols, nullptr};

  RAFT_CUDA_TRY(cudaMalloc(reinterpret_cast<void**>(&params.row_num), num_ids * sizeof(uint64_t)));
  RAFT_CUDA_TRY(cudaMemcpy(params.row_num, ids, num_ids * sizeof(uint64_t), cudaMemcpyHostToDevice));

  RAFT_CUDA_TRY(
    cudaMalloc(reinterpret_cast<void**>(&params.is_deleted_row), graph_rows * sizeof(bool)));
  RAFT_CUDA_TRY(cudaMemset(params.is_deleted_row, 0, graph_rows * sizeof(bool)));

  constexpr uint32_t threads = 256;
  auto blocks                = static_cast<uint32_t>((num_ids + threads - 1) / threads);
  cuvs_bang_delete_kernel<<<blocks, threads>>>(params);
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  RAFT_CUDA_TRY(cudaDeviceSynchronize());

  std::printf("[bang] kernel processed=%zu ids\n", num_ids);

  RAFT_CUDA_TRY(cudaFree(params.is_deleted_row));
  RAFT_CUDA_TRY(cudaFree(params.row_num));
}
}  // namespace cuvs::bench::detail