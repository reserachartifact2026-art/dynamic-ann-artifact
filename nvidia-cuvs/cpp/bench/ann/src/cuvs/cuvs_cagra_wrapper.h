/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../../src/neighbors/detail/cagra/utils.hpp"
#include "../common/ann_types.hpp"
#include "../common/cuda_huge_page_resource.hpp"
#include "cuvs_ann_bench_utils.h"
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/composite/merge.hpp>
#include <cuvs/neighbors/dynamic_batching.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <memory>
#include <optional>
#include <raft/util/integer_utils.hpp>
//#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

#define DIST_THREDHOLD1_INDEX 15
#define DIST_THREDHOLD2_INDEX 31

namespace cuvs::bench {

namespace detail {
template <typename T>
void launch_cuvs_bang_insert_kernel(const uint64_t* ids,
                                    size_t num_ids,
                                    const uint32_t* graph,
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
                                    uint32_t* reverse_counts);

void launch_cuvs_bang_delete_kernel(const uint64_t* ids,
                                    size_t num_ids,
                                    const uint32_t* graph,
                                    int64_t graph_rows,
                                    int64_t graph_cols,
                                    const std::shared_ptr<rmm::device_buffer>& d_deleted_rows,
                                    uint32_t* reverse_graph,
                                    int64_t reverse_graph_cols,
                                    uint32_t* reverse_row_ptr,
                                    uint32_t* reverse_counts);

void launch_cuvs_bang_build_reverse_graph_kernel(const uint32_t* graph,
                                                 int64_t graph_rows,
                                                 int64_t graph_cols,
                                                 uint32_t* reverse_graph,
                                                 int64_t reverse_graph_cols,
                                                 const uint8_t* deleted_rows,
                                                 uint32_t* reverse_row_ptr,
                                                 uint32_t* reverse_counts);

}

enum class AllocatorType { kHostPinned, kHostHugePage, kDevice };
enum class CagraBuildAlgo { kAuto, kIvfPq, kNnDescent };
enum class CagraMergeType { kPhysical, kLogical };

template <typename T, typename IdxT>
class cuvs_cagra : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;
  using algo<T>::dim_;
  using algo<T>::metric_;

  struct search_param : public search_param_base {
    cuvs::neighbors::cagra::search_params p;
    float refine_ratio;
    AllocatorType graph_mem   = AllocatorType::kDevice;
    AllocatorType dataset_mem = AllocatorType::kDevice;
    [[nodiscard]] auto needs_dataset() const -> bool override { return true; }
    /* Dynamic batching */
    bool dynamic_batching = false;
    int64_t dynamic_batching_k;
    int64_t dynamic_batching_max_batch_size     = 4;
    double dynamic_batching_dispatch_timeout_ms = 0.01;
    size_t dynamic_batching_n_queues            = 8;
    bool dynamic_batching_conservative_dispatch = false;
  };

  struct build_param {
    // The optimal defaults depend on the dataset shape and thus only available once the build
    // function is called.
    using dataset_dependent_params = std::function<cuvs::neighbors::cagra::index_params(
      raft::matrix_extent<int64_t>, cuvs::distance::DistanceType)>;
    dataset_dependent_params cagra_params;
    size_t num_dataset_splits = 1;
    CagraMergeType merge_type = CagraMergeType::kPhysical;
  };

  struct insert_param {
    bool persist           = false;
    uint32_t max_chunk_size = 0;
  };

  cuvs_cagra(Metric metric, int rows, int dim, const build_param& param, int concurrent_searches = 1)
    : algo<T>(metric, dim),
      index_params_(param),

      dataset_(std::make_shared<raft::device_matrix<T, int64_t, raft::row_major>>(
        std::move(raft::make_device_matrix<T, int64_t>(handle_, 0, 0)))),
      graph_(std::make_shared<raft::device_matrix<IdxT, int64_t, raft::row_major>>(
        std::move(raft::make_device_matrix<IdxT, int64_t>(handle_, 0, 0)))),
      reverse_graph_(std::make_shared<raft::device_matrix<IdxT, int64_t, raft::row_major>>(
        std::move(raft::make_device_matrix<IdxT, int64_t>(handle_, 0, 0)))),
      input_dataset_v_(
        std::make_shared<raft::device_matrix_view<const T, int64_t, raft::row_major>>(
          nullptr, 0, 0)),
      rows_(rows)    
  {
    // rows can be ZRRO via legacy c'tor (cuvs benchmark flow)
    auto initial_rows = rows > 0 ? static_cast<size_t>(rows_) : size_t{0};
    d_deleted_rows_   = std::make_shared<rmm::device_buffer>(
      initial_rows * sizeof(uint8_t), handle_.get_sync_stream(), get_mr(AllocatorType::kDevice));
    // all rows are deemed to be deleted at this point  
    RAFT_CUDA_TRY(cudaMemsetAsync(
      d_deleted_rows_->data(), 1, initial_rows * sizeof(uint8_t), handle_.get_sync_stream()));
      //sync after memset to ensure deleted rows are marked before any search/build operation
    handle_.get_sync_stream().synchronize();
    d_dist_threshold1_ = std::make_shared<rmm::device_buffer>(
    sizeof(float) * initial_rows, handle_.get_sync_stream(), get_mr(AllocatorType::kDevice));
    d_dist_threshold2_ = std::make_shared<rmm::device_buffer>(
    sizeof(float) * initial_rows, handle_.get_sync_stream(), get_mr(AllocatorType::kDevice));

  }
// legacy constructor used by cuvs benchmark flow
  cuvs_cagra(Metric metric, int dim, const build_param& param, int concurrent_searches = 1)
    : cuvs_cagra(metric, 0, dim, param, concurrent_searches)
  {
  }

  void build(const T* dataset, size_t nrow) final;
  void preprocess_built_index();

  void set_search_param(const search_param_base& param, const void* filter_bitset) override;

  void set_search_dataset(const T* dataset, size_t nrow) override;

  void set_insert_param(const insert_param& param);
  void insert(const T* vectors, size_t num_vectors, const uint64_t* ids = nullptr) override;
  void insert_wait(const T* vectors, size_t num_vectors, const uint64_t* ids = nullptr);
  void insert_wait1(const T* vectors, size_t num_vectors, const uint64_t* ids = nullptr);
  void delete_vectors(const uint64_t* ids, size_t num_ids) override;
  void set_insert_param_from_json(const nlohmann::json& conf) override;

  inline bool has_common_neighbour(
    IdxT nid,
    IdxT new_row_id,
    const std::vector<IdxT>& graph,
    int64_t cols,
    int64_t new_rows,
    IdxT& replace_candidate) const;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;
  void search_ex(const T* queries,
                 int batch_size,
                 int k,
                 algo_base::index_type* neighbors,
                 float* distances,
                 int64_t* ids) const override;
  void search_base(const T* queries,
                   int batch_size,
                   int k,
                   algo_base::index_type* neighbors,
                   float* distances) const;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream();
  }

  [[nodiscard]] auto uses_stream() const noexcept -> bool override
  {
    // If the algorithm uses persistent kernel, the CPU has to synchronize by the end of computing
    // the result. Hence it guarantees the benchmark CUDA stream is empty by the end of the
    // execution. Hence we inform the benchmark to not waste the time on recording & synchronizing
    // the event.
    return !search_params_.persistent;
  }

  // to enable dataset access from GPU memory
  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHostMmap;
    property.query_memory_type   = MemoryType::kDevice;
    return property;
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  void set_build_output_file(const std::string& file) override { graph_file_ = file; }
  void save_to_hnswlib(const std::string& file) const;
  std::unique_ptr<algo<T>> copy() override;

  auto get_index() const -> const cuvs::neighbors::cagra::index<T, IdxT>* { return index_.get(); }

 private:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  rmm::mr::pinned_host_memory_resource mr_pinned_;
  raft::mr::cuda_huge_page_resource mr_huge_page_;
  AllocatorType graph_mem_{AllocatorType::kDevice};
  AllocatorType dataset_mem_{AllocatorType::kDevice};
  float refine_ratio_;
  build_param index_params_;
  bool need_dataset_update_{true};
  cuvs::neighbors::cagra::search_params search_params_;
  std::shared_ptr<cuvs::neighbors::cagra::index<T, IdxT>> index_;
  std::shared_ptr<raft::device_matrix<IdxT, int64_t, raft::row_major>> graph_;
  std::shared_ptr<raft::device_matrix<T, int64_t, raft::row_major>> dataset_;
  std::shared_ptr<raft::device_matrix_view<const T, int64_t, raft::row_major>> input_dataset_v_;

  std::shared_ptr<cuvs::neighbors::dynamic_batching::index<T, algo_base::index_type>>
    dynamic_batcher_;
  cuvs::neighbors::dynamic_batching::search_params dynamic_batcher_sp_{};
  int64_t dynamic_batching_max_batch_size_;
  size_t dynamic_batching_n_queues_;
  bool dynamic_batching_conservative_dispatch_;

  std::shared_ptr<cuvs::neighbors::filtering::base_filter> filter_;
  std::vector<std::shared_ptr<cuvs::neighbors::cagra::index<T, IdxT>>> sub_indices_;
  insert_param insert_params_{};
  bool has_inserted_data_{false};
  std::optional<std::string> graph_file_;
  // START: For delete experiments
  std::shared_ptr<rmm::device_buffer> d_deleted_rows_;
  std::shared_ptr<rmm::device_buffer> d_reverse_row_ptr_;
  std::shared_ptr<rmm::device_buffer> d_reverse_counts_;
  // END: For delete experiments
  // Adding a reverse graph to store the IN neighbors for each node
  std::shared_ptr<raft::device_matrix<IdxT, int64_t, raft::row_major>> reverse_graph_; 
  double reverse_graph_col_scale_{1.0}; // scale factor for reverse graph column size relative to forward graph
  int rows_;
   // dataset size supplied at build time
  int build_rows_;

  std::shared_ptr<rmm::device_buffer> d_dist_threshold1_;
  std::shared_ptr<rmm::device_buffer> d_dist_threshold2_;

  std::shared_ptr<rmm::device_buffer> d_delete_frontier_curr_;
  std::shared_ptr<rmm::device_buffer> d_delete_frontier_next_;
  std::shared_ptr<rmm::device_buffer> d_delete_frontier_size_;
  std::shared_ptr<rmm::device_buffer> d_delete_next_size_;
  int64_t delete_workspace_frontier_capacity_{0};
  size_t delete_workspace_max_ids_{0};

  void ensure_delete_workspace(size_t required_ids, int64_t graph_rows);

  inline void ensure_reverse_counts_buffer(int64_t graph_rows)
  {
    if (graph_rows <= 0) { return; }
    auto required_bytes = static_cast<size_t>(graph_rows) * sizeof(uint32_t);
    if (!d_reverse_counts_ || d_reverse_counts_->size() < required_bytes) {
      d_reverse_counts_ = std::make_shared<rmm::device_buffer>(
        required_bytes, handle_.get_sync_stream(), get_mr(AllocatorType::kDevice));
    }
  }

  inline void ensure_reverse_row_ptr_buffer(int64_t graph_rows)
  {
    if (graph_rows <= 0) { return; }
    auto required_bytes = static_cast<size_t>(graph_rows + 1) * sizeof(uint32_t);
    if (!d_reverse_row_ptr_ || d_reverse_row_ptr_->size() < required_bytes) {
      d_reverse_row_ptr_ = std::make_shared<rmm::device_buffer>(
        required_bytes, handle_.get_sync_stream(), get_mr(AllocatorType::kDevice));
    }
  }

  [[nodiscard]] inline auto get_reverse_graph_cols(int64_t graph_cols) const -> int64_t
  {
    if (graph_cols <= 0) { return 0; }
    auto scaled_cols = static_cast<int64_t>(std::ceil(static_cast<double>(graph_cols) *
                                                      reverse_graph_col_scale_));
    return std::max<int64_t>(graph_cols, scaled_cols);
  }

  inline rmm::device_async_resource_ref get_mr(AllocatorType mem_type)
  {
    switch (mem_type) {
      case (AllocatorType::kHostPinned): return &mr_pinned_;
      case (AllocatorType::kHostHugePage): return &mr_huge_page_;
      default: return rmm::mr::get_current_device_resource();
    }
  }
};

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::build(const T* dataset, size_t nrow)
{
  auto dataset_extents = raft::make_extents<IdxT>(nrow, dim_);
  auto params          = index_params_.cagra_params(dataset_extents, parse_metric_type(metric_));
  std::cout << "[cuvs_cagra::build] Building index with nrow=" << nrow << std::endl;

  auto dataset_view_host =
    raft::make_mdspan<const T, IdxT, raft::row_major, true, false>(dataset, dataset_extents);
  auto dataset_view_device =
    raft::make_mdspan<const T, IdxT, raft::row_major, false, true>(dataset, dataset_extents);
  bool dataset_is_on_host = raft::get_device_for_address(dataset) == -1;
  if (index_params_.num_dataset_splits <= 1) {
    index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(std::move(
      dataset_is_on_host ? cuvs::neighbors::cagra::build(handle_, params, dataset_view_host)
                         : cuvs::neighbors::cagra::build(handle_, params, dataset_view_device)));
  } else {
    IdxT rows_per_split =
      raft::ceildiv<IdxT>(nrow, static_cast<IdxT>(index_params_.num_dataset_splits));
    for (size_t i = 0; i < index_params_.num_dataset_splits; ++i) {
      IdxT start = static_cast<IdxT>(i * rows_per_split);
      if (start >= nrow) break;
      IdxT rows        = std::min(rows_per_split, static_cast<IdxT>(nrow) - start);
      const T* sub_ptr = dataset + static_cast<size_t>(start) * dim_;
      auto sub_host =
        raft::make_host_matrix_view<const T, int64_t, raft::row_major>(sub_ptr, rows, dim_);
      auto sub_dev =
        raft::make_device_matrix_view<const T, int64_t, raft::row_major>(sub_ptr, rows, dim_);

      auto sub_index = cuvs::neighbors::cagra::index<T, IdxT>(handle_, params.metric);
      if (index_params_.merge_type == CagraMergeType::kPhysical) {
        if (dataset_is_on_host) {
          sub_index.update_dataset(handle_, sub_host);
        } else {
          sub_index.update_dataset(handle_, sub_dev);
        }
      }
      if (index_params_.merge_type == CagraMergeType::kLogical) {
        if (dataset_is_on_host) {
          sub_index = cuvs::neighbors::cagra::build(handle_, params, sub_host);
        } else {
          sub_index = cuvs::neighbors::cagra::build(handle_, params, sub_dev);
        }
      }
      auto sub_index_shared =
        std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(std::move(sub_index));
      sub_indices_.push_back(std::move(sub_index_shared));
    }
    if (index_params_.merge_type == CagraMergeType::kPhysical) {
      std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> indices;
      indices.reserve(sub_indices_.size());
      for (auto& ptr : sub_indices_) {
        indices.push_back(ptr.get());
      }

      index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(
        std::move(cuvs::neighbors::cagra::merge(handle_, params, indices)));
    }
  }

  build_rows_ = nrow;
  // if we are coming from legacy constructor
  if (rows_ == 0)
  {
    rows_ = static_cast<int>(nrow);
    d_deleted_rows_   = std::make_shared<rmm::device_buffer>(
      rows_, handle_.get_sync_stream(), get_mr(AllocatorType::kDevice));
    d_dist_threshold1_ = std::make_shared<rmm::device_buffer>(
    sizeof(float) * rows_, handle_.get_sync_stream(), get_mr(AllocatorType::kDevice));
    d_dist_threshold2_ = std::make_shared<rmm::device_buffer>(
    sizeof(float) * rows_, handle_.get_sync_stream(), get_mr(AllocatorType::kDevice));      
  }
  
  auto not_deleted_rows_bytes = static_cast<size_t>(nrow) * sizeof(uint8_t);

  if (not_deleted_rows_bytes > 0) {
    RAFT_CUDA_TRY(cudaMemsetAsync(
      d_deleted_rows_->data(), 0, not_deleted_rows_bytes, handle_.get_sync_stream()));
    raft::resource::sync_stream(handle_);
  }

  
 
}

inline auto allocator_to_string(AllocatorType mem_type) -> std::string
{
  if (mem_type == AllocatorType::kDevice) {
    return "device";
  } else if (mem_type == AllocatorType::kHostPinned) {
    return "host_pinned";
  } else if (mem_type == AllocatorType::kHostHugePage) {
    return "host_huge_page";
  }
  return "<invalid allocator type>";
}

template <typename IdxT>
auto has_common_neighbour(IdxT src_row,
                          IdxT dst_row,
                          const std::vector<IdxT>& host_graph,
                          int64_t cols,
                          int64_t n_rows,
                          IdxT& replace_candidate) -> bool
{
  auto to_i64 = [](IdxT v) { return static_cast<int64_t>(v); };
  auto is_valid_node = [&](IdxT v) {
    auto iv = to_i64(v);
    return iv >= 0 && iv < n_rows;
  };

  for (int64_t i = 0; i < cols; ++i) {
    auto a = host_graph[static_cast<size_t>(to_i64(src_row) * cols + i)];
    if (!is_valid_node(a) || a == src_row || a == dst_row) { continue; }
    for (int64_t j = 0; j < cols; ++j) {
      auto b = host_graph[static_cast<size_t>(to_i64(dst_row) * cols + j)];
      if (a == b) {
        replace_candidate = a;
        return true;
      }
    }
  }
  return false;
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::preprocess_built_index()
{
  if (!index_) { return; }

  auto graph_view = index_->graph();
  auto rows       = graph_view.extent(0);
  auto cols       = graph_view.extent(1);
  if (rows == 0 || cols == 0) { return; }

  if (!dataset_ || dataset_->extent(0) < rows || dataset_->extent(1) < dim_ || rows != rows_) {
    throw std::runtime_error(
      "preprocess_built_index: dataset_ is not initialized or has incompatible extents.");
  }

  auto stream = raft::resource::get_cuda_stream(handle_);

  std::vector<IdxT> host_graph(static_cast<size_t>(rows * cols));
  raft::copy(host_graph.data(), graph_view.data_handle(), host_graph.size(), stream);

  auto dataset_cols = dataset_->extent(1);
  std::vector<T> host_dataset(static_cast<size_t>(rows * dataset_cols));
  raft::copy(host_dataset.data(), dataset_->data_handle(), host_dataset.size(), stream);
  raft::resource::sync_stream(handle_);

  auto distance_type = parse_metric_type(metric_);
  auto row_distance  = [&](int64_t lhs, int64_t rhs) {
    const T* a = host_dataset.data() + static_cast<size_t>(lhs) * static_cast<size_t>(dataset_cols);
    const T* b = host_dataset.data() + static_cast<size_t>(rhs) * static_cast<size_t>(dataset_cols);

    float sum_sq = 0.0f;
    float dot    = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (int64_t d = 0; d < dim_; ++d) {
      float av   = static_cast<float>(a[d]);
      float bv   = static_cast<float>(b[d]);
      float diff = av - bv;
      sum_sq += diff * diff;
      dot += av * bv;
      norm_a += av * av;
      norm_b += bv * bv;
    }

    switch (distance_type) {
      case cuvs::distance::DistanceType::L2SqrtExpanded:
      case cuvs::distance::DistanceType::L2SqrtUnexpanded: return std::sqrt(sum_sq);
      case cuvs::distance::DistanceType::InnerProduct: return -dot;
      case cuvs::distance::DistanceType::CosineExpanded: {
        auto denom = std::sqrt(norm_a) * std::sqrt(norm_b);
        return (denom > 0.0f) ? (1.0f - (dot / denom)) : 1.0f;
      }
      default: return sum_sq;
    }
  };
  std::vector<uint8_t> deleted_flags(static_cast<size_t>(rows), uint8_t{0});
  if (d_deleted_rows_) {
    raft::copy(
      deleted_flags.data(), static_cast<const uint8_t*>(d_deleted_rows_->data()), deleted_flags.size(), stream);
    raft::resource::sync_stream(handle_);
  }

  for (int64_t r = 0; r < rows; ++r) {
    // if the row is deleted, we don't care about the neighbors and can skip the sorting

    if (deleted_flags[static_cast<size_t>(r)] != 0) {
      continue;
    }
    
    std::vector<std::pair<float, IdxT>> valid_neighbors;
    std::vector<IdxT> invalid_neighbors;
    valid_neighbors.reserve(static_cast<size_t>(cols));
    invalid_neighbors.reserve(static_cast<size_t>(cols));

    for (int64_t c = 0; c < cols; ++c) {
      auto nb     = host_graph[static_cast<size_t>(r * cols + c)];
      auto nb_i64 = static_cast<int64_t>(nb);
      if (nb_i64 < 0 || nb_i64 >= rows) {
        invalid_neighbors.push_back(nb);
        continue;
      }
      valid_neighbors.emplace_back(row_distance(r, nb_i64), nb);
    }

    std::stable_sort(valid_neighbors.begin(), valid_neighbors.end(), [](const auto& lhs, const auto& rhs) {
      return lhs.first < rhs.first;
    });

    int64_t out_col = 0;
    std::vector<float> distance_threshold1(static_cast<size_t>(rows), uint8_t{0});
    std::vector<float> distance_threshold2(static_cast<size_t>(rows), uint8_t{0});
    for (const auto& [dist, nb] : valid_neighbors) {
      
      host_graph[static_cast<size_t>(r * cols + out_col)] = nb;
      if (out_col == DIST_THREDHOLD1_INDEX) {
        distance_threshold1[static_cast<size_t>(r)] = dist;
      }
      if (out_col == DIST_THREDHOLD2_INDEX) {
        distance_threshold2[static_cast<size_t>(r)] = dist;
      }
      ++out_col;
    }
    // dont think any element would get into this
    for (auto nb : invalid_neighbors) {
      if (out_col >= cols) { break; }
      host_graph[static_cast<size_t>(r * cols + out_col)] = nb;
      if (out_col == DIST_THREDHOLD1_INDEX) {
        distance_threshold1[static_cast<size_t>(r)] =  raft::upper_bound<float>();
      }
      if (out_col == DIST_THREDHOLD2_INDEX) {
        distance_threshold2[static_cast<size_t>(r)] =  raft::upper_bound<float>();
      }
      ++out_col;
    }
  }
 

  if (!graph_ || graph_->extent(0) != rows || graph_->extent(1) != cols) {
    *graph_ = raft::make_device_mdarray<IdxT, int64_t>(
      handle_, get_mr(graph_mem_), raft::make_extents<int64_t>(rows, cols));
  }

  auto reverse_cols = get_reverse_graph_cols(cols);
  if (!reverse_graph_ || reverse_graph_->extent(0) != rows ||
      reverse_graph_->extent(1) != reverse_cols) {
    *reverse_graph_ = raft::make_device_mdarray<IdxT, int64_t>(
      handle_, get_mr(graph_mem_), raft::make_extents<int64_t>(rows, reverse_cols));
    if (rows > 0 && reverse_cols > 0) {
      RAFT_CUDA_TRY(cudaMemsetAsync(reverse_graph_->data_handle(),
                                    0,
                                    static_cast<size_t>(rows * reverse_cols) * sizeof(IdxT),
                                    stream));
    }
  }

  raft::copy(graph_->data_handle(), host_graph.data(), host_graph.size(), stream);
  raft::resource::sync_stream(handle_);
  index_->update_graph(handle_, make_const_mdspan(graph_->view()));
}
template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::set_search_param(const search_param_base& param,
                                           const void* filter_bitset)
{
  if (index_) { filter_ = make_cuvs_filter(filter_bitset, index_->size()); }
  auto sp = dynamic_cast<const search_param&>(param);
  bool needs_dynamic_batcher_update =
    (dynamic_batching_max_batch_size_ != sp.dynamic_batching_max_batch_size) ||
    (dynamic_batching_n_queues_ != sp.dynamic_batching_n_queues) ||
    (dynamic_batching_conservative_dispatch_ != sp.dynamic_batching_conservative_dispatch);
  dynamic_batching_max_batch_size_        = sp.dynamic_batching_max_batch_size;
  dynamic_batching_n_queues_              = sp.dynamic_batching_n_queues;
  dynamic_batching_conservative_dispatch_ = sp.dynamic_batching_conservative_dispatch;
  search_params_                          = sp.p;
  refine_ratio_                           = sp.refine_ratio;
  search_params_.deleted_rows_ptr =
    d_deleted_rows_ ? static_cast<const uint8_t*>(d_deleted_rows_->data()) : nullptr;

  // Keep the requested memory preferences even if migration cannot happen yet.
  auto requested_graph_mem   = sp.graph_mem;
  auto requested_dataset_mem = sp.dataset_mem;

  // Enable cuvs logging for debugging
  //raft::default_logger().set_level( rapids_logger::level_enum::debug);
  if (requested_graph_mem != graph_mem_) {
    auto old_graph = index_ ? index_->graph()
                            : raft::device_matrix_view<const IdxT, int64_t, raft::row_major>(
                                nullptr, 0, 0);
    bool graph_ready = old_graph.data_handle() != nullptr && old_graph.extent(0) > 0 &&
                       old_graph.extent(1) > 0;

    // Deferred-apply path: remember preference and postpone heavy migration until graph exists.
    if (!graph_ready) {
      graph_mem_ = requested_graph_mem;
      needs_dynamic_batcher_update = true;
    } else {
    // Move graph to correct memory space
    graph_mem_ = requested_graph_mem;
    RAFT_LOG_DEBUG("moving graph to new memory space: %s", allocator_to_string(graph_mem_).c_str());
    // We create a new graph and copy to it from existing graph
    auto mr = get_mr(graph_mem_);

    // Create a new graph, then copy, and __only then__ replace the shared pointer.
    auto new_graph = raft::make_device_mdarray<IdxT, int64_t>(handle_, mr, old_graph.extents());
    raft::copy(new_graph.data_handle(),
               old_graph.data_handle(),
               old_graph.size(),
               raft::resource::get_cuda_stream(handle_));

    auto reverse_cols = get_reverse_graph_cols(old_graph.extent(1));
    auto new_reverse_graph =
      raft::make_device_mdarray<IdxT, int64_t>(
        handle_, mr, raft::make_extents<int64_t>(old_graph.extent(0), reverse_cols));
    if (reverse_graph_ && reverse_graph_->extent(0) == old_graph.extent(0) &&
        reverse_graph_->extent(1) == reverse_cols) {
      raft::copy(new_reverse_graph.data_handle(),
                 reverse_graph_->data_handle(),
                 reverse_graph_->size(),
                 raft::resource::get_cuda_stream(handle_));
    } else if (old_graph.extent(0) > 0 && reverse_cols > 0) {
      RAFT_CUDA_TRY(cudaMemsetAsync(new_reverse_graph.data_handle(),
                                    0,
                                    static_cast<size_t>(old_graph.extent(0) * reverse_cols) *
                                      sizeof(IdxT),
                                    raft::resource::get_cuda_stream(handle_)));
    }
    raft::resource::sync_stream(handle_);
    *graph_ = std::move(new_graph);
    *reverse_graph_ = std::move(new_reverse_graph);

    // NB: update_graph() only stores a view in the index. We need to keep the graph object alive.
    index_->update_graph(handle_, make_const_mdspan(graph_->view()));
    needs_dynamic_batcher_update = true;
    }
  }

  bool dataset_mem_changed = (requested_dataset_mem != dataset_mem_);
  if (dataset_mem_changed) {
    dataset_mem_ = requested_dataset_mem;
    // Force future dataset materialization to use the new memory resource.
    need_dataset_update_ = true;
  }

  if (need_dataset_update_) {
    bool dataset_source_ready = input_dataset_v_ && input_dataset_v_->data_handle() != nullptr &&
                                input_dataset_v_->extent(0) > 0 &&
                                input_dataset_v_->extent(1) == this->dim_;

    if (dataset_source_ready) {
      // First free up existing memory
      *dataset_ = raft::make_device_matrix<T, int64_t>(handle_, 0, 0);
      if (index_) { index_->update_dataset(handle_, make_const_mdspan(dataset_->view())); }

      // Allocate space using the correct memory resource.
      RAFT_LOG_DEBUG("moving dataset to new memory space: %s",
                     allocator_to_string(dataset_mem_).c_str());

      auto mr = get_mr(dataset_mem_);
      cuvs::neighbors::cagra::detail::copy_with_padding(handle_, *dataset_, *input_dataset_v_, mr);

      auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
        dataset_->data_handle(), dataset_->extent(0), this->dim_, dataset_->extent(1));
      if (index_) { index_->update_dataset(handle_, dataset_view); }

      need_dataset_update_         = false;
      needs_dynamic_batcher_update = true;
    }
  }

  // dynamic batching
  if (sp.dynamic_batching) {
    if (!dynamic_batcher_ || needs_dynamic_batcher_update) {
      dynamic_batcher_ =
        std::make_shared<cuvs::neighbors::dynamic_batching::index<T, algo_base::index_type>>(
          handle_,
          cuvs::neighbors::dynamic_batching::index_params{
            {},
            sp.dynamic_batching_k,
            sp.dynamic_batching_max_batch_size,
            sp.dynamic_batching_n_queues,
            sp.dynamic_batching_conservative_dispatch},
          *index_,
          search_params_,
          filter_.get());
    }
    dynamic_batcher_sp_.dispatch_timeout_ms = sp.dynamic_batching_dispatch_timeout_ms;
  } else {
    if (dynamic_batcher_) { dynamic_batcher_.reset(); }
  }
  // print graph and dataset extents for debugging
#ifdef KVDEBUG  
  auto graph = index_->graph();
  auto dataset = index_->dataset();

  std::cout << "[cuvs_cagra::set_search_param] graph extents: (" << graph.extent(0) << ", " << graph.extent(1) << ")" << std::endl;
  std::cout << "[cuvs_cagra::set_search_param] dataset extents: (" << dataset.extent(0) << ", " << dataset.extent(1) << ")" << std::endl; 
#endif
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::set_search_dataset(const T* dataset, size_t nrow)
{
  // print debug info
  std::cout << "[cuvs_cagra::set_search_dataset] REAL called with nrow=" << nrow << ", dim=" << dim_ << std::endl;
  #if 1
  // After an insert, callers may still pass the original base dataset from config.
  // Keep the internally expanded dataset instead if the provided one is smaller.
  if (has_inserted_data_ && index_ && nrow < static_cast<size_t>(index_->graph().extent(0))) {

    // log error
    std::cerr << "Warning: set_search_dataset called with nrow=" << nrow
              << " which is smaller than current index size=" << index_->graph().extent(0)
              << ". Ignoring the new dataset and keeping the existing one." << std::endl;
    return;
  }

  if (build_rows_ != index_->graph().extent(0)) {
    // log error and bail out to keep build-time row metadata consistent.
    std::cerr << "Error: set_search_dataset called with build_rows_=" << build_rows_
              << " which does not match current graph rows=" << index_->graph().extent(0)
              << ". Aborting set_search_dataset." << std::endl;
    return;
  } 
  auto stream              = raft::resource::get_cuda_stream(handle_);
  auto log_device_mem_diag = [&](const char* phase) {
    raft::resource::sync_stream(handle_);
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    RAFT_CUDA_TRY(cudaMemGetInfo(&free_bytes, &total_bytes));
    auto used_bytes = total_bytes - free_bytes;
    std::cout << "[set_search_dataset] memory " << phase << ": used=" << used_bytes
              << " free=" << free_bytes << " total=" << total_bytes << " bytes"
              << std::endl;
  };
  const auto expected_rows = rows_ > 0 ? static_cast<size_t>(build_rows_) : size_t{0};
  // I want dataset to be atleast the size of the build index
  if (expected_rows != 0 && nrow < expected_rows) { // relaxed the check when running as EXE (CUVS_CAGRA_ANN_BENCH , legacy flow rows_ = 0)
    // log error
    throw std::runtime_error(
      "set_search_dataset: search dataset row count (" + std::to_string(nrow) +
      ") does not match initial dataset row count used during algo creation (" +
      std::to_string(expected_rows) + ").");
  }
  auto old_graph = index_->graph();
  auto old_rows  = old_graph.extent(0);
  auto cols      = old_graph.extent(1);

  // Always expand/capacity-align search-time graph and dataset to rows_.
  auto target_rows = static_cast<int64_t>(rows_);
  if (target_rows < old_rows) { target_rows = old_rows; }

  std::cout << "[set_search_dataset] planning: input_rows=" << nrow
            << " old_graph_rows=" << old_rows
            << " old_graph_cols=" << cols
            << " rows_=" << rows_
            << " build_rows_=" << build_rows_
            << " target_rows=" << target_rows
            << " has_inserted_data_=" << has_inserted_data_ << std::endl;

  // Keep any host-side expansion buffer alive for the full function scope.
  std::vector<T> expanded_dataset;
  const T* active_dataset = dataset;
  auto active_nrow        = nrow;
  
  bool dataset_is_on_host = raft::get_device_for_address(dataset) == -1;

  if (!dataset_is_on_host)
  {
    // log an error and return
    std::cerr << "Warning: set_search_dataset called with a device pointer. "
              << "This is not supported. Ignoring the new dataset." << std::endl;
    return;
  }
  
  // if target_rows > nrow, expand the input argument dataset by copying to expanded memory zero-padding on the host memory
  if (target_rows > static_cast<int64_t>(active_nrow)) {
    std::cout << "[set_search_dataset] dataset expansion (host padding): from " << active_nrow
              << " rows to " << target_rows << " rows" << std::endl;
    // allocate new host memory and copy existing dataset
    expanded_dataset.resize(static_cast<size_t>(target_rows * dim_), T{0});
    for (size_t i = 0; i < active_nrow; ++i) {
      std::memcpy(expanded_dataset.data() + i * dim_,
                  active_dataset + i * dim_,
                  static_cast<size_t>(dim_) * sizeof(T));
    }
    // Keep a stable pointer/size pair for the rest of this function.
    active_dataset = expanded_dataset.data();
    active_nrow    = static_cast<size_t>(target_rows);
  }

  if (index_params_.num_dataset_splits > 1 &&
      index_params_.merge_type == CagraMergeType::kLogical) {
    IdxT rows_per_split =
      raft::ceildiv<IdxT>(active_nrow, static_cast<IdxT>(index_params_.num_dataset_splits));
    for (size_t i = 0; i < sub_indices_.size(); ++i) {
      IdxT start = static_cast<IdxT>(i * rows_per_split);
      if (start >= active_nrow) break;
      IdxT rows        = std::min(rows_per_split, static_cast<IdxT>(active_nrow) - start);
      const T* sub_ptr = active_dataset + static_cast<size_t>(start) * dim_;
      auto sub_host =
        raft::make_host_matrix_view<const T, int64_t, raft::row_major>(sub_ptr, rows, dim_);
      auto sub_index = sub_indices_[i].get();
      if (index_params_.merge_type == CagraMergeType::kLogical) {
        sub_index->update_dataset(handle_, sub_host);
      }
    }
    need_dataset_update_ = false;
  } else {
    using ds_idx_type = decltype(index_->data().n_rows());
    bool is_vpq =
      dynamic_cast<const cuvs::neighbors::vpq_dataset<half, ds_idx_type>*>(&index_->data()) ||
      dynamic_cast<const cuvs::neighbors::vpq_dataset<float, ds_idx_type>*>(&index_->data());
    // It can happen that we are re-using a previous algo object which already has
    // the dataset set. Check if we need update.
#if 1
    auto index_dataset = index_->dataset();
    bool can_reuse_index_dataset =
      !need_dataset_update_ && !has_inserted_data_ && index_dataset.data_handle() != nullptr &&
      index_dataset.extent(1) == this->dim_ &&
      static_cast<size_t>(index_dataset.extent(0)) == active_nrow;

    if (can_reuse_index_dataset) {
      // Keep search setup lightweight: reuse dataset already attached at build/load.
      *input_dataset_v_ = raft::make_device_matrix_view<const T, int64_t>(
        index_dataset.data_handle(), index_dataset.extent(0), this->dim_);
      need_dataset_update_ = false;
    } else if (static_cast<size_t>(input_dataset_v_->extent(0)) != active_nrow ||
               input_dataset_v_->data_handle() != active_dataset) {
      auto materialized_rows = static_cast<int64_t>(active_nrow);
      std::cout << "[set_search_dataset] dataset expansion (device copy_with_padding): source_rows="
                << active_nrow << " target_rows=" << materialized_rows << " dim=" << this->dim_
                << std::endl;
      auto expected_dataset_bytes =
        static_cast<size_t>(materialized_rows) * static_cast<size_t>(this->dim_) * sizeof(T);
      std::cout << "[set_search_dataset] expected dataset_ allocation bytes="
                << expected_dataset_bytes << " (rows=" << materialized_rows
                << ", cols=" << this->dim_ << ", sizeof(T)=" << sizeof(T) << ")"
                << std::endl;
      log_device_mem_diag("before dataset expansion");
      auto host_dataset_view =
        raft::make_host_matrix_view<const T, int64_t, raft::row_major>(active_dataset, active_nrow, this->dim_);
      auto dataset_mr = get_mr(dataset_mem_);
      cuvs::neighbors::cagra::detail::copy_with_padding(
        handle_, *dataset_, host_dataset_view, dataset_mr);
      auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
        dataset_->data_handle(), materialized_rows, this->dim_, dataset_->extent(1));
      index_->update_dataset(handle_, dataset_view);

      std::cout << "[set_search_dataset] dataset update complete: index_dataset_rows="
                << index_->dataset().extent(0)
                << " index_dataset_cols=" << index_->dataset().extent(1) << std::endl;
      log_device_mem_diag("after dataset expansion");

      *input_dataset_v_ =
        raft::make_device_matrix_view<const T, int64_t>(
          dataset_->data_handle(), materialized_rows, this->dim_);

      need_dataset_update_ = false;
    }
    #endif
  }


  if (old_rows < target_rows) {
    auto mr                 = get_mr(graph_mem_);
    log_device_mem_diag("before forward graph expansion");
    std::cout << "[set_search_dataset] graph expansion: old_rows=" << old_rows
              << " target_rows=" << target_rows << " cols=" << cols << std::endl;
    auto expected_new_graph_bytes =
      static_cast<size_t>(target_rows) * static_cast<size_t>(cols) * sizeof(IdxT);
    std::cout << "[set_search_dataset] expected new_expanded_graph allocation bytes="
              << expected_new_graph_bytes << " (rows=" << target_rows << ", cols=" << cols
              << ", sizeof(IdxT)=" << sizeof(IdxT) << ")" << std::endl;
    auto new_expanded_graph = raft::make_device_mdarray<IdxT, int64_t>(
      handle_, mr, raft::make_extents<int64_t>(target_rows, cols));

    if (old_rows > 0) {
      raft::copy(
        new_expanded_graph.data_handle(), old_graph.data_handle(), old_graph.size(), stream);
    }

    // Zero-initialize the tail rows [old_rows, target_rows).
    // raft::make_device_mdarray does NOT zero-initialize, so without this the tail
    // contains garbage neighbor IDs.  When CAGRA search traverses those IDs it
    // computes distances against invalid dataset addresses → OOB illegal access.
    auto tail_offset = static_cast<size_t>(old_rows) * static_cast<size_t>(cols);
    auto tail_elems  = static_cast<size_t>(target_rows - old_rows) * static_cast<size_t>(cols);
    if (tail_elems > 0) {
      RAFT_CUDA_TRY(cudaMemsetAsync(
        new_expanded_graph.data_handle() + tail_offset,
        0,
        tail_elems * sizeof(IdxT),
        stream));
    }
    raft::resource::sync_stream(handle_);

    std::cout << "[set_search_dataset] expanded graph: old_rows=" << old_rows
              << " target_rows=" << target_rows << " cols=" << cols
              << " zeroed_tail_elems=" << tail_elems << std::endl;

    *graph_ = std::move(new_expanded_graph);
    index_->update_graph(handle_, make_const_mdspan(graph_->view()));
    log_device_mem_diag("after forward graph expansion");

    std::cout << "[set_search_dataset] forward graph committed; reverse graph will be rebuilt"
              << " after the committed graph is visible." << std::endl;

  #if 0
    // Also expand deleted_rows buffer and mark expanded rows as deleted
    if (d_deleted_rows_) {
      auto old_deleted_bytes = static_cast<size_t>(old_rows) * sizeof(uint8_t);
      auto new_deleted_bytes = static_cast<size_t>(target_rows) * sizeof(uint8_t);
      auto new_deleted_rows  = std::make_shared<rmm::device_buffer>(
        new_deleted_bytes, stream, get_mr(AllocatorType::kDevice));

      // Copy old (built) rows as not deleted (0)
      if (old_rows > 0 && old_deleted_bytes > 0) {
        RAFT_CUDA_TRY(cudaMemcpyAsync(
          new_deleted_rows->data(), d_deleted_rows_->data(), old_deleted_bytes, 
          cudaMemcpyDeviceToDevice, stream));
      }

      // Mark expanded rows (old_rows to target_rows) as deleted (1)
      auto new_deleted_ptr = static_cast<uint8_t*>(new_deleted_rows->data());
      auto expanded_bytes  = static_cast<size_t>(target_rows - old_rows) * sizeof(uint8_t);
      if (expanded_bytes > 0) {
        RAFT_CUDA_TRY(cudaMemsetAsync(
          new_deleted_ptr + old_rows, 1, expanded_bytes, stream));
      }
      raft::resource::sync_stream(handle_);

      d_deleted_rows_ = new_deleted_rows;

      // expand dist_threshold buffers as well, initialize new rows with infinity
      auto new_dist_threshold1 = std::make_shared<rmm::device_buffer>(
        sizeof(float) * target_rows, stream, get_mr(AllocatorType::kDevice));
      auto new_dist_threshold2 = std::make_shared<rmm::device_buffer>(
        sizeof(float) * target_rows, stream, get_mr(AllocatorType::kDevice)); 
      if (old_rows > 0) {
        RAFT_CUDA_TRY(cudaMemcpyAsync(
          new_dist_threshold1->data(), d_dist_threshold1_->data(), sizeof(float) * old_rows, 
          cudaMemcpyDeviceToDevice, stream));
        RAFT_CUDA_TRY(cudaMemcpyAsync(
          new_dist_threshold2->data(), d_dist_threshold2_->data(), sizeof(float) * old_rows, 
          cudaMemcpyDeviceToDevice, stream));
    }
      // Initialize new rows with infinity
      if (target_rows > old_rows) {
        RAFT_CUDA_TRY(cudaMemsetAsync(
          static_cast<float*>(new_dist_threshold1->data()) + old_rows, 0x7f, sizeof(float) * (target_rows - old_rows), stream));
        RAFT_CUDA_TRY(cudaMemsetAsync(
          static_cast<float*>(new_dist_threshold2->data()) + old_rows, 0x7f, sizeof(float) * (target_rows - old_rows), stream));
      }
      raft::resource::sync_stream(handle_);

      d_dist_threshold1_ = new_dist_threshold1;
      d_dist_threshold2_ = new_dist_threshold2;
    }
#endif
  }

  // set_search_dataset is the canonical place where reverse-graph rebuild is triggered.
  static_assert(sizeof(IdxT) == sizeof(uint32_t),
                "IdxT must be uint32_t for launch_cuvs_bang_build_reverse_graph_kernel");
  auto curr_graph = index_->graph();
  auto curr_rows  = curr_graph.extent(0);
  auto curr_cols  = curr_graph.extent(1);
  if (curr_rows > 0 && curr_cols > 0) {
    auto reverse_cols = get_reverse_graph_cols(curr_cols);
    if (!reverse_graph_ || reverse_graph_->extent(0) != curr_rows ||
        reverse_graph_->extent(1) != reverse_cols) {
      log_device_mem_diag("before reverse graph expansion");
      auto old_reverse_rows = reverse_graph_ ? reverse_graph_->extent(0) : int64_t{0};
      auto old_reverse_cols = reverse_graph_ ? reverse_graph_->extent(1) : int64_t{0};
      std::cout << "[set_search_dataset] reverse graph expansion (shape fixup): old_rows="
                << old_reverse_rows << " old_cols=" << old_reverse_cols
                << " new_rows=" << curr_rows << " new_cols=" << reverse_cols << std::endl;
      auto expected_reverse_graph_bytes =
        static_cast<size_t>(curr_rows) * static_cast<size_t>(reverse_cols) * sizeof(IdxT);
      std::cout << "[set_search_dataset] expected reverse_graph_ allocation bytes="
                << expected_reverse_graph_bytes << " (rows=" << curr_rows
                << ", cols=" << reverse_cols << ", sizeof(IdxT)=" << sizeof(IdxT) << ")"
                << std::endl;
      *reverse_graph_ = raft::make_device_mdarray<IdxT, int64_t>(
        handle_, get_mr(graph_mem_), raft::make_extents<int64_t>(curr_rows, reverse_cols));
      log_device_mem_diag("after reverse graph expansion");
    }

    ensure_reverse_counts_buffer(curr_rows);
    ensure_reverse_row_ptr_buffer(curr_rows);
    detail::launch_cuvs_bang_build_reverse_graph_kernel(
      reinterpret_cast<const uint32_t*>(curr_graph.data_handle()),
      curr_rows,
      curr_cols,
      reinterpret_cast<uint32_t*>(reverse_graph_->data_handle()),
      reverse_cols,
      d_deleted_rows_ ? static_cast<const uint8_t*>(d_deleted_rows_->data()) : nullptr,
      d_reverse_row_ptr_ ? static_cast<uint32_t*>(d_reverse_row_ptr_->data()) : nullptr,
      static_cast<uint32_t*>(d_reverse_counts_->data()));

    // Persist reverse graph after rebuild.
    std::string reverse_graph_file =
      (graph_file_ && !graph_file_->empty()) ? (*graph_file_ + ".reverse_graph.bin")
                                             : std::string("reverse_graph.bin");
    std::vector<IdxT> host_reverse_graph(static_cast<size_t>(curr_rows * reverse_cols));
    raft::copy(host_reverse_graph.data(),
               reverse_graph_->data_handle(),
               host_reverse_graph.size(),
               stream);
    raft::resource::sync_stream(handle_);
#if 0
    std::ofstream reverse_out(reverse_graph_file, std::ios::binary | std::ios::trunc);
    if (reverse_out.good()) {
      reverse_out.write(reinterpret_cast<const char*>(&curr_rows), sizeof(curr_rows));
      reverse_out.write(reinterpret_cast<const char*>(&reverse_cols), sizeof(reverse_cols));
      reverse_out.write(reinterpret_cast<const char*>(host_reverse_graph.data()),
                        static_cast<std::streamsize>(host_reverse_graph.size() * sizeof(IdxT)));
      reverse_out.close();
    } else {
      std::cerr << "Warning: failed to save reverse graph file to " << reverse_graph_file
                << std::endl;
    }
#endif
  }

  // Final consistency check after dataset/graph updates and reverse-graph rebuild.
  {
    auto final_graph = index_->graph();
    auto final_data  = index_->dataset();
    auto final_graph_rows = final_graph.extent(0);
    auto final_data_rows  = final_data.extent(0);
    if (final_graph_rows != final_data_rows) {
      std::cerr << "[set_search_dataset] Error: graph/dataset row mismatch after setup. "
                << "graph_rows=" << final_graph_rows
                << " dataset_rows=" << final_data_rows << std::endl;
      throw std::runtime_error("set_search_dataset: graph and dataset row counts must match.");
    }
    std::cout << "[set_search_dataset] final consistency check passed: graph_rows="
              << final_graph_rows << " dataset_rows=" << final_data_rows << std::endl;
  }
  #endif
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::save(const std::string& file) const
{
  if (index_params_.num_dataset_splits > 1 &&
      index_params_.merge_type == CagraMergeType::kLogical) {
    for (size_t i = 0; i < sub_indices_.size(); ++i) {
      std::string subfile = file + (i == 0 ? "" : ".subidx." + std::to_string(i));
      cuvs::neighbors::cagra::serialize(handle_, subfile, *sub_indices_[i], false);
    }
    std::ofstream f(file + ".submeta", std::ios::out);
    f << sub_indices_.size();
    f.close();
  } else {
    using ds_idx_type = decltype(index_->data().n_rows());
    bool is_vpq =
      dynamic_cast<const cuvs::neighbors::vpq_dataset<half, ds_idx_type>*>(&index_->data()) ||
      dynamic_cast<const cuvs::neighbors::vpq_dataset<float, ds_idx_type>*>(&index_->data());
    cuvs::neighbors::cagra::serialize(handle_, file, *index_, is_vpq);
  }
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::save_to_hnswlib(const std::string& file) const
{
  cuvs::neighbors::cagra::serialize_to_hnswlib(handle_, file, *index_);
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::load(const std::string& file)
{
  graph_file_ = file;
  std::ifstream meta(file + ".submeta", std::ios::in);
  if (index_params_.num_dataset_splits > 1 &&
      index_params_.merge_type == CagraMergeType::kLogical && meta.good()) {
    // Load multiple sub-indices for logical merge
    size_t count;
    meta >> count;
    meta.close();
    sub_indices_.clear();
    for (size_t i = 0; i < count; ++i) {
      std::string subfile = file + (i == 0 ? "" : ".subidx." + std::to_string(i));
      auto sub_index      = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(handle_);
      cuvs::neighbors::cagra::deserialize(handle_, subfile, sub_index.get());
      sub_indices_.push_back(std::move(sub_index));
    }
  } else {
    index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(handle_);
    cuvs::neighbors::cagra::deserialize(handle_, file, index_.get());
  }
  // No longer expecting only legacy constructor to call load
  #if 0
  if (rows_ != 0)
  {
    // throw exception
    throw std::runtime_error(
      "load: algo was created with non-zero rows (" + std::to_string(rows_) +
      ") but loaded index has " + std::to_string(index_->graph().extent(0)) + " rows.");   
  }
  #endif

  auto loaded_rows = static_cast<int>(index_->graph().extent(0));
  if (rows_ == 0) { rows_ = loaded_rows; }
  auto buffer_rows = std::max(rows_, loaded_rows);
  // log details of on the loaded index
  std::cout << "[cuvs_cagra::load] loaded index from " << file
            << " with loaded_rows=" << loaded_rows
            << " target_rows=" << rows_
            << " buffer_rows=" << buffer_rows
            << ", cols=" << index_->graph().extent(1)
            << ", dataset rows=" << index_->dataset().extent(0)
            << ", dataset cols=" << index_->dataset().extent(1) << std::endl;

  build_rows_ = loaded_rows;
  d_deleted_rows_         = std::make_shared<rmm::device_buffer>(
    static_cast<size_t>(buffer_rows) * sizeof(uint8_t), handle_.get_sync_stream(), get_mr(AllocatorType::kDevice));
  d_dist_threshold1_ = std::make_shared<rmm::device_buffer>(
    static_cast<size_t>(buffer_rows) * sizeof(float), handle_.get_sync_stream(), get_mr(AllocatorType::kDevice));
  d_dist_threshold2_ = std::make_shared<rmm::device_buffer>(
    static_cast<size_t>(buffer_rows) * sizeof(float), handle_.get_sync_stream(), get_mr(AllocatorType::kDevice));
  
  // log the value of delete_rows_init_size for debugging
  RAFT_CUDA_TRY(cudaMemsetAsync(
    d_deleted_rows_->data(), 1, static_cast<size_t>(buffer_rows) * sizeof(uint8_t), handle_.get_sync_stream()));
    raft::resource::sync_stream(handle_);
  
  auto delete_rows_init_size = static_cast<size_t>(build_rows_); // default
  if (const char* env_val = std::getenv("CUVS_BENCH_VALID_ROWS_INIT_SIZE")) {
    try {
      delete_rows_init_size = std::stoull(env_val);
    } catch (...) {
      std::cerr << "[cuvs_cagra] Warning: invalid CUVS_BENCH_VALID_ROWS_INIT_SIZE='" << env_val
                << "'; using default " << delete_rows_init_size << std::endl;
    }
  }

  std::cout << "[cuvs_cagra::load] Initializing deleted rows buffer with size: " << delete_rows_init_size << std::endl;
  // log the value of delete_rows_init_size for debugging
#ifdef _KVDEBUG
  std::cout << "[cuvs_cagra::load] Initializing deleted rows buffer with size: " << delete_rows_init_size << std::endl;
#endif
  RAFT_CUDA_TRY(cudaMemsetAsync(
    d_deleted_rows_->data(), 0, delete_rows_init_size * sizeof(uint8_t), handle_.get_sync_stream()));
    raft::resource::sync_stream(handle_);
  
}

template <typename T, typename IdxT>
std::unique_ptr<algo<T>> cuvs_cagra<T, IdxT>::copy()
{
  return std::make_unique<cuvs_cagra<T, IdxT>>(std::cref(*this));  // use copy constructor
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::ensure_delete_workspace(size_t required_ids, int64_t graph_rows)
{
  if (required_ids == 0 || graph_rows <= 0) { return; }

  auto required_frontier_capacity = graph_rows * 1;
  if (required_frontier_capacity <= 0) { required_frontier_capacity = 1; }

  const bool needs_realloc = !d_delete_frontier_curr_ || !d_delete_frontier_next_ ||
                             !d_delete_frontier_size_ || !d_delete_next_size_ ||
                             delete_workspace_max_ids_ < required_ids ||
                             delete_workspace_frontier_capacity_ < required_frontier_capacity;

  if (!needs_realloc) { return; }

  auto stream            = handle_.get_sync_stream();
  auto frontier_total    = static_cast<uint64_t>(required_ids) *
                        static_cast<uint64_t>(required_frontier_capacity);
  auto frontier_bytes    = frontier_total * sizeof(uint32_t);
  auto counters_bytes    = required_ids * sizeof(int);

  d_delete_frontier_curr_ = std::make_shared<rmm::device_buffer>(
    frontier_bytes, stream, get_mr(AllocatorType::kDevice));
  d_delete_frontier_next_ = std::make_shared<rmm::device_buffer>(
    frontier_bytes, stream, get_mr(AllocatorType::kDevice));
  d_delete_frontier_size_ = std::make_shared<rmm::device_buffer>(
    counters_bytes, stream, get_mr(AllocatorType::kDevice));
  d_delete_next_size_ = std::make_shared<rmm::device_buffer>(
    counters_bytes, stream, get_mr(AllocatorType::kDevice));

  delete_workspace_frontier_capacity_ = required_frontier_capacity;
  delete_workspace_max_ids_           = required_ids;

  std::printf("[bang] delete workspace allocated once: ids_cap=%zu frontier_cap=%lld total_bytes=%llu\n",
              delete_workspace_max_ids_,
              static_cast<long long>(delete_workspace_frontier_capacity_),
              static_cast<unsigned long long>(2 * frontier_bytes + 2 * counters_bytes));
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::set_insert_param(const insert_param& param)
{
  insert_params_ = param;
#if 0
  auto graph_rows = index_ ? index_->graph().extent(0) : static_cast<int64_t>(rows_);
  auto warmup_ids = param.max_chunk_size > 0 ? static_cast<size_t>(param.max_chunk_size) : size_t{0};
  if (warmup_ids > 0 && graph_rows > 0) {
    ensure_delete_workspace(warmup_ids, graph_rows);
  }
#endif
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::set_insert_param_from_json(const nlohmann::json& conf)
{
  insert_param ip{};
  if (conf.contains("persist")) { ip.persist = conf.at("persist").get<bool>(); }
  if (conf.contains("max_chunk_size")) {
    ip.max_chunk_size = conf.at("max_chunk_size").get<uint32_t>();
  }
  set_insert_param(ip);
}

#if 0
// serialized insert
template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::insert_singlethread(const T* vectors, size_t num_vectors, const uint64_t* ids)
{
  if (num_vectors == 0) { return; }
  int batch_size = static_cast<int>(num_vectors);

  const T* queries = vectors;
  // validate the graph and dataset are already loaded
  if (!index_) {
    throw std::runtime_error("Index must be loaded or built before insert operations.");
  }
  if (!input_dataset_v_ || input_dataset_v_->extent(1) == 0) {
    throw std::runtime_error("Dataset view is not initialized; call set_search_dataset() first.");
  }

   // verify number of nodes in the graph equals the dataset size
  if (index_->graph().extent(0) != input_dataset_v_->extent(0)) {
    throw std::runtime_error("Graph node count must match dataset size before insert operations.");
  }

  // if the index is empty (no graph/dataset), we can skip directly to building a 
  // new index with the inserted data as the initial dataset  to avoid unnecessary copying and merging
  if (index_->graph().extent(0) == 0) {
    RAFT_LOG_DEBUG("Index is empty, skipping to build with new dataset directly.");
    build(queries, num_vectors);
    return;
  }  


  // Ensure the index has a dataset attached (deserialize may not include it).
  if (index_->dim() == 0 || need_dataset_update_) {
    auto mr = get_mr(dataset_mem_);
    cuvs::neighbors::cagra::detail::copy_with_padding(handle_, *dataset_, *input_dataset_v_, mr);
    auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
      dataset_->data_handle(), dataset_->extent(0), this->dim_, dataset_->extent(1));
    index_->update_dataset(handle_, dataset_view);
    need_dataset_update_ = false;
  }
  if (index_->dim() == 0) {
    throw std::runtime_error("Index has dim=0 after dataset attach; load/build the index first.");
  }

  auto stream = raft::resource::get_cuda_stream(handle_);
  bool vectors_on_device = raft::get_device_for_address(vectors) >= 0;
  std::unique_ptr<rmm::device_uvector<T>> device_vectors_buf;
  if (!vectors_on_device) {
    device_vectors_buf = std::make_unique<rmm::device_uvector<T>>(
      static_cast<size_t>(num_vectors) * static_cast<size_t>(dim_), stream);
    raft::copy(device_vectors_buf->data(),
               vectors,
               static_cast<size_t>(num_vectors) * static_cast<size_t>(dim_),
               stream);
    queries = device_vectors_buf->data();
  }

  rmm::device_uvector<algo_base::index_type> neighbors_buf(
    static_cast<size_t>(num_vectors) * static_cast<size_t>(index_->graph().extent(1)), stream);
  rmm::device_uvector<float> distances_buf(
    static_cast<size_t>(num_vectors) * static_cast<size_t>(index_->graph().extent(1)), stream);
  algo_base::index_type* neighbors = neighbors_buf.data();
  float* distances = distances_buf.data();

  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);
  auto k                         = static_cast<int>(index_->graph().extent(1));
  auto k0                       = static_cast<size_t>(refine_ratio_ * k);
  const bool disable_refinement = k0 <= static_cast<size_t>(k);
  const raft::resources& res    = handle_;
  // NOTE: caching mem_type to reduce mutex locks
  // raft::get_device_for_address call cuda API to get the pointer properties,
  // this means it locks the context mutex for a very small amount of time.
  // In the event of thread contention (such as thousands threads), this time can actually increase.
  // Hence we try to bypass this check for repeated search calls.
  thread_local MemoryType mem_type                   = MemoryType::kDevice;
  thread_local algo_base::index_type* prev_neighbors = nullptr;
  if (prev_neighbors != neighbors) {
    prev_neighbors = neighbors;
    mem_type =
      raft::get_device_for_address(neighbors) >= 0 ? MemoryType::kDevice : MemoryType::kHostPinned;
  }

  // If dynamic batching is used and there's no sync between benchmark laps, multiple sequential
  // requests can group together. The data is copied asynchronously, and if the same intermediate
  // buffer is used for multiple requests, they can override each other's data. Hence, we need to
  // allocate as much space as required by the maximum number of sequential requests.
  auto max_dyn_grouping = dynamic_batcher_ ? raft::div_rounding_up_safe<int64_t>(
                                               dynamic_batching_max_batch_size_, batch_size) *
                                               dynamic_batching_n_queues_
                                           : 1;
  auto tmp_buf_size =
    ((disable_refinement ? 0 : (sizeof(float) + sizeof(algo_base::index_type)))) * batch_size * k0;
  auto& tmp_buf = get_tmp_buffer_from_global_pool(tmp_buf_size * max_dyn_grouping);
  thread_local static int64_t group_id = 0;
  auto* candidates_ptr                 = reinterpret_cast<algo_base::index_type*>(
    reinterpret_cast<uint8_t*>(tmp_buf.data(mem_type)) + tmp_buf_size * group_id);
  group_id = (group_id + 1) % max_dyn_grouping;
  auto* candidate_dists_ptr =
    reinterpret_cast<float*>(candidates_ptr + (disable_refinement ? 0 : batch_size * k0));

  if (disable_refinement) {
    search_base(queries, batch_size, k, neighbors, distances);
  } else {
    search_base(queries, batch_size, k0, candidates_ptr, candidate_dists_ptr);

    if (mem_type == MemoryType::kHostPinned && uses_stream()) {
      // If the algorithm uses a stream to synchronize (non-persistent kernel), but the data is in
      // the pinned host memory, we need to synchronize before the refinement operation to wait for
      // the data being available for the host.
      raft::resource::sync_stream(res);
    }

    auto candidate_ixs =
      raft::make_device_matrix_view<const algo_base::index_type, algo_base::index_type>(
        candidates_ptr, batch_size, k0);
    auto queries_v =
      raft::make_device_matrix_view<const T, algo_base::index_type>(queries, batch_size, dim_);
    refine_helper(
      res, *input_dataset_v_, queries_v, candidate_ixs, k, neighbors, distances, index_->metric());
  }

#if 0
  // Debug: Copy first query's neighbors and distances from device to host for logging
  if (batch_size > 0) {
    std::vector<algo_base::index_type> host_neighbors_debug(k);
    std::vector<float> host_distances_debug(k);

    raft::copy(host_neighbors_debug.data(),
               neighbors,
               k,
               raft::resource::get_cuda_stream(handle_));
    raft::copy(host_distances_debug.data(),
               distances,
               k,
               raft::resource::get_cuda_stream(handle_));
    raft::resource::sync_stream(handle_);

    RAFT_LOG_DEBUG("First query's %d nearest neighbors:", k);
    for (int i = 0; i < std::min(k, 5); ++i) {
      RAFT_LOG_DEBUG("  [%d] index=%u, distance=%f",
                     i,
                     host_neighbors_debug[i],
                     host_distances_debug[i]);
    }
  }
#endif

  raft::resource::sync_stream(res);
  bool queries_on_device = raft::get_device_for_address(queries) >= 0;
  bool neighbors_on_device = (mem_type == MemoryType::kDevice);
  bool distances_on_device = raft::get_device_for_address(distances) >= 0;

  std::vector<T> host_queries(static_cast<size_t>(batch_size) * static_cast<size_t>(dim_));
  if (queries_on_device) {
    raft::copy(host_queries.data(),
               queries,
               static_cast<size_t>(batch_size) * static_cast<size_t>(dim_),
               raft::resource::get_cuda_stream(res));
  } else {
    std::memcpy(host_queries.data(),
                queries,
                static_cast<size_t>(batch_size) * static_cast<size_t>(dim_) * sizeof(T));
  }

  std::vector<algo_base::index_type> host_neighbors(static_cast<size_t>(batch_size) *
                                                    static_cast<size_t>(k));
  if (neighbors_on_device) {
    raft::copy(host_neighbors.data(),
               neighbors,
               static_cast<size_t>(batch_size) * static_cast<size_t>(k),
               raft::resource::get_cuda_stream(res));
  } else {
    std::memcpy(host_neighbors.data(),
                neighbors,
                static_cast<size_t>(batch_size) * static_cast<size_t>(k) *
                  sizeof(algo_base::index_type));
  }


  raft::resource::sync_stream(res);

  // Build temporary host-backed graph with new rows inserted and serialize that temporary index.
  {
    auto orig_graph = index_->graph();
    int64_t old_rows = orig_graph.extent(0);
    int64_t cols     = orig_graph.extent(1);
    //int64_t new_rows = old_rows + batch_size;

    std::vector<IdxT> host_graph(static_cast<size_t>(old_rows * cols));

    // copy existing graph into host buffer (works for host or device-backed original)
    if (old_rows > 0) {
      raft::copy(host_graph.data(),
                 orig_graph.data_handle(),
                 static_cast<size_t>(old_rows * cols),
                 raft::resource::get_cuda_stream(handle_));
      raft::resource::sync_stream(handle_);
    }

    // Get dataset pointer on host
    int64_t dim = dim_;
    int64_t old_dataset_rows = dataset_->extent(0);
    //int64_t total_rows = old_dataset_rows + batch_size;  // Include new rows

    // Copy dataset to host (if on device) and allocate space for new rows
    std::vector<T> host_dataset(static_cast<size_t>(old_dataset_rows) * static_cast<size_t>(dim));
    raft::copy(host_dataset.data(),
               dataset_->data_handle(),
               static_cast<size_t>(old_dataset_rows) * static_cast<size_t>(dim),
               raft::resource::get_cuda_stream(handle_));
    raft::resource::sync_stream(handle_);

    // --- Insert new nodes and make edges bidirectional
    for (int64_t nRowIter = 0; nRowIter < batch_size; nRowIter++) {
      // copy new query vector to host dataset (appending to existing dataset)
      std::memcpy(host_dataset.data() + (size_t(ids[nRowIter]) * size_t(dim)),
                  host_queries.data() + (size_t(nRowIter) * size_t(dim)),
                  static_cast<size_t>(dim) * sizeof(T));

      // --- New adjacency update logic ------------------------------------------------------
      int64_t new_row_id = ids[nRowIter]; // old_rows + nRowIter; // Assuming ids are sequential and start from old_rows, otherwise we need to find the correct position for each id in the new graph.

      // ---------------------------------------------------------
      // 1 FORWARD EDGES: Build a clean fixed-degree row
      // ---------------------------------------------------------

      std::vector<IdxT> vecAdjList;

      // First pass: collect all candidates
      for (int j = 0; j < k; j++) {
        IdxT nid = neighbors_on_device ? host_neighbors[size_t(nRowIter) * k + j]
                                        : neighbors[size_t(nRowIter) * k + j];

        vecAdjList.push_back(nid);
      }

      // ToDo: Don't padd with zeros, instead try to fill with existing neighbors of neighbors
      // (2-hop) or even random existing nodes in the graph, to maintain better connectivity and
      // avoid isolated nodes.
      // Second pass: fill missing slots without applying filter (fallback)
      if (vecAdjList.size() < size_t(cols)) {
        vecAdjList.resize(cols, 0);
      }

      // Write compact adjacency (NO HOLES)
      for (int j = 0; j < cols; j++) {
        host_graph[new_row_id * cols + j] = vecAdjList[j];
      }

      // ---------------------------------------------------------
      // 2 REVERSE EDGES: Common-neighbour based rewiring
      // ---------------------------------------------------------
      bool inserted_any = false;

      for (IdxT nid : vecAdjList) {
        if (nid >= old_rows || nid == new_row_id) continue;

        // Check if nid already links to new_row_id
        bool already_present = false;
        for (int64_t j = 0; j < cols; j++) {
          if (host_graph[nid * cols + j] == new_row_id) {
            already_present = true;
            break;
          }
        }
        if (already_present) continue;

        IdxT replace_candidate;
        if (!has_common_neighbour(
              nid, new_row_id, host_graph, cols, old_rows, replace_candidate))
          continue;

        // Replace the common neighbor with new_row_id
        for (int64_t j = 0; j < cols; j++) {
          if (host_graph[nid * cols + j] == replace_candidate) {
            host_graph[nid * cols + j] = new_row_id;
            inserted_any = true;
            break;
          }
        }
      }

      // Fallback safety: ensure at least one reverse edge to the newly inserted point
      if (!inserted_any && !vecAdjList.empty()) {
        IdxT nid = vecAdjList[0];
        host_graph[nid * cols + 0] = new_row_id;
      }
    }  // For loop on each query

    // copy the host dataset with new vectors back to device (if original dataset was on device)
    if (raft::get_device_for_address(dataset_->data_handle()) >= 0) {
      if (dataset_->extent(0) != old_dataset_rows || dataset_->extent(1) != dim) {
        *dataset_ = raft::make_device_matrix<T, int64_t>(handle_, old_dataset_rows, dim);
      }
      raft::copy(dataset_->data_handle(),
                 host_dataset.data(),
                 static_cast<size_t>(old_dataset_rows) * static_cast<size_t>(dim),
                 raft::resource::get_cuda_stream(handle_));
      raft::resource::sync_stream(handle_);
      index_->update_dataset(handle_, make_const_mdspan(dataset_->view()));
    }

    // copy the updated host graph with new rows back to device (if original graph was on device)
    if (raft::get_device_for_address(index_->graph().data_handle()) >= 0) {
      if (graph_->extent(0) != old_rows || graph_->extent(1) != cols) {
        *graph_ = raft::make_device_matrix<IdxT, int64_t>(handle_, old_rows, cols);
      }
      raft::copy(graph_->data_handle(),
                 host_graph.data(),
                 static_cast<size_t>(old_rows) * static_cast<size_t>(cols),
                 raft::resource::get_cuda_stream(handle_));
      raft::resource::sync_stream(handle_);
      index_->update_graph(handle_, make_const_mdspan(graph_->view()));
    }

    // log the size of the updated graph and dataset for debugging
    std::cout << "Updated graph size: " << old_rows << " rows x " << cols << " cols\n";
    std::cout << "Updated dataset size: " << old_dataset_rows << " rows x " << dim << " cols\n";
    #if 0
    // if the persist flag is set, serialize the updated index back to disk (overwriting original)
    if (persist_insert_) {
      // dont hardcode the path here, instead use the original path from which the index was loaded (graph_file_)
      std::string file =
        (graph_file_ && !graph_file_->empty()) ? *graph_file_ : "cuvs_cagra_updated.graph";

      cuvs::neighbors::cagra::serialize(handle_, file, *index_, false);
      std::cout << "\n [INFO] Serialized updated CAGRA index to disk after insert." << std::endl;
    } 
    #endif
  }  // end of INSERT block

}

#endif
#if 0
template <typename T, typename IdxT>  
void cuvs_cagra<T, IdxT>::insert_old(const T* vectors, size_t num_vectors, const uint64_t* ids)
{
  (void)ids;
  if (num_vectors == 0) { return; }
  if (vectors == nullptr) {
    throw std::runtime_error("insert requires a non-null vectors pointer when num_vectors > 0");
  }

  if (!index_) {
      throw std::runtime_error("Index must be loaded or built before insert operations.");
  }
  if (!input_dataset_v_ || input_dataset_v_->extent(1) == 0) {
    throw std::runtime_error("Dataset view is not initialized; call set_search_dataset() first.");
  }

  std::cout << "[insert] begin: num_vectors=" << num_vectors
            << " dim=" << dim_
            << " current_graph_rows=" << index_->graph().extent(0)
            << " current_graph_degree=" << index_->graph().extent(1)
            << " current_dataset_rows=" << input_dataset_v_->extent(0)
            << std::endl;

  auto stream = raft::resource::get_cuda_stream(handle_);
  auto batch_size = static_cast<int64_t>(num_vectors);
  auto dim        = static_cast<int64_t>(dim_);
  bool vectors_on_device = raft::get_device_for_address(vectors) >= 0;

  auto graph_view = index_->graph();
  if (graph_view.extent(0) == 0) {
    std::cout << "[insert] graph is empty, delegating to build()" << std::endl;
    build(vectors, num_vectors);
    has_inserted_data_ = true;
    *input_dataset_v_ = raft::make_device_matrix_view<const T, int64_t>(
      dataset_->data_handle(), dataset_->extent(0), this->dim_);
    need_dataset_update_ = false;
    if (insert_params_.persist && graph_file_.has_value() && !graph_file_->empty()) {
      save(*graph_file_);
      std::cout << "[insert] persisted rebuilt index to " << *graph_file_ << std::endl;
    }
    std::cout << "[insert] complete via build: graph_rows=" << index_->graph().extent(0)
              << " dataset_rows=" << index_->data().n_rows() << std::endl;
    return;
  }

  if (graph_view.extent(0) != input_dataset_v_->extent(0)) {
    throw std::runtime_error("Graph node count must match dataset size before insert operations.");
  }

  bool dataset_cache_invalid = (dataset_->extent(0) != input_dataset_v_->extent(0)) ||
                               (dataset_->extent(1) < static_cast<int64_t>(dim_));
  if (index_->dim() == 0 || need_dataset_update_ || dataset_cache_invalid) {
    auto mr = get_mr(dataset_mem_);
    cuvs::neighbors::cagra::detail::copy_with_padding(handle_, *dataset_, *input_dataset_v_, mr);
    auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
      dataset_->data_handle(), dataset_->extent(0), this->dim_, dataset_->extent(1));
    index_->update_dataset(handle_, dataset_view);
    need_dataset_update_ = false;
  }
  if (index_->dim() == 0) {
    throw std::runtime_error("Index has dim=0 after dataset attach; load/build the index first.");
  }

  int64_t old_rows = graph_view.extent(0);
  int64_t cols     = graph_view.extent(1);
  int64_t new_rows = old_rows + batch_size;

  int64_t old_dataset_rows = dataset_->extent(0);
  int64_t old_dataset_cols = dataset_->extent(1);
  if (old_dataset_rows != old_rows || old_dataset_cols < dim) {
    throw std::runtime_error("Dataset storage is not aligned with graph before insert.");
  }

  // copy old dataset to host
  int64_t total_rows = old_dataset_rows + batch_size;
  std::vector<T> host_dataset(static_cast<size_t>(total_rows) * static_cast<size_t>(dim));
  if (old_dataset_rows > 0) {
    std::vector<T> host_dataset_padded(static_cast<size_t>(old_dataset_rows) *
                                       static_cast<size_t>(old_dataset_cols));
    raft::copy(host_dataset_padded.data(),
               dataset_->data_handle(),
               static_cast<size_t>(old_dataset_rows) * static_cast<size_t>(old_dataset_cols),
               stream);
    raft::resource::sync_stream(handle_);
    for (int64_t r = 0; r < old_dataset_rows; ++r) {
      std::memcpy(host_dataset.data() + static_cast<size_t>(r * dim),
                  host_dataset_padded.data() + static_cast<size_t>(r * old_dataset_cols),
                  static_cast<size_t>(dim) * sizeof(T));
    }
  }

  T* host_insert_dst = host_dataset.data() + static_cast<size_t>(old_dataset_rows * dim);
  if (vectors_on_device) {
    raft::copy(host_insert_dst,
               vectors,
               static_cast<size_t>(batch_size) * static_cast<size_t>(dim),
               stream);
  } else {
    std::memcpy(host_insert_dst,
                vectors,
                static_cast<size_t>(batch_size) * static_cast<size_t>(dim) * sizeof(T));
  }
  raft::resource::sync_stream(handle_);

  // Rebuild index using the consolidated dataset.
  build(host_dataset.data(), static_cast<size_t>(total_rows));

  
  auto host_dataset_view = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
    host_dataset.data(), total_rows, dim);
  auto dataset_mr = get_mr(dataset_mem_);
  cuvs::neighbors::cagra::detail::copy_with_padding(handle_, *dataset_, host_dataset_view, dataset_mr);
  auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
    dataset_->data_handle(), dataset_->extent(0), this->dim_, dataset_->extent(1));
  index_->update_dataset(handle_, dataset_view);

  *input_dataset_v_ = raft::make_device_matrix_view<const T, int64_t>(
    dataset_->data_handle(), dataset_->extent(0), this->dim_);
  need_dataset_update_ = false;
  has_inserted_data_   = true;
/*
  if (insert_params_.persist && graph_file_.has_value() && !graph_file_->empty()) {
    save(*graph_file_);
    std::cout << "[insert] persisted updated index to " << *graph_file_ << std::endl;
  }*/

  std::cout << "[insert] complete: old_rows=" << old_rows
            << " inserted=" << batch_size
            << " new_rows=" << new_rows
            << " graph_cols=" << index_->graph().extent(1)
            << " dataset_rows=" << dataset_->extent(0)
            << std::endl;

}
// ...existing code...
#endif

// My multi-threaded implementation
template <typename T, typename IdxT>  
void cuvs_cagra<T, IdxT>::insert(const T* vectors, size_t num_vectors, const uint64_t* ids)
{
  if (num_vectors == 0) { return; }
  if (vectors == nullptr) {
    throw std::runtime_error("insert requires a non-null vectors pointer when num_vectors > 0");
  }

  if (!index_) {
      throw std::runtime_error("Index must be loaded or built before insert operations.");
  }
  if (!input_dataset_v_ || input_dataset_v_->extent(1) == 0) {
    throw std::runtime_error("Dataset view is not initialized; call set_search_dataset() first.");
  }

  std::cout << "[insert] begin: num_vectors=" << num_vectors
            << " dim=" << dim_
            << " current_graph_rows=" << index_->graph().extent(0)
            << " current_graph_degree=" << index_->graph().extent(1)
            << " current_dataset_rows=" << input_dataset_v_->extent(0)
            << std::endl;

  auto stream = raft::resource::get_cuda_stream(handle_);
  bool vectors_on_device = raft::get_device_for_address(vectors) >= 0;

  auto start = std::chrono::high_resolution_clock::now();

  // copy vectors to device if they are not already there, 
  std::unique_ptr<rmm::device_uvector<T>> device_vectors_buf;
  const T* queries = vectors;
  if (!vectors_on_device) {
    device_vectors_buf = std::make_unique<rmm::device_uvector<T>>(
      static_cast<size_t>(num_vectors) * static_cast<size_t>(dim_), stream);
    raft::copy(device_vectors_buf->data(),
               vectors,
               static_cast<size_t>(num_vectors) * static_cast<size_t>(dim_),
               stream);
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    queries = device_vectors_buf->data();
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration_ms = end - start;
  std::cout << "[insert] Copying vectors from host to device took " << duration_ms.count() << " ms for "  << num_vectors << " vectors." << std::endl;

  start = std::chrono::high_resolution_clock::now(); 
  auto graph_view = index_->graph();
  if (graph_view.extent(0) == 0) {
    std::cout << "[insert] graph is empty, delegating to build()" << std::endl;
    build(queries, num_vectors);
    has_inserted_data_ = true;
    if (insert_params_.persist && graph_file_.has_value() && !graph_file_->empty()) {
      save(*graph_file_);
      std::cout << "[insert] persisted rebuilt index to " << *graph_file_ << std::endl;
    }
    std::cout << "[insert] complete via build: graph_rows=" << index_->graph().extent(0)
              << " dataset_rows=" << index_->data().n_rows() << std::endl;
    return;
  }

  if (graph_view.extent(0) != input_dataset_v_->extent(0)) {
    throw std::runtime_error("Graph node count must match dataset size before insert operations.");
  }

  bool dataset_cache_invalid = (dataset_->extent(0) != input_dataset_v_->extent(0)) ||
                               (dataset_->extent(1) < static_cast<int64_t>(dim_));
  if (index_->dim() == 0 || need_dataset_update_ || dataset_cache_invalid) {
    auto mr = get_mr(dataset_mem_);
    cuvs::neighbors::cagra::detail::copy_with_padding(handle_, *dataset_, *input_dataset_v_, mr);
    auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
      dataset_->data_handle(), dataset_->extent(0), this->dim_, dataset_->extent(1));
    index_->update_dataset(handle_, dataset_view);
    need_dataset_update_ = false;
  }
  if (index_->dim() == 0) {
    throw std::runtime_error("Index has dim=0 after dataset attach; load/build the index first.");
  }

  auto batch_size = static_cast<int64_t>(num_vectors);
  auto k          = static_cast<int>(index_->graph().extent(1));
  if (k <= 0) {
    throw std::runtime_error("Index graph has zero out-degree; cannot derive insert neighbors.");
  }

  end = std::chrono::high_resolution_clock::now();
  duration_ms = end - start;
  std::cout << "[insert] validations took " << duration_ms.count() << " ms for " << batch_size << " queries." << std::endl;

  rmm::device_uvector<algo_base::index_type> neighbors_buf(
    static_cast<size_t>(batch_size) * static_cast<size_t>(k), stream);
  rmm::device_uvector<float> distances_buf(
    static_cast<size_t>(batch_size) * static_cast<size_t>(k), stream);
  algo_base::index_type* neighbors = neighbors_buf.data();
  float* distances                 = distances_buf.data();


  // filter_ is only set by set_search_param(); ensure it's valid before calling search_ex.
  if (!filter_) {
    filter_ = std::make_shared<cuvs::neighbors::filtering::none_sample_filter>();
  }
// mesure the time taken for search_ex
  start = std::chrono::high_resolution_clock::now(); 
  search_ex(queries, // on device
            static_cast<int>(batch_size),
            k,
            neighbors, // on device
            distances, // on device
            nullptr);
  raft::resource::sync_stream(handle_);
  end = std::chrono::high_resolution_clock::now();
  duration_ms = end - start;
  std::cout << "[insert] search_ex took " << duration_ms.count() << " ms for " << batch_size << " queries." << std::endl;

#if 0
// debugging 
// for queries 500000 to 500,100 verify if the retrieved neighbors are < 5000,0000
// copy returned neighbors to host 
   std::vector<algo_base::index_type> host_neighbors(static_cast<size_t>(batch_size) *
                                                     static_cast<size_t>(k));
  raft::copy(host_neighbors.data(), neighbors, static_cast<size_t>(batch_size) * static_cast<size_t>(k), stream);
  raft::resource::sync_stream(handle_);
  
// scan the host_neighbors and check if any neighbor is >= 5000000
  for (int64_t i = 0; i < 10; ++i) {
    for (int j = 0; j < k; ++j) {

      // print the ids and the corresponding neighbors for the first 10 queries
      std::printf("[insert] query_id123=%llu neighbor[%d]=%u\n",
                  static_cast<unsigned long long>(ids[i]),
                  j,
                  static_cast<unsigned>(host_neighbors[i * k + j]));
      /*if (host_neighbors[i * k + j] >= 500000) {
        std::cerr << "[insert] Error: neighbor " << host_neighbors[i * k + j]
                  << " for query " << i << " is out of bounds." << std::endl;*/
    }
  } 
#endif

  
  //auto graph_view = index_->graph();
  auto* graph_ptr = reinterpret_cast<const uint32_t*>(graph_view.data_handle());
  auto graph_rows = graph_view.extent(0);
  auto graph_cols = graph_view.extent(1);
  ensure_reverse_counts_buffer(graph_rows);
  ensure_reverse_row_ptr_buffer(graph_rows);
  auto* cached_graph_ptr = graph_ ? reinterpret_cast<const void*>(graph_->data_handle()) : nullptr;
  auto cached_graph_rows = graph_ ? graph_->extent(0) : int64_t{0};
  auto cached_graph_cols = graph_ ? graph_->extent(1) : int64_t{0};
  // get handle to dataset pointer
  auto* dataset_ptr = reinterpret_cast<T*>(dataset_->data_handle());
#if 1
  std::printf(
    "[insert_vectors] num_ids=%zu idx_graph_ptr=%p idx_rows=%lld idx_cols=%lld cached_graph_ptr=%p "
    "cached_rows=%lld cached_cols=%lld dataset_ptr=%p\n",
    num_vectors,
    (const void*)graph_ptr,
    (long long)graph_rows,
    (long long)graph_cols,
    cached_graph_ptr,
    (long long)cached_graph_rows,
    (long long)cached_graph_cols,
    (const void*)dataset_ptr);
 
  detail::launch_cuvs_bang_insert_kernel(ids,
                                         num_vectors,
                                         graph_ptr,
                                         graph_rows,
                                         graph_cols,
                                         d_deleted_rows_,
                                         d_dist_threshold1_,
                                         d_dist_threshold2_,
                                         queries,
                                         dataset_ptr,
                                         dim_,
                                        neighbors,
                                        distances,
                                        k,
                                        reverse_graph_ ? reinterpret_cast<uint32_t*>(reverse_graph_->data_handle()) : nullptr,
                                        reverse_graph_ ? reverse_graph_->extent(1) : int64_t{0},
                                        d_reverse_counts_ ? static_cast<uint32_t*>(d_reverse_counts_->data()) : nullptr);
#endif
  std::cout << "[insert] completed: old_rows=" 
            /*<< old_rows
            << " inserted=" << batch_size
            << " new_rows=" << new_rows
            << " graph_cols=" << cols*/
            << " dataset_rows=" << dataset_->extent(0)
            << std::endl;

 // save index
 #if 0
   if (insert_params_.persist && graph_file_.has_value() && !graph_file_->empty()) {
      save(*graph_file_+"_after_10kinsert");
      std::cout << "[insert] persisted updated index to " << *graph_file_+"_after_10kinsert" << std::endl;
    } 
 #endif     
 // print the reverse graph for debugging
 #if 0
   auto curr_graph = index_->graph();
  auto curr_rows  = curr_graph.extent(0);
  auto curr_cols  = curr_graph.extent(1);
  auto reverse_cols       = get_reverse_graph_cols(curr_cols);
  if (reverse_graph_) {
    auto reverse_graph_view = reverse_graph_->view();
    std::vector<uint32_t> host_reverse_graph(static_cast<size_t>(reverse_graph_view.extent(0) * reverse_graph_view.extent(1)));
    raft::copy(host_reverse_graph.data(),
               reverse_graph_view.data_handle(),
               static_cast<size_t>(reverse_graph_view.extent(0) * reverse_graph_view.extent(1)),
               raft::resource::get_cuda_stream(handle_));
    raft::resource::sync_stream(handle_);
        std::string reverse_graph_file =
      (graph_file_ && !graph_file_->empty()) ? (*graph_file_ + ".reverse_graph_insert.bin")
                                             : std::string("reverse_graph.bin");
  std::ofstream reverse_out(reverse_graph_file, std::ios::binary | std::ios::trunc);
    if (reverse_out.good()) {
      reverse_out.write(reinterpret_cast<const char*>(&curr_rows), sizeof(curr_rows));
      reverse_out.write(reinterpret_cast<const char*>(&reverse_cols), sizeof(reverse_cols));
      reverse_out.write(reinterpret_cast<const char*>(host_reverse_graph.data()),
                        static_cast<std::streamsize>(host_reverse_graph.size() * sizeof(IdxT)));
      reverse_out.close();
    } else {
      std::cerr << "Warning: failed to save reverse graph file to " << reverse_graph_file
                << std::endl;
    }
  }
  #endif
}

template <typename T, typename IdxT>  
void cuvs_cagra<T, IdxT>::delete_vectors(const uint64_t* ids, size_t num_ids)
{
  if (num_ids == 0) { return; }
  if (ids == nullptr) {
    throw std::runtime_error("delete requires a non-null ids pointer when num_ids > 0");
  }
  if (!index_) {
    throw std::runtime_error("delete requires a valid index; call load/build before delete");
  }

  static_assert(sizeof(IdxT) == sizeof(uint32_t), "IdxT must be uint32_t for launch_cuvs_bang_delete_kernel");
  
  auto graph_view = index_->graph();
  auto* graph_ptr = reinterpret_cast<const uint32_t*>(graph_view.data_handle());
  auto graph_rows = graph_view.extent(0);
  auto graph_cols = graph_view.extent(1);
  auto* cached_graph_ptr = graph_ ? reinterpret_cast<const void*>(graph_->data_handle()) : nullptr;
  auto cached_graph_rows = graph_ ? graph_->extent(0) : int64_t{0};
  auto cached_graph_cols = graph_ ? graph_->extent(1) : int64_t{0};
  std::printf(
    "[delete_vectors] num_ids=%zu idx_graph_ptr=%p idx_rows=%lld idx_cols=%lld cached_graph_ptr=%p "
    "cached_rows=%lld cached_cols=%lld\n",
    num_ids,
    (const void*)graph_ptr,
    (long long)graph_rows,
    (long long)graph_cols,
    cached_graph_ptr,
    (long long)cached_graph_rows,
    (long long)cached_graph_cols);
#if 0
  ensure_delete_workspace(num_ids, graph_rows);
#endif

  #if 1
    detail::launch_cuvs_bang_delete_kernel(
      ids,
      num_ids,
      graph_ptr,
      graph_rows,
      graph_cols,
      d_deleted_rows_,
      reverse_graph_ ? reinterpret_cast<uint32_t*>(reverse_graph_->data_handle()) : nullptr,
      reverse_graph_ ? reverse_graph_->extent(1) : int64_t{0},
      d_reverse_row_ptr_ ? static_cast<uint32_t*>(d_reverse_row_ptr_->data()) : nullptr,
      d_reverse_counts_ ? static_cast<uint32_t*>(d_reverse_counts_->data()) : nullptr
      #if 0
      ,
      static_cast<uint32_t*>(d_delete_frontier_curr_->data()),
      static_cast<uint32_t*>(d_delete_frontier_next_->data()),
      static_cast<int*>(d_delete_frontier_size_->data()),
      static_cast<int*>(d_delete_next_size_->data()),
      delete_workspace_frontier_capacity_,
      delete_workspace_max_ids_
      #endif
    );
  #endif

  // log the graph dimensions after deletion for debugging
  raft::resource::sync_stream(handle_);
  auto post_delete_graph_view = index_->graph();
  std::cout << "[delete_vectors] post-delete graph dimensions: rows=" << post_delete_graph_view.extent(0)
            << " cols=" << post_delete_graph_view.extent(1) << std::endl;
#if 0
 auto curr_graph = index_->graph();
  auto curr_rows  = curr_graph.extent(0);
  auto curr_cols  = curr_graph.extent(1);
  auto reverse_cols       = get_reverse_graph_cols(curr_cols);
  if (reverse_graph_) {
    auto reverse_graph_view = reverse_graph_->view();
    std::vector<uint32_t> host_reverse_graph(static_cast<size_t>(reverse_graph_view.extent(0) * reverse_graph_view.extent(1)));
    raft::copy(host_reverse_graph.data(),
               reverse_graph_view.data_handle(),
               static_cast<size_t>(reverse_graph_view.extent(0) * reverse_graph_view.extent(1)),
               raft::resource::get_cuda_stream(handle_));
    raft::resource::sync_stream(handle_); 
        std::string reverse_graph_file =
      (graph_file_ && !graph_file_->empty()) ? (*graph_file_ + ".reverse_graph_delete.bin")
                                             : std::string("reverse_graph.bin");
  std::ofstream reverse_out(reverse_graph_file, std::ios::binary | std::ios::trunc);
    if (reverse_out.good()) {
      reverse_out.write(reinterpret_cast<const char*>(&curr_rows), sizeof(curr_rows));
      reverse_out.write(reinterpret_cast<const char*>(&reverse_cols), sizeof(reverse_cols));
      reverse_out.write(reinterpret_cast<const char*>(host_reverse_graph.data()),
                        static_cast<std::streamsize>(host_reverse_graph.size() * sizeof(IdxT)));
      reverse_out.close();
    } else {
      std::cerr << "Warning: failed to save reverse graph file to " << reverse_graph_file
                << std::endl;
    }
  }
  #endif            
}

/*
template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::insertkernel(const T* vectors, size_t num_vectors, const int64_t* ids)
{

  // spawn a kernel
}
*/
template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::search_base(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);

  auto queries_view = raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, dim_);
  auto neighbors_view =
    raft::make_device_matrix_view<algo_base::index_type, int64_t>(neighbors, batch_size, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k);

  if (dynamic_batcher_) {
    cuvs::neighbors::dynamic_batching::search(handle_,
                                              dynamic_batcher_sp_,
                                              *dynamic_batcher_,
                                              queries_view,
                                              neighbors_view,
                                              distances_view);
  } else {
    if (index_params_.num_dataset_splits <= 1 ||
        index_params_.merge_type == CagraMergeType::kPhysical) {
      cuvs::neighbors::cagra::search(
        handle_, search_params_, *index_, queries_view, neighbors_view, distances_view, *filter_);
    } else {
      if (index_params_.merge_type == CagraMergeType::kLogical) {
        // TODO: index merge must happen outside of search, otherwise what are we benchmarking?
        cuvs::neighbors::cagra::merge_params merge_params{cuvs::neighbors::cagra::index_params{}};
        merge_params.merge_strategy = cuvs::neighbors::MergeStrategy::MERGE_STRATEGY_LOGICAL;

        // Create wrapped indices for composite merge
        std::vector<std::shared_ptr<cuvs::neighbors::IndexWrapper<T, IdxT, algo_base::index_type>>>
          wrapped_indices;
        wrapped_indices.reserve(sub_indices_.size());
        for (auto& ptr : sub_indices_) {
          auto index_wrapper =
            cuvs::neighbors::cagra::make_index_wrapper<T, IdxT, algo_base::index_type>(ptr.get());
          wrapped_indices.push_back(index_wrapper);
        }

        raft::resources composite_handle(handle_);
        size_t n_streams = wrapped_indices.size();
        raft::resource::set_cuda_stream_pool(composite_handle,
                                             std::make_shared<rmm::cuda_stream_pool>(n_streams));

        auto merged_index =
          cuvs::neighbors::composite::merge(composite_handle, merge_params, wrapped_indices);
        cuvs::neighbors::filtering::none_sample_filter empty_filter;
        merged_index->search(composite_handle,
                             search_params_,
                             queries_view,
                             neighbors_view,
                             distances_view,
                             empty_filter);
      }
    }
  }
}

// Returns true if nid and new_row_id share a common neighbor.
// ToDo: Pick the WORST (farthest) common neighbor as replacement candidate.
template <typename T, typename IdxT>
inline bool  cuvs_cagra<T, IdxT>::has_common_neighbour(
    IdxT nid,
    IdxT new_row_id,
    const std::vector<IdxT>& graph,
    int64_t cols,
    int64_t new_rows,
    IdxT& replace_candidate) const
{
    replace_candidate = IdxT(-1);

    // Scan neighbors of nid
    for (int64_t i = 0; i < cols; ++i) {
        IdxT c = graph[nid * cols + i];
        if (c >= new_rows || c == nid) continue;

        // Check if c is also neighbor of new_row_id
        for (int64_t j = 0; j < cols; ++j) {
            if (graph[new_row_id * cols + j] == c) {
                replace_candidate = c;
                return true;
            }
        }
    }
    return false;
}


template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{

  search_ex(queries, batch_size, k, neighbors, distances, nullptr);
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::search_ex(
  const T* queries,
  int batch_size,
  int k,
  algo_base::index_type* neighbors,
  float* distances,
  int64_t* ids) const
{
  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);
  auto start = std::chrono::high_resolution_clock::now(); 
  if (neighbors == nullptr || distances == nullptr) {
    throw std::runtime_error("search_ex requires non-null neighbors and distances pointers");
  }
    // scan the adjacency list of graph and validate all enighbours are < 1000 for debugging
  auto graph_view = index_->graph();
  auto graph_rows = graph_view.extent(0);
  auto graph_cols = graph_view.extent(1);
  #if 0
  // copy graph to host for validation
  std::vector<IdxT> graph_host(static_cast<size_t>(graph_rows * graph_cols), IdxT{0});
  raft::copy(graph_host.data(), graph_view.data_handle(), graph_rows * graph_cols, raft::resource::get_cuda_stream(handle_));
   raft::resource::sync_stream(handle_);
  // print graph dimensions for debugging
#ifdef KVDEBUG  
  std::cout << "[search] graph dimensions: rows=" << graph_rows << " cols=" << graph_cols << std::endl;     
#endif
  for (int64_t r = 0; r < graph_rows; ++r)  {
      for (int64_t c = 0; c < graph_cols; ++c) {
          auto neighbor_id = graph_host[r * graph_cols + c];
          if (neighbor_id >= 1000) {
              //std::cerr << "Invalid neighbor id " << neighbor_id << " at graph[" << r << "][" << c << "]" << std::endl;
              //throw std::runtime_error("Invalid neighbor id found in graph");
          }
      }
  } 
#endif
  auto k0                       = static_cast<size_t>(refine_ratio_ * k);
  const bool disable_refinement = k0 <= static_cast<size_t>(k);
  const raft::resources& res    = handle_;
  const auto stream             = raft::resource::get_cuda_stream(res);
  const auto n_elems = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(k);
  const auto n_query_elems = static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(dim_);

  if (queries == nullptr && batch_size > 0) {
    throw std::runtime_error("search_ex requires a non-null queries pointer when batch_size > 0");
  }

  auto queries_on_device = raft::get_device_for_address(queries) >= 0;
  rmm::device_uvector<T> queries_dev_staging(0, stream);
  const T* search_queries = queries;
  if (!queries_on_device) {
    queries_dev_staging.resize(n_query_elems, stream);
    raft::copy(queries_dev_staging.data(), queries, n_query_elems, stream);
    search_queries = queries_dev_staging.data();
  }

  auto neighbors_on_device = raft::get_device_for_address(neighbors) >= 0;
  auto distances_on_device = raft::get_device_for_address(distances) >= 0;
  if (neighbors_on_device != distances_on_device) {
    throw std::runtime_error("neighbors and distances must both be on device or both on host");
  }

  rmm::device_uvector<algo_base::index_type> neighbors_dev_staging(0, stream);
  rmm::device_uvector<float> distances_dev_staging(0, stream);
  auto* search_neighbors = neighbors;
  auto* search_distances = distances;

  if (!neighbors_on_device) {
    // allocate staging buffers on device if output buffers are on host, to avoid cudaMemcpyDeviceToHost in refinement step
    neighbors_dev_staging.resize(n_elems, stream);
    distances_dev_staging.resize(n_elems, stream);
    search_neighbors = neighbors_dev_staging.data();
    search_distances = distances_dev_staging.data();
  }

  auto exec_mem_type = raft::get_device_for_address(search_neighbors) >= 0
                         ? MemoryType::kDevice
                         : MemoryType::kHostPinned;

  // NOTE: caching mem_type to reduce mutex locks 
  // raft::get_device_for_address call cuda API to get the pointer properties, (But, still we are calling get_dev_for_addr everytime)
  // this means it locks the context mutex for a very small amount of time.
  // In the event of thread contention (such as thousands threads), this time can actually increase.
  // Hence we try to bypass this check for repeated search calls.

  // If the same output pointer is reused across calls, its memory location (host/device) should not change, 
  // so they reuse cached mem_type instead of re-deriving it.
  thread_local MemoryType mem_type                   = MemoryType::kDevice;
  thread_local algo_base::index_type* prev_neighbors = nullptr;
  if (prev_neighbors != search_neighbors) {
    prev_neighbors = search_neighbors;
    mem_type       = exec_mem_type;
  }

  // If dynamic batching is used and there's no sync between benchmark laps, multiple sequential
  // requests can group together. The data is copied asynchronously, and if the same intermediate
  // buffer is used for multiple requests, they can override each other's data. Hence, we need to
  // allocate as much space as required by the maximum number of sequential requests.
  auto max_dyn_grouping = dynamic_batcher_ ? raft::div_rounding_up_safe<int64_t>(
                                               dynamic_batching_max_batch_size_, batch_size) *
                                               dynamic_batching_n_queues_
                                           : 1;
  auto tmp_buf_size =
    ((disable_refinement ? 0 : (sizeof(float) + sizeof(algo_base::index_type)))) * batch_size * k0;
  auto& tmp_buf = get_tmp_buffer_from_global_pool(tmp_buf_size * max_dyn_grouping);
  thread_local static int64_t group_id = 0;
  auto* candidates_ptr                 = reinterpret_cast<algo_base::index_type*>(
    reinterpret_cast<uint8_t*>(tmp_buf.data(mem_type)) + tmp_buf_size * group_id);
  group_id = (group_id + 1) % max_dyn_grouping;
  auto* candidate_dists_ptr =
    reinterpret_cast<float*>(candidates_ptr + (disable_refinement ? 0 : batch_size * k0));

  if (disable_refinement) {
    search_base(search_queries, batch_size, k, search_neighbors, search_distances);
  } else {
    search_base(search_queries, batch_size, k0, candidates_ptr, candidate_dists_ptr);

    if (mem_type == MemoryType::kHostPinned && uses_stream()) {
      // If the algorithm uses a stream to synchronize (non-persistent kernel), but the data is in
      // the pinned host memory, we need to synchronize before the refinement operation to wait for
      // the data being available for the host.
      raft::resource::sync_stream(res);
    }

    auto candidate_ixs =
      raft::make_device_matrix_view<const algo_base::index_type, algo_base::index_type>(
        candidates_ptr, batch_size, k0);
    auto queries_v =
      raft::make_device_matrix_view<const T, algo_base::index_type>(
        search_queries, batch_size, dim_);
    refine_helper(
      res,
      *input_dataset_v_,
      queries_v,
      candidate_ixs,
      k,
      search_neighbors,
      search_distances,
      index_->metric());
  }

  // only if neighbors and distances are on host, we copy back the results from device to host. 
  // If they are already on device, we can directly use them without copy.
  if (!neighbors_on_device) {
    raft::copy(neighbors, search_neighbors, n_elems, stream);
    raft::copy(distances, search_distances, n_elems, stream);
    raft::resource::sync_stream(res);
  }
  // ids is expected to be on host, so we always copy back the neighbor indices if ids is not nullptr, regardless of where neighbors are.
  if (ids != nullptr) {
    if (neighbors_on_device) {
      if constexpr (std::is_same_v<algo_base::index_type, int64_t>) {
        raft::copy(ids, search_neighbors, n_elems, stream);
        raft::resource::sync_stream(res);
      } else {
        std::vector<algo_base::index_type> host_neighbors(n_elems);
        raft::copy(host_neighbors.data(), search_neighbors, n_elems, stream);
        raft::resource::sync_stream(res);
        for (std::size_t idx = 0; idx < n_elems; ++idx) {
          ids[idx] = static_cast<int64_t>(host_neighbors[idx]);
        }
      }
    } else {
      if constexpr (std::is_same_v<algo_base::index_type, int64_t>) {
        std::memcpy(ids, neighbors, n_elems * sizeof(algo_base::index_type));
      } else {
        for (std::size_t idx = 0; idx < n_elems; ++idx) {
          ids[idx] = static_cast<int64_t>(neighbors[idx]);
        }
      }
    }
  }
  #if 0
  // print first k neighbours and distances of first and second query for debugging
  {
    constexpr std::size_t kDebugTopN = 10;
    auto debug_count                 = std::min<std::size_t>(kDebugTopN, static_cast<std::size_t>(k));
    if (batch_size > 0) {
      if (neighbors_on_device) {
        // Allocate for first and second query results
        std::size_t copy_count = std::min<std::size_t>(debug_count * 2, n_elems);
        std::vector<algo_base::index_type> host_neighbors(copy_count);
        std::vector<float> host_distances(copy_count);
        raft::copy(host_neighbors.data(), search_neighbors, copy_count, stream);
        raft::copy(host_distances.data(), search_distances, copy_count, stream);
        raft::resource::sync_stream(res);
        
        // Print first query
        std::cout << "[search_ex] query 0 neighbors: ";
        for (size_t i = 0; i < debug_count; ++i) {
          std::cout << host_neighbors[i] << "(" << host_distances[i] << ") ";
        }
        std::cout << std::endl;
        
        // Print second query if available
        if (batch_size > 1) {
          std::cout << "[search_ex] query 1 neighbors: ";
          for (size_t i = 0; i < debug_count && (debug_count + i) < copy_count; ++i) {
            std::cout << host_neighbors[debug_count + i] << "(" << host_distances[debug_count + i] << ") ";
          }
          std::cout << std::endl;
        }
      } else {
        // Print first query
        std::cout << "[search_ex] query 0 neighbors: ";
        for (size_t i = 0; i < debug_count; ++i) {
          std::cout << neighbors[i] << "(" << distances[i] << ") ";
        }
        std::cout << std::endl;
        
        // Print second query if available
        if (batch_size > 1) {
          std::cout << "[search_ex] query 1 neighbors: ";
          for (size_t i = 0; i < debug_count; ++i) {
            std::cout << neighbors[static_cast<size_t>(k) + i] << "(" 
                      << distances[static_cast<size_t>(k) + i] << ") ";
          }
          std::cout << std::endl;
        }
      }
    }
  }
  #endif
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> duration_ms = end - start;
  std::cout << "[search_ex] search_ex took " << duration_ms.count() << " ms for " << batch_size << " queries." << std::endl;  
}
}  // namespace cuvs::bench
