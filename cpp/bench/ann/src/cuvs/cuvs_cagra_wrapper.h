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
                                    int recall_at_k);

void launch_cuvs_bang_delete_kernel(const uint64_t* ids,
                                    size_t num_ids,
                                    const uint32_t* graph,
                                    int64_t graph_rows,
                                    int64_t graph_cols,
                                    const std::shared_ptr<rmm::device_buffer>& d_deleted_rows);
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

  inline bool  has_common_neighbour(
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
  // END: For delete experiments
  // Max rows the dataset can expand upto
  int rows_;
   // dataset size supplied at build time
  int build_rows_;

  std::shared_ptr<rmm::device_buffer> d_dist_threshold1_;
  std::shared_ptr<rmm::device_buffer> d_dist_threshold2_;

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
    
  // Enable cuvs logging for debugging
  raft::default_logger().set_level( rapids_logger::level_enum::info);
  if (sp.graph_mem != graph_mem_) {
    // Move graph to correct memory space
    graph_mem_ = sp.graph_mem;
    RAFT_LOG_DEBUG("moving graph to new memory space: %s", allocator_to_string(graph_mem_).c_str());
    // We create a new graph and copy to it from existing graph
    auto mr = get_mr(graph_mem_);

    // Create a new graph, then copy, and __only then__ replace the shared pointer.
    auto old_graph =
      index_->graph();  // view of graph_ if it exists, of an internal index member otherwise
    auto new_graph = raft::make_device_mdarray<IdxT, int64_t>(handle_, mr, old_graph.extents());
    raft::copy(new_graph.data_handle(),
               old_graph.data_handle(),
               old_graph.size(),
               raft::resource::get_cuda_stream(handle_));
    raft::resource::sync_stream(handle_);
    *graph_ = std::move(new_graph);

    // NB: update_graph() only stores a view in the index. We need to keep the graph object alive.
    index_->update_graph(handle_, make_const_mdspan(graph_->view()));
    needs_dynamic_batcher_update = true;
  }

  if (sp.dataset_mem != dataset_mem_ || need_dataset_update_) {
    dataset_mem_ = sp.dataset_mem;

    // First free up existing memory
    *dataset_ = raft::make_device_matrix<T, int64_t>(handle_, 0, 0);
    index_->update_dataset(handle_, make_const_mdspan(dataset_->view()));

    // Allocate space using the correct memory resource.
    RAFT_LOG_DEBUG("moving dataset to new memory space: %s",
                   allocator_to_string(dataset_mem_).c_str());

    auto mr = get_mr(dataset_mem_);
    cuvs::neighbors::cagra::detail::copy_with_padding(handle_, *dataset_, *input_dataset_v_, mr);

    auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
      dataset_->data_handle(), dataset_->extent(0), this->dim_, dataset_->extent(1));
    index_->update_dataset(handle_, dataset_view);

    need_dataset_update_         = false;
    needs_dynamic_batcher_update = true;
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
  auto graph = index_->graph();
  auto dataset = index_->dataset();
  std::cout << "[cuvs_cagra::set_search_param] graph extents: (" << graph.extent(0) << ", " << graph.extent(1) << ")" << std::endl;
  std::cout << "[cuvs_cagra::set_search_param] dataset extents: (" << dataset.extent(0) << ", " << dataset.extent(1) << ")" << std::endl; 
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::set_search_dataset(const T* dataset, size_t nrow)
{
  
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
    // log error
    std::cerr << "Warning: set_search_dataset called with nrow=" << nrow
              << " which does not match the number of rows used during build time="
              << index_->graph().extent(0) << ". This may lead to unexpected behavior." << std::endl;
              return;
  } 


  auto stream              = raft::resource::get_cuda_stream(handle_);
  const auto expected_rows = rows_ > 0 ? static_cast<size_t>(build_rows_) : size_t{0};
  if (expected_rows != 0 && nrow != expected_rows) { // relaxed the check when running as EXE (CUVS_CAGRA_ANN_BENCH , legacy flow)
    // log error
    throw std::runtime_error(
      "set_search_dataset: search dataset row count (" + std::to_string(nrow) +
      ") does not match initial dataset row count used during algo creation (" +
      std::to_string(expected_rows) + ").");
  }
  auto old_graph   = index_->graph();
  auto old_rows    = old_graph.extent(0);
  auto cols        = old_graph.extent(1);
  auto target_rows = static_cast<int64_t>(rows_);

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
    bool dataset_is_on_host = raft::get_device_for_address(active_dataset) == -1;
    IdxT rows_per_split =
      raft::ceildiv<IdxT>(active_nrow, static_cast<IdxT>(index_params_.num_dataset_splits));
    for (size_t i = 0; i < sub_indices_.size(); ++i) {
      IdxT start = static_cast<IdxT>(i * rows_per_split);
      if (start >= active_nrow) break;
      IdxT rows        = std::min(rows_per_split, static_cast<IdxT>(active_nrow) - start);
      const T* sub_ptr = active_dataset + static_cast<size_t>(start) * dim_;
      auto sub_host =
        raft::make_host_matrix_view<const T, int64_t, raft::row_major>(sub_ptr, rows, dim_);
      auto sub_dev =
        raft::make_device_matrix_view<const T, int64_t, raft::row_major>(sub_ptr, rows, dim_);
      auto sub_index = sub_indices_[i].get();
      if (index_params_.merge_type == CagraMergeType::kLogical) {
        if (dataset_is_on_host) {
          sub_index->update_dataset(handle_, sub_host);
        } else {
          sub_index->update_dataset(handle_, sub_dev);
        }
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

    if (static_cast<size_t>(input_dataset_v_->extent(0)) != active_nrow ||
        input_dataset_v_->data_handle() != active_dataset) {
      bool dataset_is_on_host = raft::get_device_for_address(active_dataset) == -1;
      if (dataset_is_on_host) {
        auto host_dataset_view =
          raft::make_host_matrix_view<const T, int64_t, raft::row_major>(active_dataset, active_nrow, this->dim_);
        auto dataset_mr = get_mr(dataset_mem_);
        cuvs::neighbors::cagra::detail::copy_with_padding(
          handle_, *dataset_, host_dataset_view, dataset_mr);
        auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
          dataset_->data_handle(), this->rows_, this->dim_, dataset_->extent(1));
        index_->update_dataset(handle_, dataset_view);

        *input_dataset_v_ =
          raft::make_device_matrix_view<const T, int64_t>(dataset_->data_handle(), this->rows_, this->dim_);
 
        need_dataset_update_ = false;
      } else {
        // Question: should dataset_ and index_->update_dataset() also be updated in this case? 
        *input_dataset_v_ =
          raft::make_device_matrix_view<const T, int64_t>(active_dataset, active_nrow, this->dim_);
        need_dataset_update_ = !is_vpq;  // ignore update if this is a VPQ dataset.
        // log an error and return
        std::cerr << "Warning: set_search_dataset called with a device pointer. "
                  << std::endl;
        return;
      }
    }
  }

  // If we are attaching to a bigger dataset than the current graph size,
  // expand graph_ by replicating the last valid row.

  std::cout << "[cuvs_cagra::set_search_dataset] expanding old graph, old_rows=" << old_rows
            << ", target_rows=" << target_rows << std::endl;
  if (old_rows < target_rows) {
    auto mr                 = get_mr(graph_mem_);
    auto stream             = raft::resource::get_cuda_stream(handle_);
    auto new_expanded_graph = raft::make_device_mdarray<IdxT, int64_t>(
      handle_, mr, raft::make_extents<int64_t>(target_rows, cols));

    if (old_rows > 0) {
      raft::copy(
        new_expanded_graph.data_handle(), old_graph.data_handle(), old_graph.size(), stream);
      raft::resource::sync_stream(handle_);

    }

    *graph_ = std::move(new_expanded_graph);
    index_->update_graph(handle_, make_const_mdspan(graph_->view()));

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
  // expecting only legacy constructor to call load
  if (rows_ != 0)
  {
    // throw exception
    throw std::runtime_error(
      "load: algo was created with non-zero rows (" + std::to_string(rows_) +
      ") but loaded index has " + std::to_string(index_->graph().extent(0)) + " rows.");   
  }
  rows_ = index_->graph().extent(0);
  
  d_deleted_rows_         = std::make_shared<rmm::device_buffer>(
    static_cast<size_t>(rows_) * sizeof(uint8_t), handle_.get_sync_stream(), get_mr(AllocatorType::kDevice));
  d_dist_threshold1_ = std::make_shared<rmm::device_buffer>(
    static_cast<size_t>(rows_) * sizeof(float), handle_.get_sync_stream(), get_mr(AllocatorType::kDevice));
  d_dist_threshold2_ = std::make_shared<rmm::device_buffer>(
    static_cast<size_t>(rows_) * sizeof(float), handle_.get_sync_stream(), get_mr(AllocatorType::kDevice));
  
  //ToDo: Temp hack to let cuvs_bench know which are valid/delete rows in the graph for --search option
  RAFT_CUDA_TRY(cudaMemsetAsync(
    d_deleted_rows_->data(), 0, static_cast<size_t>(9000) * sizeof(uint8_t), handle_.get_sync_stream()));
    raft::resource::sync_stream(handle_);
  
}

template <typename T, typename IdxT>
std::unique_ptr<algo<T>> cuvs_cagra<T, IdxT>::copy()
{
  return std::make_unique<cuvs_cagra<T, IdxT>>(std::cref(*this));  // use copy constructor
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::set_insert_param(const insert_param& param)
{
  insert_params_ = param;
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
void cuvs_cagra<T, IdxT>::insert(const T* vectors, size_t num_vectors, const uint64_t* ids)
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

  std::cout << "[insert-single] begin: num_vectors=" << num_vectors
            << " dim=" << dim_
            << " current_graph_rows=" << index_->graph().extent(0)
            << " current_graph_degree=" << index_->graph().extent(1)
            << " current_dataset_rows=" << input_dataset_v_->extent(0) << std::endl;

  auto stream            = raft::resource::get_cuda_stream(handle_);
  bool vectors_on_device = raft::get_device_for_address(vectors) >= 0;

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

  auto graph_view = index_->graph();
  if (graph_view.extent(0) == 0) {
    std::cout << "[insert-single] graph is empty, delegating to build()" << std::endl;
    build(queries, num_vectors);
    has_inserted_data_ = true;
    if (insert_params_.persist && graph_file_.has_value() && !graph_file_->empty()) {
      save(*graph_file_);
      std::cout << "[insert-single] persisted rebuilt index to " << *graph_file_ << std::endl;
    }
    std::cout << "[insert-single] complete via build: graph_rows=" << index_->graph().extent(0)
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

  rmm::device_uvector<algo_base::index_type> neighbors_buf(
    static_cast<size_t>(batch_size) * static_cast<size_t>(k), stream);
  rmm::device_uvector<float> distances_buf(
    static_cast<size_t>(batch_size) * static_cast<size_t>(k), stream);
  algo_base::index_type* neighbors = neighbors_buf.data();
  float* distances                 = distances_buf.data();

  if (!filter_) {
    filter_ = std::make_shared<cuvs::neighbors::filtering::none_sample_filter>();
  }

  search_ex(queries, static_cast<int>(batch_size), k, neighbors, distances, nullptr);
  raft::resource::sync_stream(handle_);

  bool queries_on_device = raft::get_device_for_address(queries) >= 0;
  std::vector<T> host_queries(static_cast<size_t>(batch_size) * static_cast<size_t>(dim_));
  if (queries_on_device) {
    raft::copy(host_queries.data(),
               queries,
               static_cast<size_t>(batch_size) * static_cast<size_t>(dim_),
               stream);
  } else {
    std::memcpy(host_queries.data(),
                queries,
                static_cast<size_t>(batch_size) * static_cast<size_t>(dim_) * sizeof(T));
  }

  std::vector<algo_base::index_type> host_neighbors(static_cast<size_t>(batch_size) *
                                                    static_cast<size_t>(k));
  raft::copy(host_neighbors.data(),
             neighbors,
             static_cast<size_t>(batch_size) * static_cast<size_t>(k),
             stream);
  raft::resource::sync_stream(handle_);

  auto old_graph  = index_->graph();
  int64_t old_rows = old_graph.extent(0);
  int64_t cols     = old_graph.extent(1);
  int64_t new_rows = old_rows + batch_size;

  std::vector<IdxT> host_graph(static_cast<size_t>(new_rows * cols), IdxT{0});
  if (old_rows > 0) {
    raft::copy(host_graph.data(),
               old_graph.data_handle(),
               static_cast<size_t>(old_rows) * static_cast<size_t>(cols),
               stream);
    raft::resource::sync_stream(handle_);
  }

  int64_t dim              = static_cast<int64_t>(dim_);
  int64_t old_dataset_rows = dataset_->extent(0);
  int64_t old_dataset_cols = dataset_->extent(1);
  if (old_dataset_rows != old_rows || old_dataset_cols < dim) {
    throw std::runtime_error("Dataset storage is not aligned with graph before insert.");
  }

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

  // Single-thread baseline insert path.
  for (int64_t row = 0; row < batch_size; ++row) {
    std::memcpy(host_dataset.data() + static_cast<size_t>((old_dataset_rows + row) * dim),
                host_queries.data() + static_cast<size_t>(row * dim),
                static_cast<size_t>(dim) * sizeof(T));

    int64_t new_row_id = old_rows + row;
    std::vector<IdxT> adjacency;
    adjacency.reserve(static_cast<size_t>(cols));

    for (int j = 0; j < k && adjacency.size() < static_cast<size_t>(cols); ++j) {
      auto nid = static_cast<IdxT>(host_neighbors[static_cast<size_t>(row) *
                                                  static_cast<size_t>(k) +
                                                  static_cast<size_t>(j)]);
      auto nid_i64 = static_cast<int64_t>(nid);
      if (nid_i64 < 0 || nid_i64 >= old_rows) { continue; }
      if (std::find(adjacency.begin(), adjacency.end(), nid) == adjacency.end()) {
        adjacency.push_back(nid);
      }
    }

    if (adjacency.empty() && old_rows > 0) { adjacency.push_back(static_cast<IdxT>(0)); }
    while (adjacency.size() < static_cast<size_t>(cols)) {
      adjacency.push_back(adjacency.empty() ? static_cast<IdxT>(0) : adjacency.front());
    }

    for (int64_t c = 0; c < cols; ++c) {
      host_graph[static_cast<size_t>(new_row_id * cols + c)] = adjacency[static_cast<size_t>(c)];
    }

    bool inserted_any = false;
    for (IdxT nid : adjacency) {
      auto nid_i64 = static_cast<int64_t>(nid);
      if (nid_i64 < 0 || nid_i64 >= old_rows || nid_i64 == new_row_id) { continue; }

      bool already_present = false;
      for (int64_t c = 0; c < cols; ++c) {
        if (host_graph[static_cast<size_t>(nid_i64 * cols + c)] == static_cast<IdxT>(new_row_id)) {
          already_present = true;
          break;
        }
      }
      if (already_present) { continue; }

      IdxT replace_candidate{};
      if (!has_common_neighbour(
            static_cast<IdxT>(nid_i64), static_cast<IdxT>(new_row_id), host_graph, cols, new_rows, replace_candidate)) {
        continue;
      }

      for (int64_t c = 0; c < cols; ++c) {
        if (host_graph[static_cast<size_t>(nid_i64 * cols + c)] == replace_candidate) {
          host_graph[static_cast<size_t>(nid_i64 * cols + c)] = static_cast<IdxT>(new_row_id);
          inserted_any = true;
          break;
        }
      }
    }

    if (!inserted_any && !adjacency.empty()) {
      auto nid_i64 = static_cast<int64_t>(adjacency.front());
      if (nid_i64 >= 0 && nid_i64 < old_rows) {
        host_graph[static_cast<size_t>(nid_i64 * cols)] = static_cast<IdxT>(new_row_id);
      }
    }
  }

  auto host_dataset_view = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
    host_dataset.data(), total_rows, dim);
  auto dataset_mr = get_mr(dataset_mem_);
  cuvs::neighbors::cagra::detail::copy_with_padding(handle_, *dataset_, host_dataset_view, dataset_mr);
  auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
    dataset_->data_handle(), dataset_->extent(0), this->dim_, dataset_->extent(1));
  index_->update_dataset(handle_, dataset_view);

  auto graph_mr = get_mr(graph_mem_);
  *graph_       = raft::make_device_mdarray<IdxT, int64_t>(
    handle_, graph_mr, raft::make_extents<int64_t>(new_rows, cols));
  raft::copy(graph_->data_handle(),
             host_graph.data(),
             static_cast<size_t>(new_rows) * static_cast<size_t>(cols),
             stream);
  raft::resource::sync_stream(handle_);
  index_->update_graph(handle_, make_const_mdspan(graph_->view()));

  *input_dataset_v_ = raft::make_device_matrix_view<const T, int64_t>(
    dataset_->data_handle(), dataset_->extent(0), this->dim_);
  need_dataset_update_ = false;
  has_inserted_data_   = true;

  if (insert_params_.persist && graph_file_.has_value() && !graph_file_->empty()) {
    save(*graph_file_);
    std::cout << "[insert-single] persisted updated index to " << *graph_file_ << std::endl;
  }

  std::cout << "[insert-single] complete: old_rows=" << old_rows
            << " inserted=" << batch_size
            << " new_rows=" << new_rows
            << " graph_cols=" << cols
            << " dataset_rows=" << dataset_->extent(0) << std::endl;
}

#else
template <typename T, typename IdxT>  
void cuvs_cagra<T, IdxT>::insert_wait(const T* vectors, size_t num_vectors, const uint64_t* ids)
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

  search_ex(queries, // on device
            static_cast<int>(batch_size),
            k,
            neighbors, // on device
            distances, // on device
            nullptr);
  raft::resource::sync_stream(handle_);

  //auto graph_view = index_->graph();
  auto* graph_ptr = reinterpret_cast<const uint32_t*>(graph_view.data_handle());
  auto graph_rows = graph_view.extent(0);
  auto graph_cols = graph_view.extent(1);
  auto* cached_graph_ptr = graph_ ? reinterpret_cast<const void*>(graph_->data_handle()) : nullptr;
  auto cached_graph_rows = graph_ ? graph_->extent(0) : int64_t{0};
  auto cached_graph_cols = graph_ ? graph_->extent(1) : int64_t{0};
  // get handle to dataset pointer
  auto* dataset_ptr = reinterpret_cast<T*>(dataset_->data_handle());

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
                                         vectors,
                                         dataset_ptr,
                                         dim_,
                                        neighbors,
                                        distances,
                                        k);

 #if 0
  bool queries_on_device = raft::get_device_for_address(queries) >= 0;
  std::vector<T> host_queries(static_cast<size_t>(batch_size) * static_cast<size_t>(dim_));
  if (queries_on_device) {
    raft::copy(host_queries.data(),
               queries,
               static_cast<size_t>(batch_size) * static_cast<size_t>(dim_),
               stream);
  } else {
    std::memcpy(host_queries.data(),
                queries,
                static_cast<size_t>(batch_size) * static_cast<size_t>(dim_) * sizeof(T));
  }

  std::vector<algo_base::index_type> host_neighbors(static_cast<size_t>(batch_size) *
                                                    static_cast<size_t>(k));
  raft::copy(host_neighbors.data(),
             neighbors,
             static_cast<size_t>(batch_size) * static_cast<size_t>(k),
             stream);
  raft::resource::sync_stream(handle_);

  auto old_graph  = index_->graph();
  int64_t old_rows = old_graph.extent(0);
  int64_t cols     = old_graph.extent(1);
  //int64_t new_rows = old_rows + batch_size;

  std::vector<IdxT> host_graph(static_cast<size_t>(new_rows * cols), IdxT{0});
  if (old_rows > 0) {
    raft::copy(host_graph.data(),
               old_graph.data_handle(),
               static_cast<size_t>(old_rows * cols),
               stream);
    raft::resource::sync_stream(handle_);
  }

  int64_t dim = static_cast<int64_t>(dim_);
  int64_t old_dataset_rows = dataset_->extent(0);
  int64_t old_dataset_cols = dataset_->extent(1);
  if (old_dataset_rows != old_rows || old_dataset_cols < dim) {
    throw std::runtime_error("Dataset storage is not aligned with graph before insert.");
  }

  // copy old dataset to host
  int64_t total_rows = old_dataset_rows ;
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

  // Per-row adjacency lists computed in Phase 1 and consumed in Phase 2.
  std::vector<std::vector<IdxT>> per_row_adjacency(static_cast<size_t>(batch_size));

  // ---------------------------------------------------------------
  // Phase 1 (fully parallel): dataset copy + forward edge wiring
  //
  // Safe because every thread writes to a unique region:
  //   host_dataset: offset (id[row])*dim  — unique per row
  //   host_graph:   row  new_row_id = old_rows+row     — unique per row
  // ---------------------------------------------------------------
#pragma omp parallel for schedule(static)
  for (int64_t row = 0; row < batch_size; ++row) {
    // copy the basedataset 
    std::memcpy(host_dataset.data() + static_cast<size_t>((id[row]) * dim),
                host_queries.data() + static_cast<size_t>(row * dim),
                static_cast<size_t>(dim) * sizeof(T));

    //int64_t new_row_id = old_rows + row;
    // Set this row as not deleted
    RAFT_CUDA_TRY(cudaMemsetAsync(
      (uint8_t*)d_deleted_rows_->data()+(id[row]), 0, 1, handle_.get_sync_stream()));
    
    std::vector<IdxT> adjacency;
    adjacency.reserve(static_cast<size_t>(cols));

    for (int j = 0; j < k && adjacency.size() < static_cast<size_t>(cols); ++j) {
      auto nid = static_cast<IdxT>(host_neighbors[(static_cast<size_t>(row) * static_cast<size_t>(k)) +
                                                  static_cast<size_t>(j)]);
      auto nid_i64 = static_cast<int64_t>(nid);
      if (nid_i64 < 0 || nid_i64 >= old_rows) { continue; }
      // copy d_deleted_rows_ to host and check if the neighbor is marked as deleted
      uint8_t deleted_flag = 1;
      RAFT_CUDA_TRY(cudaMemcpyAsync(&deleted_flag,
                                     (uint8_t*)d_deleted_rows_->data() + nid,
                                     1,
                                     cudaMemcpyDeviceToHost,
                                     handle_.get_sync_stream()));
      RAFT_CUDA_TRY(cudaStreamSynchronize(handle_.get_sync_stream()));
      if (deleted_flag  == 1)
      {
        // log an error message here, as this should not happen ideally.
        std::cerr << "[cuvs_cagra_wrapper] Encountered deleted neighbor: " << nid << " for row: "
        << id[static_cast<size_t>(row)] << std::endl;
        continue;  // skip deleted neighbor
      }
      bool exists = std::find(adjacency.begin(), adjacency.end(), nid) != adjacency.end();
      if (!exists) { adjacency.push_back(nid); }
    }
    // collected the neighbours of the newly inserted vector

    if (adjacency.empty() && old_rows > 0) { adjacency.push_back(static_cast<IdxT>(0)); }

    // padded the adjacency list to have fixed size of cols (out degree)
    while (adjacency.size() < static_cast<size_t>(cols)) {
      adjacency.push_back(adjacency.empty() ? static_cast<IdxT>(0) : adjacency.front());
    }

    for (int64_t c = 0; c < cols; ++c) {
      host_graph[static_cast<size_t>(new_row_id * cols + c)] = adjacency[static_cast<size_t>(c)];
    }

    per_row_adjacency[static_cast<size_t>(row)] = std::move(adjacency);
  }  // end Phase 1

  // ---------------------------------------------------------------
  // Phase 2 (naive lock-free): each inserted row updates its own
  // local copy of old rows, then we merge old rows by selecting one
  // row-local version per old row.
  // ---------------------------------------------------------------
  std::vector<IdxT> host_graph_old(static_cast<size_t>(old_rows * cols));
  if (old_rows > 0) {
    std::copy(host_graph.begin(),
              host_graph.begin() + static_cast<size_t>(old_rows * cols),
              host_graph_old.begin());
  }

  std::vector<std::vector<IdxT>> row_local_old_graphs(static_cast<size_t>(batch_size),
                                                       host_graph_old);
  std::vector<std::vector<char>> row_old_row_touched(
    static_cast<size_t>(batch_size), std::vector<char>(static_cast<size_t>(old_rows), 0));

#pragma omp parallel for schedule(dynamic)
  for (int64_t row = 0; row < batch_size; ++row) {
    auto& local_graph = row_local_old_graphs[static_cast<size_t>(row)];
    auto& row_touched = row_old_row_touched[static_cast<size_t>(row)];

    int64_t new_row_id    = old_rows + row;
    // lookup adjacency of the newly inserted row
    const auto& adjacency = per_row_adjacency[static_cast<size_t>(row)];
    bool inserted_any     = false;

    for (auto nid : adjacency) {
      int64_t nid_i64 = static_cast<int64_t>(nid);
      if (nid_i64 < 0 || nid_i64 >= old_rows || nid_i64 == new_row_id) { continue; }

      bool already_present = false;
      for (int64_t c = 0; c < cols; ++c) {
        if (local_graph[static_cast<size_t>(nid_i64 * cols + c)] == static_cast<IdxT>(new_row_id)) {
          already_present = true;
          break;
        }
      }
      if (already_present) { continue; }

      IdxT replace_candidate{};
      bool found_common = false;
      // scan the neighbors of nid_i64 i.e. forward edge 
      for (int64_t c = 0; c < cols; ++c) {
        auto cand = local_graph[static_cast<size_t>(nid_i64 * cols + c)];
        auto cand_i64 = static_cast<int64_t>(cand);
        if (cand_i64 < 0 || cand_i64 >= old_rows || cand == nid || cand == static_cast<IdxT>(new_row_id)) {
          continue;
        }
        // check if the neighbour in each fowrward edge (i.e. cand) is already present 
        // in current adj list (adjacency) of newly inserted point
        if (std::find(adjacency.begin(), adjacency.end(), cand) != adjacency.end()) {
          replace_candidate = cand;
          found_common      = true;
          break;
        }
      }
      if (!found_common) { continue; }
      // Do the actual re-wiring of the reverse edge
      for (int64_t c = 0; c < cols; ++c) {
        if (local_graph[static_cast<size_t>(nid_i64 * cols + c)] == replace_candidate) {
          local_graph[static_cast<size_t>(nid_i64 * cols + c)] = static_cast<IdxT>(new_row_id);
          inserted_any = true;
          row_touched[static_cast<size_t>(nid_i64)] = 1;
          break;
        }
      }
    }
    // fallback, if no reverse edges added for the newly inserted row then forcible add one reverse edges 
    // with first entry in adjacency list
    if (!inserted_any && !adjacency.empty()) {
      int64_t nid_i64 = static_cast<int64_t>(adjacency.front());
      if (nid_i64 >= 0 && nid_i64 < old_rows) {
        local_graph[static_cast<size_t>(nid_i64 * cols)] = static_cast<IdxT>(new_row_id);
        row_touched[static_cast<size_t>(nid_i64)] = 1;
      }
    }
  }  // end Phase 2

  // Merge old rows by selecting one row-local row copy (naive policy).
  std::vector<IdxT> host_graph_final(static_cast<size_t>(new_rows * cols), IdxT{0});
  //std::mt19937 rng(12345);
  int64_t touched_old_rows = 0;
  int64_t candidate_sum    = 0;
   unsigned uUniqueNeighbours = 0;
   unsigned uTotalNeighbours = 0;
   // for every old row, find the right version of the adj list
  // Host-side squared L2 distance helper — mirrors the per-element accumulation
  // performed by compute_distance (dataset_descriptor_base_t::compute_distance) on
  // the device, but runs on host_dataset which is already resident in CPU memory.
  auto host_l2_sq = [&](int64_t row_a, int64_t row_b) -> float {
    float dist = 0.f;
    const T* a = host_dataset.data() + static_cast<size_t>(row_a) * static_cast<size_t>(dim);
    const T* b = host_dataset.data() + static_cast<size_t>(row_b) * static_cast<size_t>(dim);
    for (int64_t d = 0; d < dim; ++d) {
      float diff = static_cast<float>(a[d]) - static_cast<float>(b[d]);
      dist += diff * diff;
    }
    return dist;
  };
  for (int64_t r = 0; r < old_rows; ++r) {
    // Its a 2D matrix
    // Rows : every newly inserted row, cols: which olds rows adj list got updates.
    // scan col major manner to know how many version of adj list are there for each old row
    std::vector<int64_t> candidate_versions;
    candidate_versions.reserve(static_cast<size_t>(batch_size));
    for (int64_t row_version = 0; row_version < batch_size; ++row_version) {
      if (row_old_row_touched[static_cast<size_t>(row_version)][static_cast<size_t>(r)] != 0) {
        candidate_versions.push_back(row_version);
      }
    }

    if (!candidate_versions.empty()) {
      ++touched_old_rows;
      candidate_sum += static_cast<int64_t>(candidate_versions.size());
    }
   
    if (candidate_versions.empty()) {
      std::copy(host_graph_old.begin() + static_cast<size_t>(r * cols),
                host_graph_old.begin() + static_cast<size_t>((r + 1) * cols),
                host_graph_final.begin() + static_cast<size_t>(r * cols));
    } else {
      /*std::uniform_int_distribution<size_t> dist(0, candidate_versions.size() - 1);
      int64_t chosen_version = candidate_versions[dist(rng)];
      const auto& chosen_graph = row_local_old_graphs[static_cast<size_t>(chosen_version)];*/
      // Compute overlap metrics across all row-local versions for this old row.
      std::vector<IdxT> vecUnion;
      for (size_t uIter = 0; uIter < candidate_versions.size(); ++uIter) {
        int64_t version_id = candidate_versions[uIter];
        const auto& version_graph = row_local_old_graphs[static_cast<size_t>(version_id)];
        vecUnion.insert(vecUnion.end(),
                        version_graph.begin() + static_cast<size_t>(r * cols),
                        version_graph.begin() + static_cast<size_t>((r + 1) * cols));
      }

      uTotalNeighbours += static_cast<unsigned>(vecUnion.size());
      std::unordered_set<IdxT> unique_elements(vecUnion.begin(), vecUnion.end());
      uUniqueNeighbours += static_cast<unsigned>(unique_elements.size());

      // Compute distances from row r to all unique neighbors, then select the closest 'cols' ones.
      // Build (distance, neighbor_id) pairs for all valid unique neighbors.
      std::vector<std::pair<float, IdxT>> dist_neighbor_pairs;
      dist_neighbor_pairs.reserve(unique_elements.size());
      for (auto nb : unique_elements) {
        auto nb_i64 = static_cast<int64_t>(nb);
        if (nb_i64 >= 0 && nb_i64 < new_rows) {
          float d = host_l2_sq(r, nb_i64);
          dist_neighbor_pairs.push_back({d, nb});
        }
      }
      
      // Sort by distance (ascending) to get closest neighbors first.
      std::sort(dist_neighbor_pairs.begin(), dist_neighbor_pairs.end());
      
      // Fill host_graph_final[r] with the closest 'cols' neighbors.
      int64_t num_to_take = std::min<int64_t>(cols, static_cast<int64_t>(dist_neighbor_pairs.size()));
      for (int64_t c = 0; c < num_to_take; ++c) {
        host_graph_final[static_cast<size_t>(r * cols + c)] = dist_neighbor_pairs[static_cast<size_t>(c)].second;
      }
      
      // Pad remaining slots with the first (closest) neighbor if needed.
      if (num_to_take < cols) {
        IdxT pad_val = num_to_take > 0 ? dist_neighbor_pairs[0].second : static_cast<IdxT>(0);
        for (int64_t c = num_to_take; c < cols; ++c) {
          host_graph_final[static_cast<size_t>(r * cols + c)] = pad_val;
        }
      }
  
    }
  }

  // Append new rows from the global host_graph built in Phase 1.
  if (batch_size > 0) {
    std::copy(host_graph.begin() + static_cast<size_t>(old_rows * cols),
              host_graph.begin() + static_cast<size_t>(new_rows * cols),
              host_graph_final.begin() + static_cast<size_t>(old_rows * cols));
  }
  host_graph = std::move(host_graph_final);

  double avg_candidates =
    (touched_old_rows > 0) ? (static_cast<double>(candidate_sum) / touched_old_rows) : 0.0;
  std::cout << "[insert][naive-merge] touched_old_rows=" << touched_old_rows << "/"
            << old_rows << " avg_candidate_rows_per_touched_old_row=" << avg_candidates
            << " common_neighbors=" << uTotalNeighbours - uUniqueNeighbours << " /" << uTotalNeighbours
            << std::endl;

  auto host_dataset_view = raft::make_host_matrix_view<const T, int64_t, raft::row_major>(
    host_dataset.data(), total_rows, dim);
  auto dataset_mr = get_mr(dataset_mem_);
  cuvs::neighbors::cagra::detail::copy_with_padding(handle_, *dataset_, host_dataset_view, dataset_mr);
  auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
    dataset_->data_handle(), dataset_->extent(0), this->dim_, dataset_->extent(1));
  index_->update_dataset(handle_, dataset_view);

  auto graph_mr = get_mr(graph_mem_);
  *graph_ = raft::make_device_mdarray<IdxT, int64_t>(
    handle_, graph_mr, raft::make_extents<int64_t>(new_rows, cols));
  raft::copy(graph_->data_handle(),
             host_graph.data(),
             static_cast<size_t>(new_rows) * static_cast<size_t>(cols),
             stream);
  raft::resource::sync_stream(handle_);
  index_->update_graph(handle_, make_const_mdspan(graph_->view()));
  

  *input_dataset_v_ = raft::make_device_matrix_view<const T, int64_t>(
    dataset_->data_handle(), dataset_->extent(0), this->dim_);
  need_dataset_update_ = false;
  has_inserted_data_   = true;

  if (insert_params_.persist && graph_file_.has_value() && !graph_file_->empty()) {
    save(*graph_file_);
    std::cout << "[insert] persisted updated index to " << *graph_file_ << std::endl;
  }
#endif
  std::cout << "[insert] complete: old_rows=" 
            /*<< old_rows
            << " inserted=" << batch_size
            << " new_rows=" << new_rows
            << " graph_cols=" << cols*/
            << " dataset_rows=" << dataset_->extent(0)
            << std::endl;
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
  detail::launch_cuvs_bang_delete_kernel(ids, num_ids, graph_ptr, graph_rows, graph_cols, d_deleted_rows_);
  // log the graph dimensions after deletion for debugging
  raft::resource::sync_stream(handle_);
  auto post_delete_graph_view = index_->graph();
  std::cout << "[delete_vectors] post-delete graph dimensions: rows=" << post_delete_graph_view.extent(0)
            << " cols=" << post_delete_graph_view.extent(1) << std::endl;
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

  if (neighbors == nullptr || distances == nullptr) {
    throw std::runtime_error("search_ex requires non-null neighbors and distances pointers");
  }
    // scan the adjacency list of graph and validate all enighbours are < 1000 for debugging
  auto graph_view = index_->graph();
  auto graph_rows = graph_view.extent(0);
  auto graph_cols = graph_view.extent(1);
  // copy graph to host for validation
  std::vector<IdxT> graph_host(static_cast<size_t>(graph_rows * graph_cols), IdxT{0});
  raft::copy(graph_host.data(), graph_view.data_handle(), graph_rows * graph_cols, raft::resource::get_cuda_stream(handle_));
   raft::resource::sync_stream(handle_);
  // print graph dimensions for debugging
  std::cout << "[search] graph dimensions: rows=" << graph_rows << " cols=" << graph_cols << std::endl;     
  
  for (int64_t r = 0; r < graph_rows; ++r)  {
      for (int64_t c = 0; c < graph_cols; ++c) {
          auto neighbor_id = graph_host[r * graph_cols + c];
          if (neighbor_id >= 1000) {
              //std::cerr << "Invalid neighbor id " << neighbor_id << " at graph[" << r << "][" << c << "]" << std::endl;
              //throw std::runtime_error("Invalid neighbor id found in graph");
          }
      }
  } 

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
}
}  // namespace cuvs::bench
