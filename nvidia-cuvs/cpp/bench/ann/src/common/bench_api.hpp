/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "ann_types.hpp"
#include "conf.hpp"
#include "dataset.hpp"
#include "util.hpp"

#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace cuvs::bench {
namespace detail {

struct lib_handle {
  void* handle{nullptr};
  explicit lib_handle(const std::string& name)
  {
#ifdef _WIN32
    (void)name;
    throw std::runtime_error("Dynamic loading is not supported on Windows.");
#else
    handle = dlopen(name.c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (handle == nullptr) {
      auto error_msg = "Failed to load " + name;
      auto err       = dlerror();
      if (err != nullptr && err[0] != '\0') { error_msg += ": " + std::string(err); }
      throw std::runtime_error(error_msg);
    }
#endif
  }
  ~lib_handle() noexcept
  {
#ifndef _WIN32
    if (handle != nullptr) { dlclose(handle); }
#endif
std::cout << "lib d'tor: "  << std::endl;
  }
};

inline auto load_lib(const std::string& algo) -> void*
{
#ifdef _WIN32
  (void)algo;
  throw std::runtime_error("Dynamic loading is not supported on Windows.");
#else
  static std::unordered_map<std::string, lib_handle> libs{};
  auto lib_short_name = algo;
  if (algo.rfind("cuvs_mg_", 0) == 0) { lib_short_name = "cuvs_mg"; }

  auto found = libs.find(lib_short_name);
  if (found != libs.end()) { return found->second.handle; }

  auto lib_name = "lib" + lib_short_name + "_ann_bench.so";
  return libs.emplace(lib_short_name, lib_name).first->second.handle;
#endif
}

inline auto load_lib_raft_compat(const std::string& algo) -> void*
{
#ifndef _WIN32
  try {
    return load_lib(algo);
  } catch (std::runtime_error& e) {
    if (algo.rfind("raft", 0) == 0) { return load_lib("cuvs" + algo.substr(4)); }
    throw e;
  }
#else
  (void)algo;
  throw std::runtime_error("Dynamic loading is not supported on Windows.");
#endif
}

inline auto get_fun_name(void* addr) -> std::string
{
#ifdef _WIN32
  (void)addr;
  throw std::runtime_error("Dynamic loading is not supported on Windows.");
#else
  Dl_info dl_info;
  if (dladdr(addr, &dl_info) != 0) {
    if (dl_info.dli_sname != nullptr && dl_info.dli_sname[0] != '\0') {
      return std::string{dl_info.dli_sname};
    }
  }
  throw std::logic_error("Failed to find out name of the looked up function");
#endif
}

inline auto pick_index(const configuration& conf,
                       const std::string& index_name,
                       std::size_t index_pos) -> const configuration::index&
{
  const auto& indices = conf.get_indices();
  if (!index_name.empty()) {
    for (const auto& idx : indices) {
      if (idx.name == index_name) { return idx; }
    }
    throw std::runtime_error("Index name not found in config: " + index_name);
  }
  if (index_pos >= indices.size()) {
    throw std::runtime_error("Index position is out of range: " + std::to_string(index_pos));
  }
  return indices[index_pos];
}

inline auto normalize_prefix(const std::string& prefix, const std::string& fallback)
  -> std::string
{
  return prefix.empty() ? fallback : prefix;
}

}  // namespace detail

template <typename T>
inline std::string get_create_algo_symbol_name();

template <typename T>
inline std::string get_create_algo_c_symbol_name();

template <>
inline std::string get_create_algo_c_symbol_name<float>()
{
  return "cuvs_bench_create_algo_float";
}

template <>
inline std::string get_create_algo_c_symbol_name<half>()
{
  return "cuvs_bench_create_algo_half";
}

template <>
inline std::string get_create_algo_c_symbol_name<std::uint8_t>()
{
  return "cuvs_bench_create_algo_uint8";
}

template <>
inline std::string get_create_algo_c_symbol_name<std::int8_t>()
{
  return "cuvs_bench_create_algo_int8";
}

template <>
inline std::string get_create_algo_symbol_name<float>() {
  return "_ZN4cuvs5bench11create_algoIfEESt10unique_ptrINS0_4algoIT_EESt14default_deleteIS5_EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESG_iRKN8nlohmann16json_abi_v3_12_010basic_jsonISt3mapSt6vectorSE_blmdSaNSI_14adl_serializerESL_IhSaIhEEvEE";
}

template <>
inline std::string get_create_algo_symbol_name<half>() {
  return "_ZN4cuvs5bench11create_algoI6__halfEESt10unique_ptrINS0_4algoIT_EESt14default_deleteIS6_EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESH_iRKN8nlohmann16json_abi_v3_12_010basic_jsonISt3mapSt6vectorSF_blmdSaNSJ_14adl_serializerESM_IhSaIhEEvEE";
}

template <>
inline std::string get_create_algo_symbol_name<std::uint8_t>() {
  return "_ZN4cuvs5bench11create_algoIhEESt10unique_ptrINS0_4algoIT_EESt14default_deleteIS5_EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESG_iRKN8nlohmann16json_abi_v3_12_010basic_jsonISt3mapSt6vectorSE_blmdSaNSI_14adl_serializerESL_IhSaIhEEvEE";
}

template <>
inline std::string get_create_algo_symbol_name<std::int8_t>() {
  return "_ZN4cuvs5bench11create_algoIaEESt10unique_ptrINS0_4algoIT_EESt14default_deleteIS5_EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESG_iRKN8nlohmann16json_abi_v3_12_010basic_jsonISt3mapSt6vectorSE_blmdSaNSI_14adl_serializerESL_IhSaIhEEvEE";
}

template <typename T>
inline auto create_algo(const std::string& algo,
                        const std::string& distance,
                        int dim,
                        const nlohmann::json& conf) -> std::unique_ptr<cuvs::bench::algo<T>>
{
#ifdef _WIN32
  (void)algo;
  (void)distance;
  (void)dim;
  (void)conf;
  throw std::runtime_error("Dynamic loading is not supported on Windows.");
#else
  auto c_fname      = get_create_algo_c_symbol_name<T>();
  auto mangled_name = get_create_algo_symbol_name<T>();
  auto handle       = detail::load_lib_raft_compat(algo);
  auto fun_addr     = dlsym(handle, c_fname.c_str());
  if (fun_addr == nullptr) {
    // Backward compatibility with older libraries exposing only mangled symbols.
    fun_addr = dlsym(handle, mangled_name.c_str());
  }
  if (fun_addr == nullptr) {
    throw std::runtime_error("Couldn't load the create_algo function (" + algo + "): " +
                             std::string(dlerror()));
  }
  auto fun = reinterpret_cast<decltype(&create_algo<T>)>(fun_addr);
  return fun(algo, distance, dim, conf);
#endif
}

template <typename T>
inline std::string get_create_search_param_symbol_name();

template <typename T>
inline std::string get_create_search_param_c_symbol_name();

template <>
inline std::string get_create_search_param_c_symbol_name<float>()
{
  return "cuvs_bench_create_search_param_float";
}

template <>
inline std::string get_create_search_param_c_symbol_name<half>()
{
  return "cuvs_bench_create_search_param_half";
}

template <>
inline std::string get_create_search_param_c_symbol_name<std::uint8_t>()
{
  return "cuvs_bench_create_search_param_uint8";
}

template <>
inline std::string get_create_search_param_c_symbol_name<std::int8_t>()
{
  return "cuvs_bench_create_search_param_int8";
}

template <>
inline std::string get_create_search_param_symbol_name<float>() {
  return "_ZN4cuvs5bench19create_search_paramIfEESt10unique_ptrINS0_4algoIT_E12search_paramESt14default_deleteIS6_EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKN8nlohmann16json_abi_v3_12_010basic_jsonISt3mapSt6vectorSF_blmdSaNSJ_14adl_serializerESM_IhSaIhEEvEE";
}

template <>
inline std::string get_create_search_param_symbol_name<half>() {
  return "_ZN4cuvs5bench19create_search_paramI6__halfEESt10unique_ptrINS0_4algoIT_E12search_paramESt14default_deleteIS7_EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKN8nlohmann16json_abi_v3_12_010basic_jsonISt3mapSt6vectorSG_blmdSaNSK_14adl_serializerESN_IhSaIhEEvEE";
}

template <>
inline std::string get_create_search_param_symbol_name<std::uint8_t>() {
  return "_ZN4cuvs5bench19create_search_paramIhEESt10unique_ptrINS0_4algoIT_E12search_paramESt14default_deleteIS6_EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKN8nlohmann16json_abi_v3_12_010basic_jsonISt3mapSt6vectorSF_blmdSaNSJ_14adl_serializerESM_IhSaIhEEvEE";
}

template <>
inline std::string get_create_search_param_symbol_name<std::int8_t>() {
  return "_ZN4cuvs5bench19create_search_paramIaEESt10unique_ptrINS0_4algoIT_E12search_paramESt14default_deleteIS6_EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKN8nlohmann16json_abi_v3_12_010basic_jsonISt3mapSt6vectorSF_blmdSaNSJ_14adl_serializerESM_IhSaIhEEvEE";
}

template <typename T>
inline auto create_search_param(const std::string& algo, const nlohmann::json& conf)
  -> std::unique_ptr<typename cuvs::bench::algo<T>::search_param>
{
#ifdef _WIN32
  (void)algo;
  (void)conf;
  throw std::runtime_error("Dynamic loading is not supported on Windows.");
#else
  auto c_fname      = get_create_search_param_c_symbol_name<T>();
  auto mangled_name = get_create_search_param_symbol_name<T>();
  auto handle       = detail::load_lib_raft_compat(algo);
  auto fun_addr     = dlsym(handle, c_fname.c_str());
  if (fun_addr == nullptr) {
    // Backward compatibility with older libraries exposing only mangled symbols.
    fun_addr = dlsym(handle, mangled_name.c_str());
  }
  if (fun_addr == nullptr) {
    throw std::runtime_error("Couldn't load the create_search_param function (" + algo + "): " +
                             std::string(dlerror()));
  }
  auto fun = reinterpret_cast<decltype(&create_search_param<T>)>(fun_addr);
  return fun(algo, conf);
#endif
}

template <typename T>
inline auto make_dataset(const configuration::dataset_conf& dataset_conf,
                         bool use_filtering) -> std::shared_ptr<const dataset<T>>
{
  auto filtering = use_filtering ? dataset_conf.filtering_rate : std::nullopt;
  return std::make_shared<bench::dataset<T>>(dataset_conf.name,
                                             dataset_conf.base_file,
                                             dataset_conf.subset_first_row,
                                             dataset_conf.subset_size,
                                             dataset_conf.query_file,
                                             dataset_conf.distance,
                                             dataset_conf.groundtruth_neighbors_file,
                                             filtering);
}

inline auto select_query_count(std::size_t query_set_size, std::size_t requested) -> std::size_t
{
  if (requested == 0) { return query_set_size; }
  return std::min(query_set_size, requested);
}

inline void ensure_stream_open(std::ifstream& conf_stream, const std::string& conf_path)
{
  if (!conf_stream) {
    throw std::runtime_error("Cannot open configuration file: " + conf_path);
  }
}

inline void validate_index_file(const std::string& index_file, bool allow_overwrite)
{
  if (file_exists(index_file) && !allow_overwrite) {
    throw std::runtime_error("Index file already exists: " + index_file);
  }
}

inline void ensure_index_exists(const std::string& index_file)
{
  if (!file_exists(index_file)) {
    throw std::runtime_error("Index file is missing: " + index_file);
  }
}

inline void ensure_search_params(const configuration::index& index, std::size_t search_param_ix)
{
  if (index.search_params.empty()) {
    throw std::runtime_error("search_params is empty for index: " + index.name);
  }
  if (search_param_ix >= index.search_params.size()) {
    throw std::runtime_error("search_param_ix is out of range for index: " + index.name);
  }
}

inline auto parse_algo_property_override(const algo_property& prop,
                                         const nlohmann::json& conf) -> algo_property
{
  auto updated = prop;
  if (conf.contains("dataset_memory_type")) {
    updated.dataset_memory_type = parse_memory_type(conf.at("dataset_memory_type"));
  }
  if (conf.contains("query_memory_type")) {
    updated.query_memory_type = parse_memory_type(conf.at("query_memory_type"));
  }
  return updated;
}

inline void set_thread_context() noexcept
{
  benchmark_thread_id = 0;
  benchmark_n_threads = 1;
}

inline auto get_search_k(const nlohmann::json& sp_json) -> std::uint32_t
{
  if (!sp_json.contains("k")) {
    throw std::runtime_error("search_params must contain 'k'");
  }
  return sp_json.at("k");
}

inline auto get_query_count_from_conf(const nlohmann::json& sp_json) -> std::size_t
{
  if (!sp_json.contains("n_queries")) { return 0; }
  return sp_json.at("n_queries");
}

// Global algo cache to avoid reloading from disk when build->insert->search are called in sequence
namespace {
struct algo_cache_entry {
  void* algo_ptr;         // Type-erased algo pointer
  std::string algo_name;  // Algorithm name for type validation
  std::string dtype;      // Data type string
  int dim;                // Dimension for validation
  std::chrono::steady_clock::time_point timestamp;
  std::function<void(void*)> deleter;  // Custom deleter to properly cleanup algo object
};

inline auto get_algo_cache_map() -> std::unordered_map<std::string, algo_cache_entry>&
{
  static std::unordered_map<std::string, algo_cache_entry> g_algo_cache;
  return g_algo_cache;
}

inline auto get_algo_cache_mutex() -> std::mutex&
{
  static std::mutex g_algo_cache_mutex;
  return g_algo_cache_mutex;
}

inline auto get_bench_api_mutex() -> std::recursive_mutex&
{
  static std::recursive_mutex g_bench_api_mutex;
  return g_bench_api_mutex;
}

template <typename T>
inline void sync_algo_before_python_return(const std::unique_ptr<algo<T>>& algo_obj)
{
#ifndef BUILD_CPU_ONLY
  if (!algo_obj) { return; }

  // Ensure all GPU work is complete before returning to Cython/Python.
  // This prevents Python-side buffers from being reclaimed while kernels
  // are still reading/writing through pointers passed into C++.
  if (auto gpu_ann = dynamic_cast<algo_gpu*>(algo_obj.get()); gpu_ann != nullptr) {
    if (gpu_ann->uses_stream()) {
      auto stream = gpu_ann->get_sync_stream();
      if (stream != nullptr) {
        cudaStreamSynchronize(stream);
        return;
      }
    }
  }
  cudaDeviceSynchronize();
#else
  (void)algo_obj;
#endif
}

template <typename T>
void cache_algo(const std::string& index_file,
                const std::string& algo_name,
                const std::string& dtype,
                int dim,
                std::unique_ptr<algo<T>>& algo_obj)
{
  sync_algo_before_python_return(algo_obj);

  std::lock_guard<std::mutex> lock(get_algo_cache_mutex());
  algo_cache_entry entry;
  entry.algo_ptr   = algo_obj.release();
  entry.algo_name  = algo_name;
  entry.dtype      = dtype;
  entry.dim        = dim;
  entry.timestamp  = std::chrono::steady_clock::now();
  
  // Store deleter function to properly cleanup this type
  entry.deleter = [](void* ptr) {
    if (ptr) {
      delete static_cast<algo<T>*>(ptr);
    }
  };
  
  get_algo_cache_map()[index_file] = entry;
}

template <typename T>
auto get_cached_algo(const std::string& index_file,
                     const std::string& algo_name,
                     const std::string& dtype,
                     int dim) -> std::unique_ptr<algo<T>>
{
  auto& cache = get_algo_cache_map();
  std::lock_guard<std::mutex> lock(get_algo_cache_mutex());
  auto it = cache.find(index_file);
  if (it != cache.end()) {
    auto& entry = it->second;
    // Validate algo matches
    if (entry.algo_name == algo_name && entry.dtype == dtype && entry.dim == dim) {
      // Check if cache entry is recent (within 5 minutes)
      auto elapsed = std::chrono::steady_clock::now() - entry.timestamp;
      if (std::chrono::duration_cast<std::chrono::minutes>(elapsed).count() < 5) {
        std::unique_ptr<algo<T>> result(static_cast<algo<T>*>(entry.algo_ptr));
        cache.erase(it);  // Remove from cache as we're transferring ownership
        std::cout << "[ALGO_CACHE] Using cached algo for: " << index_file << std::endl;
        return result;
      }
    }
  }
  return nullptr;
}

inline void clear_algo_cache()
{
  auto& cache = get_algo_cache_map();
  std::lock_guard<std::mutex> lock(get_algo_cache_mutex());
  
  // Properly destroy all cached algo objects using their stored deleters
  // This ensures GPU cleanup happens while CUDA context is still alive
  for (auto& pair : cache) {
    auto& entry = pair.second;
    if (entry.algo_ptr && entry.deleter) {
      entry.deleter(entry.algo_ptr);
      entry.algo_ptr = nullptr;
    }
  }
  
  cache.clear();
}

template <typename T>
inline auto get_dtype_string() -> std::string
{
  if (std::is_same_v<T, float>) return "float";
  if (std::is_same_v<T, half>) return "half";
  if (std::is_same_v<T, std::uint8_t>) return "uint8";
  if (std::is_same_v<T, std::int8_t>) return "int8";
  return "unknown";
}
}  // namespace

template <typename T>
inline void build_from_config(const std::string& conf_path,
                              const std::string& index_name,
                              std::size_t index_pos,
                              const std::string& data_prefix,
                              const std::string& index_prefix,
                              bool force_overwrite)
{
  std::lock_guard<std::recursive_mutex> api_lock(get_bench_api_mutex());

  auto data_prefix_norm  = detail::normalize_prefix(data_prefix, "data");
  auto index_prefix_norm = detail::normalize_prefix(index_prefix, "index");
  std::ifstream conf_stream(conf_path);
  ensure_stream_open(conf_stream, conf_path);
  auto& conf = configuration::initialize(conf_stream, data_prefix_norm, index_prefix_norm);
  const auto& index = detail::pick_index(conf, index_name, index_pos);

  set_thread_context();

  validate_index_file(index.file, force_overwrite);

  auto dataset = make_dataset<T>(conf.get_dataset_conf(), false);
  // print dataset dim and distance for debugging
  std::cout << "[DEBUG bench_api] dataset dim: " << dataset->dim() << ", distance: " << dataset->distance() << std::endl;
  std::cout << "[DEBUG bench_api] build_param: " << index.build_param.dump() << std::endl;

  std::unique_ptr<algo<T>> algo;
  algo = create_algo<T>(index.algo, dataset->distance(), dataset->dim(), index.build_param);

  std::cout << "[DEBUG bench_api] algo created for: " << index.algo << std::endl;

  std::cout << "[DEBUG bench_api] About to call get_preference" << std::endl;
  const auto algo_property = parse_algo_property_override(algo->get_preference(), index.build_param);
  std::cout << "[DEBUG bench_api] get_preference completed"  << std::endl;

  std::cout << "[DEBUG bench_api] About to call dataset->base_set" << std::endl;
  const T* base_set        = dataset->base_set(algo_property.dataset_memory_type);
  std::cout << "[DEBUG bench_api] dataset->base_set completed" << std::endl;
  std::size_t index_size   = dataset->base_set_size();

  std::cout << "[DEBUG bench_api] About to call algo->build(), index_size=" << index_size << std::endl;
  make_sure_parent_dir_exists(index.file);
  algo->set_build_output_file(index.file);
  auto build_t0 = std::chrono::steady_clock::now();
  algo->build(base_set, index_size);
  auto build_t1 = std::chrono::steady_clock::now();
  auto build_ms = std::chrono::duration_cast<std::chrono::milliseconds>(build_t1 - build_t0).count();
  std::cout << "[DEBUG bench_api] algo->build() elapsed_ms=" << build_ms << std::endl;
  std::cout << "[DEBUG bench_api] algo->build() returned successfully" << std::endl;

  // Cache the algo for subsequent insert/search operations
  cache_algo<T>(index.file, index.algo, get_dtype_string<T>(), dataset->dim(), algo);
  std::cout << "[ALGO_CACHE] Cached algo for: " << index.file << std::endl;
}

template <typename T>
inline void get_search_shape_from_config(const std::string& conf_path,
                                         const std::string& index_name,
                                         std::size_t index_pos,
                                         std::size_t search_param_ix,
                                         std::size_t query_count,
                                         const std::string& data_prefix,
                                         const std::string& index_prefix,
                                         std::size_t* out_queries,
                                         std::size_t* out_k)
{
  auto data_prefix_norm  = detail::normalize_prefix(data_prefix, "data");
  auto index_prefix_norm = detail::normalize_prefix(index_prefix, "index");
  std::ifstream conf_stream(conf_path);
  ensure_stream_open(conf_stream, conf_path);
  auto& conf = configuration::initialize(conf_stream, data_prefix_norm, index_prefix_norm);
  const auto& index = detail::pick_index(conf, index_name, index_pos);
  ensure_search_params(index, search_param_ix);

  auto dataset = make_dataset<T>(conf.get_dataset_conf(), true);

  const auto& sp_json   = index.search_params[search_param_ix];
  std::uint32_t k       = get_search_k(sp_json);
  std::size_t available = dataset->query_set_size();
  std::size_t selected  = select_query_count(available, query_count);

  if (out_queries != nullptr) { *out_queries = selected; }
  if (out_k != nullptr) { *out_k = static_cast<std::size_t>(k); }
}

template <typename T>
inline void search_from_config(const std::string& conf_path,
                               const std::string& index_name,
                               std::size_t index_pos,
                               std::size_t search_param_ix,
                               std::size_t query_count,
                               const std::string& data_prefix,
                               const std::string& index_prefix,
                               algo_base::index_type* neighbors,
                               float* distances,
                               int64_t* search_ids,
                               std::size_t* out_queries,
                               std::size_t* out_k)
{
  std::lock_guard<std::recursive_mutex> api_lock(get_bench_api_mutex());

  #if 1
  auto data_prefix_norm  = detail::normalize_prefix(data_prefix, "data");
  auto index_prefix_norm = detail::normalize_prefix(index_prefix, "index");
  std::ifstream conf_stream(conf_path);
  ensure_stream_open(conf_stream, conf_path);
  auto& conf = configuration::initialize(conf_stream, data_prefix_norm, index_prefix_norm);
  const auto& index = detail::pick_index(conf, index_name, index_pos);
  ensure_search_params(index, search_param_ix);

  set_thread_context();

  ensure_index_exists(index.file);

  auto dataset = make_dataset<T>(conf.get_dataset_conf(), true);

  const auto& sp_json = index.search_params[search_param_ix];
  std::uint32_t k     = get_search_k(sp_json);

  std::unique_ptr<typename algo<T>::search_param> search_param;
  std::unique_ptr<algo<T>> algo = get_cached_algo<T>(
    index.file, index.algo, get_dtype_string<T>(), dataset->dim());

  if (!algo) {
    algo = create_algo<T>(index.algo, dataset->distance(), dataset->dim(), index.build_param);
    algo->load(index.file);
    std::cout << "[SEARCH] Loading index from: " << index.file << std::endl;
  } else {
    std::cout << "[SEARCH] Using in-memory algo from build/insert" << std::endl;
  }
  search_param = create_search_param<T>(index.algo, sp_json);

  const auto algo_property = parse_algo_property_override(algo->get_preference(), sp_json);

  if (search_param->needs_dataset()) {
    algo->set_search_dataset(dataset->base_set(algo_property.dataset_memory_type),
                             dataset->base_set_size());
  }
  algo->set_search_param(*search_param,
                         dataset->filter_bitset(algo_property.dataset_memory_type));

  const T* query_set = dataset->query_set(algo_property.query_memory_type);
  std::size_t available_queries = dataset->query_set_size();
  std::size_t selected_queries  = select_query_count(available_queries, query_count);

  if (neighbors == nullptr || distances == nullptr) {
    throw std::runtime_error("neighbors and distances buffers must be provided");
  }

  algo->search_ex(query_set,
                  static_cast<int>(selected_queries),
                  static_cast<int>(k),
                  neighbors,
                  distances,
                  search_ids);

  cache_algo<T>(index.file, index.algo, get_dtype_string<T>(), dataset->dim(), algo);

  if (out_queries != nullptr) { *out_queries = selected_queries; }
  if (out_k != nullptr) { *out_k = static_cast<std::size_t>(k); }
  cudaDeviceSynchronize();
  #endif
}

template <typename T>
inline void search_from_config_array(const std::string& conf_path,
                                     const std::string& index_name,
                                     std::size_t index_pos,
                                     std::size_t search_param_ix,
                                     const void* queries,
                                     std::size_t query_count,
                                     std::size_t query_dim,
                                     const std::string& data_prefix,
                                     const std::string& index_prefix,
                                     algo_base::index_type* neighbors,
                                     float* distances,
                                     int64_t* search_ids,
                                     std::size_t* out_queries,
                                     std::size_t* out_k)
{
  std::lock_guard<std::recursive_mutex> api_lock(get_bench_api_mutex());

  auto data_prefix_norm  = detail::normalize_prefix(data_prefix, "data");
  auto index_prefix_norm = detail::normalize_prefix(index_prefix, "index");
  std::ifstream conf_stream(conf_path);
  ensure_stream_open(conf_stream, conf_path);
  auto& conf = configuration::initialize(conf_stream, data_prefix_norm, index_prefix_norm);
  const auto& index = detail::pick_index(conf, index_name, index_pos);
  ensure_search_params(index, search_param_ix);

  set_thread_context();

  ensure_index_exists(index.file);

  if (queries == nullptr) { throw std::runtime_error("queries pointer must not be null"); }
  if (neighbors == nullptr || distances == nullptr) {
    throw std::runtime_error("neighbors and distances buffers must be provided");
  }

  auto dataset = make_dataset<T>(conf.get_dataset_conf(), true);
  if (query_dim != static_cast<std::size_t>(dataset->dim())) {
    throw std::runtime_error("Query vectors dimension (" + std::to_string(query_dim) +
                             ") does not match dataset dimension (" +
                             std::to_string(dataset->dim()) + ")");
  }

  const auto& sp_json = index.search_params[search_param_ix];
  std::uint32_t k     = get_search_k(sp_json);

  std::unique_ptr<typename algo<T>::search_param> search_param;

  // Try to get cached algo first (from recent build/insert)
  std::unique_ptr<algo<T>> algo = get_cached_algo<T>(
    index.file, index.algo, get_dtype_string<T>(), dataset->dim());

  if (!algo) {
    // No cached algo, need to create and load from disk
    std::cout << "[SEARCH] Loading index from: " << index.file << std::endl;
    algo = create_algo<T>(index.algo, dataset->distance(), dataset->dim(), index.build_param);
    algo->load(index.file);
    std::cout << "[SEARCH] Index loaded successfully" << std::endl;
  } else {
    std::cout << "[SEARCH] Using in-memory algo from build/insert" << std::endl;
  }

  search_param = create_search_param<T>(index.algo, sp_json);

  const auto algo_property = parse_algo_property_override(algo->get_preference(), sp_json);

  if (search_param->needs_dataset()) {
    algo->set_search_dataset(dataset->base_set(algo_property.dataset_memory_type),
                             dataset->base_set_size());
  }
  algo->set_search_param(*search_param,
                         dataset->filter_bitset(algo_property.dataset_memory_type));

  algo->search_ex(reinterpret_cast<const T*>(queries),
                  static_cast<int>(query_count),
                  static_cast<int>(k),
                  neighbors,
                  distances,
                  search_ids);

  // Keep algo alive across build/insert/search pipeline and avoid teardown at this scope.
  cache_algo<T>(index.file, index.algo, get_dtype_string<T>(), dataset->dim(), algo);

  if (out_queries != nullptr) { *out_queries = query_count; }
  if (out_k != nullptr) { *out_k = static_cast<std::size_t>(k); }
}

template <typename T>
inline void   insert_from_config(const std::string& conf_path,
                               const std::string& index_name,
                               std::size_t index_pos,
                               const std::string& insert_vectors_file,
                               const std::string& data_prefix,
                               const std::string& index_prefix,
                               const std::string& insert_params_json_str = "{}")
{
  std::lock_guard<std::recursive_mutex> api_lock(get_bench_api_mutex());

  auto data_prefix_norm  = detail::normalize_prefix(data_prefix, "data");
  auto index_prefix_norm = detail::normalize_prefix(index_prefix, "index");
  std::ifstream conf_stream(conf_path);
  ensure_stream_open(conf_stream, conf_path);
  auto& conf = configuration::initialize(conf_stream, data_prefix_norm, index_prefix_norm);
  const auto& index = detail::pick_index(conf, index_name, index_pos);

  set_thread_context();

  auto dataset = make_dataset<T>(conf.get_dataset_conf(), false);

  // Try to get cached algo first (from recent build/insert/search in same process).
  std::unique_ptr<algo<T>> algo = get_cached_algo<T>(
    index.file, index.algo, get_dtype_string<T>(), dataset->dim());

  // Read insert vectors from binary file
  std::cout << "[INSERT] Loading vectors from: " << insert_vectors_file << std::endl;
  blob<T> insert_blob{insert_vectors_file};
  const T* insert_vectors = insert_blob.data();
  std::size_t num_insert_vectors = insert_blob.n_rows();
  std::size_t insert_dim = insert_blob.n_cols();

  // Validate insert vectors dimension matches dataset dimension
  if (insert_dim != static_cast<std::size_t>(dataset->dim())) {
    throw std::runtime_error(
      "Insert vectors dimension (" + std::to_string(insert_dim) + ") does not match dataset " +
      "dimension (" + std::to_string(dataset->dim()) + ")");
  }

  bool has_index_on_disk = file_exists(index.file);

  std::cout << "[INSERT] Insert vectors loaded: " << num_insert_vectors << " vectors, "
            << insert_dim << " dimensions" << std::endl;

  // Parse insert parameters from JSON string
  nlohmann::json insert_params_json;
  try {
    insert_params_json = nlohmann::json::parse(insert_params_json_str);
  } catch (const std::exception& e) {
    std::cerr << "[INSERT] Warning: Failed to parse JSON parameters: " << e.what() << std::endl;
    std::cerr << "[INSERT] Using default insert parameters" << std::endl;
    insert_params_json = nlohmann::json::object();
  }

  if (!algo && !has_index_on_disk) {
    std::cout << "[INSERT] Index file missing; building base index from insert vectors: "
              << index.file << std::endl;
    algo = create_algo<T>(index.algo, dataset->distance(), dataset->dim(), index.build_param);
    make_sure_parent_dir_exists(index.file);
    algo->set_build_output_file(index.file);
    algo->build(insert_vectors, num_insert_vectors);
    if (insert_params_json.value("persist", false)) {
      algo->save(index.file);
      std::cout << "[INSERT] Persisted built index to: " << index.file << std::endl;
    }
    std::cout << "[INSERT] Base index build from insert vectors completed" << std::endl;
    cache_algo<T>(index.file, index.algo, get_dtype_string<T>(), dataset->dim(), algo);
    cudaDeviceSynchronize();
    return;
  }

  if (!algo) {
    std::cout << "[INSERT] Loading index from: " << index.file << std::endl;
    algo = create_algo<T>(index.algo, dataset->distance(), dataset->dim(), index.build_param);
    // Load the built index FIRST (before set_search_dataset)
    algo->load(index.file);
    std::cout << "[INSERT] Index loaded successfully" << std::endl;
  } else {
    std::cout << "[INSERT] Using in-memory algo from cache" << std::endl;
  }

  const auto algo_property = parse_algo_property_override(algo->get_preference(), index.build_param);
  const T* base_set        = dataset->base_set(algo_property.dataset_memory_type);
  std::size_t index_size   = dataset->base_set_size();

  // Set the dataset for the algorithm (after load, so index_ is not null)
  algo->set_search_dataset(base_set, index_size);

  // Perform insert operation
  std::cout << "[INSERT] Inserting " << num_insert_vectors << " vectors..." << std::endl;
  algo->set_insert_param_from_json(insert_params_json);
  algo->insert(insert_vectors, num_insert_vectors);
  std::cout << "[INSERT] Insert completed successfully" << std::endl;

  // Keep algo alive across build/insert/search pipeline and avoid teardown at this scope.
  cache_algo<T>(index.file, index.algo, get_dtype_string<T>(), dataset->dim(), algo);
  cudaDeviceSynchronize();
}

template <typename T>
inline void insert_from_config_array(const std::string& conf_path,
                                     const std::string& index_name,
                                     std::size_t index_pos,
                                     const void* insert_vectors,
                                     const int64_t* insert_ids,
                                     std::size_t num_insert_vectors,
                                     std::size_t insert_dim,
                                     const std::string& data_prefix,
                                     const std::string& index_prefix,
                                     const std::string& insert_params_json_str = "{}")
{
  std::lock_guard<std::recursive_mutex> api_lock(get_bench_api_mutex());

  auto data_prefix_norm  = detail::normalize_prefix(data_prefix, "data");
  auto index_prefix_norm = detail::normalize_prefix(index_prefix, "index");
  std::ifstream conf_stream(conf_path);
  std::unique_ptr<typename algo<T>::search_param> search_param;
  ensure_stream_open(conf_stream, conf_path);
  auto& conf = configuration::initialize(conf_stream, data_prefix_norm, index_prefix_norm);
  const auto& index = detail::pick_index(conf, index_name, index_pos);

  set_thread_context();

  if (insert_vectors == nullptr) {
    throw std::runtime_error("insert_vectors pointer must not be null");
  }

  auto dataset = make_dataset<T>(conf.get_dataset_conf(), false);

  // Try to get cached algo first (from recent build)
  std::unique_ptr<algo<T>> algo = get_cached_algo<T>(
    index.file, index.algo, get_dtype_string<T>(), dataset->dim());

  // Validate insert vectors dimension matches dataset dimension
  if (insert_dim != static_cast<std::size_t>(dataset->dim())) {
    throw std::runtime_error(
      "Insert vectors dimension (" + std::to_string(insert_dim) + ") does not match dataset " +
      "dimension (" + std::to_string(dataset->dim()) + ")");
  }

  bool has_index_on_disk = file_exists(index.file);

  std::cout << "[INSERT] Insert vectors provided from memory: " << num_insert_vectors
            << " vectors, " << insert_dim << " dimensions" << std::endl;

  // Parse insert parameters from JSON string
  nlohmann::json insert_params_json;
  try {
    insert_params_json = nlohmann::json::parse(insert_params_json_str);
  } catch (const std::exception& e) {
    std::cerr << "[INSERT] Warning: Failed to parse JSON parameters: " << e.what() << std::endl;
    std::cerr << "[INSERT] Using default insert parameters" << std::endl;
    insert_params_json = nlohmann::json::object();
  }

  if (!algo && !has_index_on_disk) {
    std::cout << "[INSERT] Index file missing; building base index from insert vectors: "
              << index.file << std::endl;
    algo = create_algo<T>(index.algo, dataset->distance(), dataset->dim(), index.build_param);
    make_sure_parent_dir_exists(index.file);
    algo->set_build_output_file(index.file);
    algo->build(reinterpret_cast<const T*>(insert_vectors), num_insert_vectors);
    if (insert_params_json.value("persist", false)) {
      algo->save(index.file);
      std::cout << "[INSERT] Persisted built index to: " << index.file << std::endl;
    }
    std::cout << "[INSERT] Base index build from insert vectors completed" << std::endl;
    cache_algo<T>(index.file, index.algo, get_dtype_string<T>(), dataset->dim(), algo);
    cudaDeviceSynchronize();
    return;
  }

  if (!algo) {
    // No cached algo, need to create and load from disk
    std::cout << "[INSERT] Loading index from: " << index.file << std::endl;
    algo = create_algo<T>(index.algo, dataset->distance(), dataset->dim(), index.build_param);
    algo->load(index.file);
    std::cout << "[INSERT] Index loaded successfully" << std::endl;
  } else {
    std::cout << "[INSERT] Using in-memory algo from build" << std::endl;
  }

  auto algo_property = parse_algo_property_override(algo->get_preference(), index.build_param);
  const T* base_set        = dataset->base_set(algo_property.dataset_memory_type);
  std::size_t index_size   = dataset->base_set_size();

  // Set the dataset for the algorithm (after load, so index_ is not null)
  algo->set_search_dataset(base_set, index_size);

  // Perform insert operation
  std::cout << "[INSERT] Inserting " << num_insert_vectors << " vectors..." << std::endl;
  algo->set_insert_param_from_json(insert_params_json);

  std::size_t search_param_ix = 0; // Hard-coded for now
  ensure_search_params(index, search_param_ix);

  const auto& sp_json = index.search_params[search_param_ix];

  search_param = create_search_param<T>(index.algo, sp_json);

  algo_property = parse_algo_property_override(algo->get_preference(), sp_json);

  // Since, we do a search in insert, lets set search params
  if (search_param->needs_dataset()) {
      algo->set_search_dataset(dataset->base_set(algo_property.dataset_memory_type),
                               dataset->base_set_size());
  }
  algo->set_search_param(*search_param,
                         dataset->filter_bitset(algo_property.dataset_memory_type));
  std::cout << "[INSERT] set_search_param completed" << std::endl;
  algo->insert(reinterpret_cast<const T*>(insert_vectors), num_insert_vectors, insert_ids);
  std::cout << "[INSERT] Insert completed successfully" << std::endl;

  // Cache the algo for subsequent search operations
  cache_algo<T>(index.file, index.algo, get_dtype_string<T>(), dataset->dim(), algo);
  std::cout << "[ALGO_CACHE] Cached algo after insert for: " << index.file << std::endl;
}

inline void build_from_config_float(const std::string& conf_path,
                                    const std::string& index_name,
                                    std::size_t index_pos,
                                    const std::string& data_prefix,
                                    const std::string& index_prefix,
                                    bool force_overwrite)
{
  build_from_config<float>(conf_path, index_name, index_pos, data_prefix, index_prefix, force_overwrite);
}

inline void build_from_config_half(const std::string& conf_path,
                                   const std::string& index_name,
                                   std::size_t index_pos,
                                   const std::string& data_prefix,
                                   const std::string& index_prefix,
                                   bool force_overwrite)
{
  build_from_config<half>(conf_path, index_name, index_pos, data_prefix, index_prefix, force_overwrite);
}

inline void build_from_config_uint8(const std::string& conf_path,
                                    const std::string& index_name,
                                    std::size_t index_pos,
                                    const std::string& data_prefix,
                                    const std::string& index_prefix,
                                    bool force_overwrite)
{
  build_from_config<std::uint8_t>(conf_path, index_name, index_pos, data_prefix, index_prefix, force_overwrite);
}

inline void build_from_config_int8(const std::string& conf_path,
                                   const std::string& index_name,
                                   std::size_t index_pos,
                                   const std::string& data_prefix,
                                   const std::string& index_prefix,
                                   bool force_overwrite)
{
  build_from_config<std::int8_t>(conf_path, index_name, index_pos, data_prefix, index_prefix, force_overwrite);
}

inline void insert_from_config_float(const std::string& conf_path,
                                     const std::string& index_name,
                                     std::size_t index_pos,
                                     const std::string& insert_vectors_file,
                                     const std::string& data_prefix,
                                     const std::string& index_prefix,
                                     const std::string& insert_params_json = "{}")
{
  insert_from_config<float>(conf_path, index_name, index_pos, insert_vectors_file, data_prefix, index_prefix, insert_params_json);
}

inline void insert_from_config_half(const std::string& conf_path,
                                    const std::string& index_name,
                                    std::size_t index_pos,
                                    const std::string& insert_vectors_file,
                                    const std::string& data_prefix,
                                    const std::string& index_prefix,
                                    const std::string& insert_params_json = "{}")
{
  insert_from_config<half>(conf_path, index_name, index_pos, insert_vectors_file, data_prefix, index_prefix, insert_params_json);
}

inline void insert_from_config_uint8(const std::string& conf_path,
                                     const std::string& index_name,
                                     std::size_t index_pos,
                                     const std::string& insert_vectors_file,
                                     const std::string& data_prefix,
                                     const std::string& index_prefix,
                                     const std::string& insert_params_json = "{}")
{
  insert_from_config<std::uint8_t>(conf_path, index_name, index_pos, insert_vectors_file, data_prefix, index_prefix, insert_params_json);
}

inline void insert_from_config_int8(const std::string& conf_path,
                                    const std::string& index_name,
                                    std::size_t index_pos,
                                    const std::string& insert_vectors_file,
                                    const std::string& data_prefix,
                                    const std::string& index_prefix,
                                    const std::string& insert_params_json = "{}")
{
  insert_from_config<std::int8_t>(conf_path, index_name, index_pos, insert_vectors_file, data_prefix, index_prefix, insert_params_json);
}

inline void insert_from_config_array_float(const std::string& conf_path,
                                           const std::string& index_name,
                                           std::size_t index_pos,
                                           const void* insert_vectors,
                                           const int64_t* insert_ids,
                                           std::size_t num_insert_vectors,
                                           std::size_t insert_dim,
                                           const std::string& data_prefix,
                                           const std::string& index_prefix,
                                           const std::string& insert_params_json = "{}")
{
  insert_from_config_array<float>(conf_path,
                                  index_name,
                                  index_pos,
                                  insert_vectors,
                                  insert_ids,
                                  num_insert_vectors,
                                  insert_dim,
                                  data_prefix,
                                  index_prefix,
                                  insert_params_json);
}

inline void insert_from_config_array_half(const std::string& conf_path,
                                          const std::string& index_name,
                                          std::size_t index_pos,
                                          const void* insert_vectors,
                                          const int64_t* insert_ids,
                                          std::size_t num_insert_vectors,
                                          std::size_t insert_dim,
                                          const std::string& data_prefix,
                                          const std::string& index_prefix,
                                          const std::string& insert_params_json = "{}")
{
  insert_from_config_array<half>(conf_path,
                                 index_name,
                                 index_pos,
                                 insert_vectors,
                                 insert_ids,
                                 num_insert_vectors,
                                 insert_dim,
                                 data_prefix,
                                 index_prefix,
                                 insert_params_json);
}

inline void insert_from_config_array_uint8(const std::string& conf_path,
                                           const std::string& index_name,
                                           std::size_t index_pos,
                                           const void* insert_vectors,
                                           const int64_t* insert_ids,
                                           std::size_t num_insert_vectors,
                                           std::size_t insert_dim,
                                           const std::string& data_prefix,
                                           const std::string& index_prefix,
                                           const std::string& insert_params_json = "{}")
{
  insert_from_config_array<std::uint8_t>(conf_path,
                                         index_name,
                                         index_pos,
                                         insert_vectors,
                                         insert_ids,
                                         num_insert_vectors,
                                         insert_dim,
                                         data_prefix,
                                         index_prefix,
                                         insert_params_json);
}

inline void insert_from_config_array_int8(const std::string& conf_path,
                                          const std::string& index_name,
                                          std::size_t index_pos,
                                          const void* insert_vectors,
                                          const int64_t* insert_ids,
                                          std::size_t num_insert_vectors,
                                          std::size_t insert_dim,
                                          const std::string& data_prefix,
                                          const std::string& index_prefix,
                                          const std::string& insert_params_json = "{}")
{
  insert_from_config_array<std::int8_t>(conf_path,
                                        index_name,
                                        index_pos,
                                        insert_vectors,
                                        insert_ids,
                                        num_insert_vectors,
                                        insert_dim,
                                        data_prefix,
                                        index_prefix,
                                        insert_params_json);
}

inline void get_search_shape_from_config_float(const std::string& conf_path,
                                               const std::string& index_name,
                                               std::size_t index_pos,
                                               std::size_t search_param_ix,
                                               std::size_t query_count,
                                               const std::string& data_prefix,
                                               const std::string& index_prefix,
                                               std::size_t* out_queries,
                                               std::size_t* out_k)
{
  get_search_shape_from_config<float>(conf_path,
                                      index_name,
                                      index_pos,
                                      search_param_ix,
                                      query_count,
                                      data_prefix,
                                      index_prefix,
                                      out_queries,
                                      out_k);
}

inline void get_search_shape_from_config_half(const std::string& conf_path,
                                              const std::string& index_name,
                                              std::size_t index_pos,
                                              std::size_t search_param_ix,
                                              std::size_t query_count,
                                              const std::string& data_prefix,
                                              const std::string& index_prefix,
                                              std::size_t* out_queries,
                                              std::size_t* out_k)
{
  get_search_shape_from_config<half>(conf_path,
                                     index_name,
                                     index_pos,
                                     search_param_ix,
                                     query_count,
                                     data_prefix,
                                     index_prefix,
                                     out_queries,
                                     out_k);
}

inline void get_search_shape_from_config_uint8(const std::string& conf_path,
                                               const std::string& index_name,
                                               std::size_t index_pos,
                                               std::size_t search_param_ix,
                                               std::size_t query_count,
                                               const std::string& data_prefix,
                                               const std::string& index_prefix,
                                               std::size_t* out_queries,
                                               std::size_t* out_k)
{
  get_search_shape_from_config<std::uint8_t>(conf_path,
                                             index_name,
                                             index_pos,
                                             search_param_ix,
                                             query_count,
                                             data_prefix,
                                             index_prefix,
                                             out_queries,
                                             out_k);
}

inline void get_search_shape_from_config_int8(const std::string& conf_path,
                                              const std::string& index_name,
                                              std::size_t index_pos,
                                              std::size_t search_param_ix,
                                              std::size_t query_count,
                                              const std::string& data_prefix,
                                              const std::string& index_prefix,
                                              std::size_t* out_queries,
                                              std::size_t* out_k)
{
  get_search_shape_from_config<std::int8_t>(conf_path,
                                            index_name,
                                            index_pos,
                                            search_param_ix,
                                            query_count,
                                            data_prefix,
                                            index_prefix,
                                            out_queries,
                                            out_k);
}

inline void search_from_config_float(const std::string& conf_path,
                                     const std::string& index_name,
                                     std::size_t index_pos,
                                     std::size_t search_param_ix,
                                     std::size_t query_count,
                                     const std::string& data_prefix,
                                     const std::string& index_prefix,
                                     algo_base::index_type* neighbors,
                                     float* distances,
                                     int64_t* search_ids,
                                     std::size_t* out_queries,
                                     std::size_t* out_k)
{
  search_from_config<float>(conf_path,
                            index_name,
                            index_pos,
                            search_param_ix,
                            query_count,
                            data_prefix,
                            index_prefix,
                            neighbors,
                            distances,
                            search_ids,
                            out_queries,
                            out_k);
}

inline void search_from_config_array_float(const std::string& conf_path,
                                           const std::string& index_name,
                                           std::size_t index_pos,
                                           std::size_t search_param_ix,
                                           const void* queries,
                                           std::size_t query_count,
                                           std::size_t query_dim,
                                           const std::string& data_prefix,
                                           const std::string& index_prefix,
                                           algo_base::index_type* neighbors,
                                           float* distances,
                                           int64_t* search_ids,
                                           std::size_t* out_queries,
                                           std::size_t* out_k)
{
  search_from_config_array<float>(conf_path,
                                  index_name,
                                  index_pos,
                                  search_param_ix,
                                  queries,
                                  query_count,
                                  query_dim,
                                  data_prefix,
                                  index_prefix,
                                  neighbors,
                                  distances,
                                  search_ids,
                                  out_queries,
                                  out_k);
}

inline void search_from_config_half(const std::string& conf_path,
                                    const std::string& index_name,
                                    std::size_t index_pos,
                                    std::size_t search_param_ix,
                                    std::size_t query_count,
                                    const std::string& data_prefix,
                                    const std::string& index_prefix,
                                    algo_base::index_type* neighbors,
                                    float* distances,
                                    int64_t* search_ids,
                                    std::size_t* out_queries,
                                    std::size_t* out_k)
{
  search_from_config<half>(conf_path,
                           index_name,
                           index_pos,
                           search_param_ix,
                           query_count,
                           data_prefix,
                           index_prefix,
                           neighbors,
                           distances,
                           search_ids,
                           out_queries,
                           out_k);
}

inline void search_from_config_array_half(const std::string& conf_path,
                                          const std::string& index_name,
                                          std::size_t index_pos,
                                          std::size_t search_param_ix,
                                          const void* queries,
                                          std::size_t query_count,
                                          std::size_t query_dim,
                                          const std::string& data_prefix,
                                          const std::string& index_prefix,
                                          algo_base::index_type* neighbors,
                                          float* distances,
                                          int64_t* search_ids,
                                          std::size_t* out_queries,
                                          std::size_t* out_k)
{
  search_from_config_array<half>(conf_path,
                                 index_name,
                                 index_pos,
                                 search_param_ix,
                                 queries,
                                 query_count,
                                 query_dim,
                                 data_prefix,
                                 index_prefix,
                                 neighbors,
                                 distances,
                                 search_ids,
                                 out_queries,
                                 out_k);
}

inline void search_from_config_uint8(const std::string& conf_path,
                                     const std::string& index_name,
                                     std::size_t index_pos,
                                     std::size_t search_param_ix,
                                     std::size_t query_count,
                                     const std::string& data_prefix,
                                     const std::string& index_prefix,
                                     algo_base::index_type* neighbors,
                                     float* distances,
                                     int64_t* search_ids,
                                     std::size_t* out_queries,
                                     std::size_t* out_k)
{
  search_from_config<std::uint8_t>(conf_path,
                                   index_name,
                                   index_pos,
                                   search_param_ix,
                                   query_count,
                                   data_prefix,
                                   index_prefix,
                                   neighbors,
                                   distances,
                                   search_ids,
                                   out_queries,
                                   out_k);
}

inline void search_from_config_array_uint8(const std::string& conf_path,
                                           const std::string& index_name,
                                           std::size_t index_pos,
                                           std::size_t search_param_ix,
                                           const void* queries,
                                           std::size_t query_count,
                                           std::size_t query_dim,
                                           const std::string& data_prefix,
                                           const std::string& index_prefix,
                                           algo_base::index_type* neighbors,
                                           float* distances,
                                           int64_t* search_ids,
                                           std::size_t* out_queries,
                                           std::size_t* out_k)
{
  search_from_config_array<std::uint8_t>(conf_path,
                                         index_name,
                                         index_pos,
                                         search_param_ix,
                                         queries,
                                         query_count,
                                         query_dim,
                                         data_prefix,
                                         index_prefix,
                                         neighbors,
                                         distances,
                                         search_ids,
                                         out_queries,
                                         out_k);
}

inline void search_from_config_int8(const std::string& conf_path,
                                    const std::string& index_name,
                                    std::size_t index_pos,
                                    std::size_t search_param_ix,
                                    std::size_t query_count,
                                    const std::string& data_prefix,
                                    const std::string& index_prefix,
                                    algo_base::index_type* neighbors,
                                    float* distances,
                                    int64_t* search_ids,
                                    std::size_t* out_queries,
                                    std::size_t* out_k)
{
  search_from_config<std::int8_t>(conf_path,
                                  index_name,
                                  index_pos,
                                  search_param_ix,
                                  query_count,
                                  data_prefix,
                                  index_prefix,
                                  neighbors,
                                  distances,
                                  search_ids,
                                  out_queries,
                                  out_k);
}

inline void search_from_config_array_int8(const std::string& conf_path,
                                          const std::string& index_name,
                                          std::size_t index_pos,
                                          std::size_t search_param_ix,
                                          const void* queries,
                                          std::size_t query_count,
                                          std::size_t query_dim,
                                          const std::string& data_prefix,
                                          const std::string& index_prefix,
                                          algo_base::index_type* neighbors,
                                          float* distances,
                                          int64_t* search_ids,
                                          std::size_t* out_queries,
                                          std::size_t* out_k)
{
  search_from_config_array<std::int8_t>(conf_path,
                                        index_name,
                                        index_pos,
                                        search_param_ix,
                                        queries,
                                        query_count,
                                        query_dim,
                                        data_prefix,
                                        index_prefix,
                                        neighbors,
                                        distances,
                                        search_ids,
                                        out_queries,
                                        out_k);
}

}  // namespace cuvs::bench
