/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * FreshBANG-specific helper layer: dlsym dispatch plus minimal bench utility wrappers
 * needed by freshbang_wrapper, without depending on common/bench_api.hpp.
 */
#pragma once

#include "../common/ann_types.hpp"
#include "../common/conf.hpp"
#include "../common/dataset.hpp"
#include "../common/util.hpp"

#include <dlfcn.h>

#include <fstream>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <unordered_map>

#include <nlohmann/json.hpp>

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

// ---------------------------------------------------------------------------
// C symbol names for create_algo (used for dlsym dispatch)
// ---------------------------------------------------------------------------

template <typename T>
inline std::string get_create_algo_rows_c_symbol_name();

template <>
inline std::string get_create_algo_rows_c_symbol_name<float>()
{
  return "cuvs_bench_create_algo_with_rows_float";
}

template <>
inline std::string get_create_algo_rows_c_symbol_name<std::uint8_t>()
{
  return "cuvs_bench_create_algo_with_rows_uint8";
}

template <>
inline std::string get_create_algo_rows_c_symbol_name<std::int8_t>()
{
  return "cuvs_bench_create_algo_with_rows_int8";
}

// Mangled fallback symbol names (for backwards compat with older .so files)
template <typename T>
inline std::string get_create_algo_symbol_name();

template <>
inline std::string get_create_algo_symbol_name<float>()
{
  return "_ZN4cuvs5bench11create_algoIfEESt10unique_ptrINS0_4algoIT_EESt14default_deleteIS5_"
         "EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESG_iRKN8nlohmann16json_abi_v3_"
         "12_010basic_jsonISt3mapSt6vectorSE_blmdSaNSI_14adl_serializerESL_IhSaIhEEvEE";
}

template <>
inline std::string get_create_algo_symbol_name<std::uint8_t>()
{
  return "_ZN4cuvs5bench11create_algoIhEESt10unique_ptrINS0_4algoIT_EESt14default_deleteIS5_"
         "EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESG_iRKN8nlohmann16json_abi_v3_"
         "12_010basic_jsonISt3mapSt6vectorSE_blmdSaNSI_14adl_serializerESL_IhSaIhEEvEE";
}

template <>
inline std::string get_create_algo_symbol_name<std::int8_t>()
{
  return "_ZN4cuvs5bench11create_algoIaEESt10unique_ptrINS0_4algoIT_EESt14default_deleteIS5_"
         "EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEESG_iRKN8nlohmann16json_abi_v3_"
         "12_010basic_jsonISt3mapSt6vectorSE_blmdSaNSI_14adl_serializerESL_IhSaIhEEvEE";
}

// ---------------------------------------------------------------------------
// create_algo<T> — dlsym dispatch into the algo plugin .so
// ---------------------------------------------------------------------------
template <typename T>
inline auto invoke_create_algo(const std::string& algo,
                               const std::string& distance,
                               int rows,
                               int dim,
                               const nlohmann::json& conf) -> std::unique_ptr<cuvs::bench::algo<T>>
{
#ifdef _WIN32
  (void)algo;
  (void)distance;
  (void)rows;
  (void)dim;
  (void)conf;
  throw std::runtime_error("Dynamic loading is not supported on Windows.");
#else
  auto rows_c_fname       = get_create_algo_rows_c_symbol_name<T>();
  auto legacy_mangled_name = get_create_algo_symbol_name<T>();
  auto handle             = load_lib_raft_compat(algo);
  auto fun_addr           = dlsym(handle, rows_c_fname.c_str());
  std::cerr << "[DEBUG] dlsym attempt 1 - rows C symbol: " << rows_c_fname.substr(0, 50)
            << "... : " << (fun_addr != nullptr ? "FOUND" : "NOT FOUND") << std::endl;
  if (fun_addr != nullptr) {
    using create_algo_with_rows_fn_t =
      std::unique_ptr<cuvs::bench::algo<T>> (*)(const std::string&,
                                                const std::string&,
                                                int,
                                                int,
                                                const nlohmann::json&);
    auto fun = reinterpret_cast<create_algo_with_rows_fn_t>(fun_addr);
    return fun(algo, distance, rows, dim, conf);
  }

  fun_addr = dlsym(handle, legacy_mangled_name.c_str());
  std::cerr << "[DEBUG] dlsym attempt 2 - legacy mangled symbol: "
            << legacy_mangled_name.substr(0, 50) << "... : "
            << (fun_addr != nullptr ? "FOUND" : "NOT FOUND") << std::endl;
  if (fun_addr == nullptr) {
    throw std::runtime_error("Couldn't load the create_algo function (" + algo + "): " +
                             std::string(dlerror()));
  }
  using legacy_create_algo_fn_t =
    std::unique_ptr<cuvs::bench::algo<T>> (*)(const std::string&,
                                              const std::string&,
                                              int,
                                              const nlohmann::json&);
  auto fun = reinterpret_cast<legacy_create_algo_fn_t>(fun_addr);
  std::cerr << "[DEBUG] using legacy create_algo ABI without rows support for algo='" << algo
            << "'" << std::endl;
  return fun(algo, distance, dim, conf);
#endif
}

template <typename T>
inline std::string get_create_search_param_c_symbol_name();

template <>
inline std::string get_create_search_param_c_symbol_name<float>()
{
  return "cuvs_bench_create_search_param_float";
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

template <typename T>
inline std::string get_create_search_param_symbol_name();

template <>
inline std::string get_create_search_param_symbol_name<float>()
{
  return "_ZN4cuvs5bench19create_search_paramIfEESt10unique_ptrINS0_4algoIT_E12search_paramE"
         "St14default_deleteIS6_EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKN8nlohmann"
         "16json_abi_v3_12_010basic_jsonISt3mapSt6vectorSF_blmdSaNSJ_14adl_serializerESM_IhSaIhEEvEE";
}

template <>
inline std::string get_create_search_param_symbol_name<std::uint8_t>()
{
  return "_ZN4cuvs5bench19create_search_paramIhEESt10unique_ptrINS0_4algoIT_E12search_paramE"
         "St14default_deleteIS6_EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKN8nlohmann"
         "16json_abi_v3_12_010basic_jsonISt3mapSt6vectorSF_blmdSaNSJ_14adl_serializerESM_IhSaIhEEvEE";
}

template <>
inline std::string get_create_search_param_symbol_name<std::int8_t>()
{
  return "_ZN4cuvs5bench19create_search_paramIaEESt10unique_ptrINS0_4algoIT_E12search_paramE"
         "St14default_deleteIS6_EERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEERKN8nlohmann"
         "16json_abi_v3_12_010basic_jsonISt3mapSt6vectorSF_blmdSaNSJ_14adl_serializerESM_IhSaIhEEvEE";
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
  auto handle       = load_lib_raft_compat(algo);
  auto fun_addr     = dlsym(handle, c_fname.c_str());
  if (fun_addr == nullptr) {
    fun_addr = dlsym(handle, mangled_name.c_str());
  }
  if (fun_addr == nullptr) {
    throw std::runtime_error("Couldn't load the create_search_param function (" + algo + "): " +
                             std::string(dlerror()));
  }
  using create_search_param_fn_t =
    std::unique_ptr<typename cuvs::bench::algo<T>::search_param> (*)(const std::string&,
                                                                      const nlohmann::json&);
  auto fun = reinterpret_cast<create_search_param_fn_t>(fun_addr);
  return fun(algo, conf);
#endif
}

}  // namespace detail

inline auto get_bench_api_mutex() -> std::recursive_mutex&
{
  static std::recursive_mutex g_freshbang_api_mutex;
  return g_freshbang_api_mutex;
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

template <typename T>
inline auto make_dataset(const configuration::dataset_conf& dataset_conf,
                         bool use_filtering,
                         bool use_insert_set = false) -> std::shared_ptr<const dataset<T>>
{
  auto filtering = use_filtering ? dataset_conf.filtering_rate : std::nullopt;
  auto insert_file = use_insert_set ? dataset_conf.insert_file : std::nullopt;
  return std::make_shared<bench::dataset<T>>(dataset_conf.name,
                                             dataset_conf.base_file,
                                             dataset_conf.subset_first_row,
                                             dataset_conf.subset_size,
                                             dataset_conf.query_file,
                                             dataset_conf.distance,
                                             dataset_conf.groundtruth_neighbors_file,
                                             insert_file,
                                             filtering);
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

}  // namespace cuvs::bench

