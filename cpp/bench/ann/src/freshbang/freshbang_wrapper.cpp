#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include <cuda_runtime.h>

#include "freshbang_helpers.hpp"
#include "freshbang.hpp"

namespace cuvs::bench {
namespace detail {

struct algo_cache_entry {
  void* algo_ptr{nullptr};
  std::string algo_name;
  std::string dtype;
  int dim{0};
  std::chrono::steady_clock::time_point timestamp{};
  std::function<void(void*)> deleter;
};

auto trim_whitespace(std::string value) -> std::string
{
  auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) { return ""; }

  auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

inline auto file_exists(const std::string& filename) -> bool
{
  struct stat statbuf;
  if (stat(filename.c_str(), &statbuf) != 0) { return false; }
  return S_ISREG(statbuf.st_mode);
}


struct FreshBANGConfig {
  std::string conf_path;
  std::string data_prefix{"data/"};
  std::string index_prefix{"index/"};
};

auto read_freshbang_config(const std::string& config_path) -> FreshBANGConfig
{
  std::ifstream config_stream(config_path);
  if (!config_stream) {
    throw std::runtime_error("Failed to open config file: " + config_path);
  }

  FreshBANGConfig config;
  std::string line;
  while (std::getline(config_stream, line)) {
    auto trimmed_line = trim_whitespace(line);
    if (trimmed_line.empty() || trimmed_line[0] == '#') { continue; }

    auto separator_pos = trimmed_line.find('=');
    if (separator_pos == std::string::npos) { continue; }

    auto key   = trim_whitespace(trimmed_line.substr(0, separator_pos));
    auto value = trim_whitespace(trimmed_line.substr(separator_pos + 1));

    if (key == "json_conf_path") {
      config.conf_path = value;
    } else if (key == "data_prefix") {
      config.data_prefix = value;
    } else if (key == "index_prefix") {
      config.index_prefix = value;
    }
  }

  return config;
}

class AlgoCacheSingleton {
 public:
  static auto instance() -> AlgoCacheSingleton&
  {
    static AlgoCacheSingleton singleton;
    return singleton;
  }

  auto get(const std::string& index_file,
           const std::string& algo_name,
           const std::string& dtype,
           int dim,
           std::chrono::minutes max_age) -> std::optional<algo_cache_entry>
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(index_file);
    if (it == cache_.end()) { return std::nullopt; }

    auto& entry = it->second;
    if (entry.algo_name != algo_name || entry.dtype != dtype || entry.dim != dim) {
      return std::nullopt;
    }

    auto elapsed = std::chrono::steady_clock::now() - entry.timestamp;
    if (std::chrono::duration_cast<std::chrono::minutes>(elapsed) >= max_age) {
      if (entry.algo_ptr != nullptr && entry.deleter) {
        entry.deleter(entry.algo_ptr);
        entry.algo_ptr = nullptr;
      }
      cache_.erase(it);
      return std::nullopt;
    }

    return entry;
  }

  void put(const std::string& index_file, algo_cache_entry&& entry)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = cache_.find(index_file);
    if (it != cache_.end()) {
      auto& existing = it->second;
      if (existing.algo_ptr != nullptr && existing.deleter) {
        existing.deleter(existing.algo_ptr);
        existing.algo_ptr = nullptr;
      }
    }
    cache_[index_file] = std::move(entry);
  }

  void clear()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& pair : cache_) {
      auto& entry = pair.second;
      if (entry.algo_ptr != nullptr && entry.deleter) {
        entry.deleter(entry.algo_ptr);
        entry.algo_ptr = nullptr;
      }
    }
    cache_.clear();
  }

 private:
  std::unordered_map<std::string, algo_cache_entry> cache_;
  std::mutex mutex_;
};

auto get_cached_algo_entry(const std::string& index_file,
                           const std::string& algo_name,
                           const std::string& dtype,
                           int dim,
                           std::chrono::minutes max_age) -> std::optional<algo_cache_entry>
{
  return AlgoCacheSingleton::instance().get(index_file, algo_name, dtype, dim, max_age);
}

void put_cached_algo_entry(const std::string& index_file, algo_cache_entry&& entry)
{
  AlgoCacheSingleton::instance().put(index_file, std::move(entry));
}

void clear_cached_algo_entries()
{
  AlgoCacheSingleton::instance().clear();
}

template <typename T>
class FreshBANGInner {
 public:
  FreshBANGInner(std::string conf_path, std::string data_prefix, std::string index_prefix)
    : conf_path_(std::move(conf_path)),
      data_prefix_(std::move(data_prefix)),
      index_prefix_(std::move(index_prefix))
  {
  }

  void set_index_info(std::string index_file, std::string algo_name, int dim)
  {
    index_file_ = std::move(index_file);
    algo_name_  = std::move(algo_name);
    dim_        = dim;
  }

  const std::string& index_file() const { return index_file_; }
  const std::string& algo_name() const { return algo_name_; }
  int dim() const { return dim_; }

  void set_algo_property(cuvs::bench::algo_property prop) { algo_property_ = prop; }
  const cuvs::bench::algo_property& algo_property() const { return algo_property_; }

  void set_search_dataset(std::shared_ptr<const cuvs::bench::dataset<T>> ds)
  {
    search_dataset_ = std::move(ds);
  }

 private:
  std::string conf_path_;
  std::string data_prefix_;
  std::string index_prefix_;

  std::string index_file_;
  std::string algo_name_;
  int dim_{0};
  cuvs::bench::algo_property algo_property_{};
  std::shared_ptr<const cuvs::bench::dataset<T>> search_dataset_;
};

}  // namespace detail
}  // namespace cuvs::bench

// ---------------------------------------------------------------------------
// FreshBANG<T> method implementations
// FreshBANG is declared at global scope (no namespace) in freshbang.hpp, so
// all method bodies must also live at global scope. Internal helpers are
// accessed via the fully-qualified cuvs::bench::detail:: prefix.
// ---------------------------------------------------------------------------

template <typename T>
FreshBANG<T>::FreshBANG() : m_pImpl(nullptr)
{
  std::cout << "[FreshBANG] Constructor called" << std::endl;
}

template <typename T>
FreshBANG<T>::~FreshBANG()
{
  delete static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);
  m_pImpl = nullptr;
  std::cout << "[FreshBANG] Destructor called" << std::endl;
}

template <typename T>
bool FreshBANG<T>::SetDatasetParams(BuildParams params)
{
  std::cout << "[FreshBANG::SetDatasetParams] dataset_dim=" << params.dataset_dim
            << " distance_measure=" << params.distance_measure << std::endl;
  user_build_params_ = std::move(params);
  return true;
}

template <typename T>
bool FreshBANG<T>::CreateAlgo()
{
  std::lock_guard<std::recursive_mutex> api_lock(cuvs::bench::get_bench_api_mutex());

  std::cout << "[FreshBANG::CreateAlgo] called" << std::endl;

  constexpr auto config_path = "freshbang.conf";
  auto config                = cuvs::bench::detail::read_freshbang_config(config_path);

  const auto data_prefix_norm  = cuvs::bench::detail::normalize_prefix(config.data_prefix, "data/");
  const auto index_prefix_norm = cuvs::bench::detail::normalize_prefix(config.index_prefix, "index/");

  std::ifstream conf_stream(config.conf_path);
  cuvs::bench::ensure_stream_open(conf_stream, config.conf_path);

  auto& conf                  =
    cuvs::bench::configuration::initialize(conf_stream, data_prefix_norm, index_prefix_norm);
  const std::string index_name = "";
  const std::size_t index_pos  = 0;
  const auto& index            = cuvs::bench::detail::pick_index(conf, index_name, index_pos);

  cuvs::bench::validate_index_file(index.file, true);

  if (m_pImpl == nullptr) {
    m_pImpl =
      new cuvs::bench::detail::FreshBANGInner<T>(config.conf_path, data_prefix_norm, index_prefix_norm);
  }
  auto* impl = static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);

  std::cout << "[FreshBANG::CreateAlgo] Calling invoke_create_algo for algo='" << index.algo
            << "' distance='" << user_build_params_.distance_measure << "' dim="
            << user_build_params_.dataset_dim << std::endl;

  auto algo = cuvs::bench::detail::invoke_create_algo<T>(index.algo,
                                                          user_build_params_.distance_measure,
                                                          static_cast<int>(user_build_params_.dataset_dim),
                                                          index.build_param);

  std::cout << "[FreshBANG::CreateAlgo] invoke_create_algo returned algo_ptr=" << algo.get()
            << std::endl;

  if (!algo) {
    std::cerr << "[FreshBANG::CreateAlgo] ERROR: create_algo returned null!" << std::endl;
    return false;
  }

  impl->set_index_info(index.file, index.algo, static_cast<int>(user_build_params_.dataset_dim));

  cuvs::bench::detail::algo_cache_entry cache_entry;
  cache_entry.algo_ptr   = algo.release();
  cache_entry.algo_name  = index.algo;
  cache_entry.dtype      = cuvs::bench::get_dtype_string<T>();
  cache_entry.dim        = static_cast<int>(user_build_params_.dataset_dim);
  cache_entry.timestamp  = std::chrono::steady_clock::now();
  cache_entry.deleter    = [](void* ptr) { delete static_cast<cuvs::bench::algo<T>*>(ptr); };

  std::cout << "[FreshBANG::CreateAlgo] Caching algo with index_file='" << index.file << "'"
            << std::endl;
  cuvs::bench::detail::put_cached_algo_entry(index.file, std::move(cache_entry));
  std::cout << "[FreshBANG::CreateAlgo] completed successfully" << std::endl;
  return true;
}

template <typename T>
bool FreshBANG<T>::BuildIndex(const T* base_vectors, uint32_t num_basevectors)
{
  std::lock_guard<std::recursive_mutex> api_lock(cuvs::bench::get_bench_api_mutex());

  auto dim = user_build_params_.dataset_dim;

  if (m_pImpl == nullptr) {
    std::cerr << "[FreshBANG::BuildIndex] Error: CreateAlgo() must be called first." << std::endl;
    return false;
  }

  auto* impl  = static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);
  auto cached = cuvs::bench::detail::get_cached_algo_entry(impl->index_file(),
                                                            impl->algo_name(),
                                                            cuvs::bench::get_dtype_string<T>(),
                                                            static_cast<int>(dim),
                                                            std::chrono::minutes(60));

  if (!cached.has_value()) {
    std::cerr << "[FreshBANG::BuildIndex] Error: algo cache miss for index file "
              << impl->index_file() << ". Call CreateAlgo() again." << std::endl;
    return false;
  }

  std::cout << "[FreshBANG::BuildIndex] num_basevectors=" << num_basevectors << " dim=" << dim
            << " base_vectors=" << static_cast<const void*>(base_vectors) << std::endl;

  auto* algo_obj = static_cast<cuvs::bench::algo<T>*>(cached->algo_ptr);
  if (algo_obj == nullptr) {
    std::cerr << "[FreshBANG::BuildIndex] Error: cached algo pointer is null." << std::endl;
    return false;
  }

  std::cout << "[FreshBANG::BuildIndex] Calling set_build_output_file('" << impl->index_file() << "')"
            << std::endl;
  algo_obj->set_build_output_file(impl->index_file());

  std::cout << "[FreshBANG::BuildIndex] Calling build() with " << num_basevectors << " vectors..."
            << std::endl;
  // Note: base_vector can be on host or device. CAGRA wrapper will handle copying if needed.
  algo_obj->build(base_vectors, num_basevectors);
  std::cout << "[FreshBANG::BuildIndex] build() completed" << std::endl;

  auto index_path = std::filesystem::path(impl->index_file());
  if (index_path.has_parent_path()) {
    std::filesystem::create_directories(index_path.parent_path());
    std::cout << "[FreshBANG::BuildIndex] Ensured directory exists: " << index_path.parent_path()
              << std::endl;
  }
  std::cout << "[FreshBANG::BuildIndex] Calling save('" << impl->index_file() << "')" << std::endl;
  algo_obj->save(impl->index_file());
  std::cout << "[FreshBANG::BuildIndex] save() completed" << std::endl;

  return true;
}

template <typename T>
bool FreshBANG<T>::SetSearchParams(SearchParams params)
{
  std::lock_guard<std::recursive_mutex> api_lock(cuvs::bench::get_bench_api_mutex());

  std::cout << "[FreshBANG::SetSearchParams] recall_at_k=" << params.recall_at_k << std::endl;
  if (params.recall_at_k == 0) {
    std::cerr << "[FreshBANG::SetSearchParams] Error: recall_at_k must be greater than zero."
              << std::endl;
    return false;
  }

  if (m_pImpl == nullptr) {
    std::cerr << "[FreshBANG::SetSearchParams] Error: CreateAlgo() must be called before "
                 "SetSearchParams()."
              << std::endl;
    return false;
  }

  auto* impl = static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);
  auto cached = cuvs::bench::detail::get_cached_algo_entry(impl->index_file(),
                                                            impl->algo_name(),
                                                            cuvs::bench::get_dtype_string<T>(),
                                                            impl->dim(),
                                                            std::chrono::minutes(60));
  if (!cached.has_value()) {
    std::cerr << "[FreshBANG::SetSearchParams] Error: algo cache miss for index file "
              << impl->index_file() << ". Call CreateAlgo() again." << std::endl;
    return false;
  }

  auto* algo_obj = static_cast<cuvs::bench::algo<T>*>(cached->algo_ptr);
  if (algo_obj == nullptr) {
    std::cerr << "[FreshBANG::SetSearchParams] Error: cached algo pointer is null." << std::endl;
    return false;
  }

  constexpr auto config_path = "freshbang.conf";
  auto config                = cuvs::bench::detail::read_freshbang_config(config_path);
  const auto data_prefix_norm =
    cuvs::bench::detail::normalize_prefix(config.data_prefix, "data/");
  const auto index_prefix_norm =
    cuvs::bench::detail::normalize_prefix(config.index_prefix, "index/");

  std::ifstream conf_stream(config.conf_path);
  cuvs::bench::ensure_stream_open(conf_stream, config.conf_path);

  auto& conf =
    cuvs::bench::configuration::initialize(conf_stream, data_prefix_norm, index_prefix_norm);
  const std::string index_name = "";
  const std::size_t index_pos  = 0;
  const auto& index            = cuvs::bench::detail::pick_index(conf, index_name, index_pos);

  if (index.search_params.empty()) {
    std::cerr << "[FreshBANG::SetSearchParams] Error: search_params is empty in config for index '"
              << index.algo << "'." << std::endl;
    return false;
  }

  auto sp_json = index.search_params[0];
  sp_json["k"] = params.recall_at_k;

  auto dataset_for_search = cuvs::bench::make_dataset<T>(conf.get_dataset_conf(), true);
  auto search_param = cuvs::bench::detail::create_search_param<T>(index.algo, sp_json);
  const auto algo_property =
    cuvs::bench::parse_algo_property_override(algo_obj->get_preference(), sp_json);

  // print the algo_property for debugging
  std::cout << "[FreshBANG::SetSearchParams] algo_property.dataset_memory_type="
            << static_cast<int>(algo_property.dataset_memory_type)
            << " algo_property.query_memory_type=" << static_cast<int>(algo_property.query_memory_type)
            << std::endl;  
  // ToDo: Is it really required? 
  // Store dataset (lifetime) and algo_property (for query staging in BatchedSearch)
  impl->set_algo_property(algo_property);
   //impl->set_search_dataset(dataset_for_search);

  // We need to explicitly set the dataset, the previous build step wouldn't set it for us
  if (search_param->needs_dataset()) { // returns true for Cagra
    // ToDo: Remove dependency on reading dataset portion of JSON config. Ideally we should only access
    // the index portion
    algo_obj->set_search_dataset(dataset_for_search->base_set(algo_property.dataset_memory_type),
                                 dataset_for_search->base_set_size());
  }

  algo_obj->set_search_param(*search_param,
                             dataset_for_search->filter_bitset(algo_property.dataset_memory_type));
  user_search_params_ = params;
  return true;
}

template <typename T>
void FreshBANG<T>::BatchedSearch(
  const T* query_vectors, uint32_t batch_size, uint64_t* ids, uint32_t* neighbors, float* distances)
{
  std::lock_guard<std::recursive_mutex> api_lock(cuvs::bench::get_bench_api_mutex());

  std::cout << "[FreshBANG::BatchedSearch] batch_size=" << batch_size
            << " query_vectors=" << static_cast<const void*>(query_vectors)
            << " ids=" << static_cast<void*>(ids)
            << " neighbors=" << static_cast<void*>(neighbors)
            << " distances=" << static_cast<void*>(distances) << std::endl;

  if (m_pImpl == nullptr) {
    std::cerr << "[FreshBANG::BatchedSearch] Error: CreateAlgo() must be called first."
              << std::endl;
    return;
  }

  if (user_search_params_.recall_at_k == 0) {
    std::cerr << "[FreshBANG::BatchedSearch] Error: SetSearchParams() must be called with a "
                 "non-zero recall_at_k before BatchedSearch()."
              << std::endl;
    return;
  }

  if (query_vectors == nullptr || neighbors == nullptr || distances == nullptr) {
    std::cerr << "[FreshBANG::BatchedSearch] Error: query_vectors, neighbors, and distances "
                 "must be non-null."
              << std::endl;
    return;
  }

  auto* impl = static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);
  auto cached = cuvs::bench::detail::get_cached_algo_entry(impl->index_file(),
                                                            impl->algo_name(),
                                                            cuvs::bench::get_dtype_string<T>(),
                                                            impl->dim(),
                                                            std::chrono::minutes(60));
  if (!cached.has_value()) {
    std::cerr << "[FreshBANG::BatchedSearch] Error: algo cache miss for index file "
              << impl->index_file() << ". Call CreateAlgo() again." << std::endl;
    return;
  }

  auto* algo_obj = static_cast<cuvs::bench::algo<T>*>(cached->algo_ptr);
  if (algo_obj == nullptr) {
    std::cerr << "[FreshBANG::BatchedSearch] Error: cached algo pointer is null." << std::endl;
    return;
  }

  const auto elem_count =
    static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(user_build_params_.dataset_dim);

  // Copy host queries to device for cagra::search path.
  T* device_query_vectors = nullptr;
  if (cudaMalloc(reinterpret_cast<void**>(&device_query_vectors), elem_count * sizeof(T)) !=
      cudaSuccess) {
    std::cerr << "[FreshBANG::BatchedSearch] Error: cudaMalloc failed for query buffer."
              << std::endl;
    return;
  }
  auto device_query_cleanup =
    std::unique_ptr<T, void (*)(T*)>(device_query_vectors, [](T* p) {
      if (p != nullptr) { cudaFree(p); }
    });

  if (cudaMemcpy(device_query_vectors,
                 query_vectors,
                 elem_count * sizeof(T),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    std::cerr << "[FreshBANG::BatchedSearch] Error: cudaMemcpyHostToDevice failed."
              << std::endl;
    return;
  }

  const T* search_queries = device_query_vectors;

  const std::size_t result_count =
    static_cast<std::size_t>(batch_size) * static_cast<std::size_t>(user_search_params_.recall_at_k);
  std::vector<cuvs::bench::algo_base::index_type> neighbors64(result_count, 0);

  algo_obj->search_ex(search_queries,
                      static_cast<int>(batch_size),
                      static_cast<int>(user_search_params_.recall_at_k),
                      neighbors64.data(),
                      distances,
                      reinterpret_cast<int64_t*>(ids));

  for (std::size_t i = 0; i < result_count; ++i) {
    const auto n = neighbors64[i];
    if (n < 0 || n > static_cast<cuvs::bench::algo_base::index_type>(
                      std::numeric_limits<uint32_t>::max())) {
      std::cerr << "[FreshBANG::BatchedSearch] Error: neighbor index out of uint32_t range." << std::endl;
      return;
    }
    neighbors[i] = static_cast<uint32_t>(n);
  }
  
}

template <typename T>
bool FreshBANG<T>::SetInsertParams()
{
  std::lock_guard<std::recursive_mutex> api_lock(cuvs::bench::get_bench_api_mutex());

  std::cout << "[FreshBANG::SetInsertParams] called" << std::endl;

  if (m_pImpl == nullptr) {
    std::cerr << "[FreshBANG::SetInsertParams] Error: CreateAlgo() must be called before "
                 "SetInsertParams()."
              << std::endl;
    return false;
  }

  auto* impl = static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);
  auto cached = cuvs::bench::detail::get_cached_algo_entry(impl->index_file(),
                                                            impl->algo_name(),
                                                            cuvs::bench::get_dtype_string<T>(),
                                                            impl->dim(),
                                                            std::chrono::minutes(60));
  if (!cached.has_value()) {
    std::cerr << "[FreshBANG::SetInsertParams] Error: algo cache miss for index file "
              << impl->index_file() << ". Call CreateAlgo() again." << std::endl;
    return false;
  }

  auto* algo_obj = static_cast<cuvs::bench::algo<T>*>(cached->algo_ptr);
  if (algo_obj == nullptr) {
    std::cerr << "[FreshBANG::SetInsertParams] Error: cached algo pointer is null." << std::endl;
    return false;
  }

  constexpr auto config_path = "freshbang.conf";
  auto config                = cuvs::bench::detail::read_freshbang_config(config_path);
  const std::size_t index_pos = 0;

  nlohmann::json insert_params_json = nlohmann::json::object();
  try {
    std::ifstream conf_json_stream(config.conf_path);
    if (!conf_json_stream) {
      throw std::runtime_error("Cannot open configuration file: " + config.conf_path);
    }

    auto conf_json = nlohmann::json::parse(conf_json_stream);
    if (conf_json.contains("index") && conf_json.at("index").is_array() &&
        conf_json.at("index").size() > index_pos) {
      const auto& idx_json = conf_json.at("index").at(index_pos);
      if (idx_json.contains("insert_params")) {
        if (idx_json.at("insert_params").is_array() && !idx_json.at("insert_params").empty()) {
          insert_params_json = idx_json.at("insert_params").at(0);
        } else if (idx_json.at("insert_params").is_object()) {
          insert_params_json = idx_json.at("insert_params");
        }
      } else if (idx_json.contains("insert_param") && idx_json.at("insert_param").is_object()) {
        insert_params_json = idx_json.at("insert_param");
      }
    }
  } catch (const std::exception& e) {
    std::cerr << "[FreshBANG::SetInsertParams] Warning: failed to parse insert params from "
              << config.conf_path << ": " << e.what() << std::endl;
    std::cerr << "[FreshBANG::SetInsertParams] Using default insert params" << std::endl;
    insert_params_json = nlohmann::json::object();
  }

  std::cout << "[FreshBANG::SetInsertParams] Applying insert params: "
            << insert_params_json.dump() << std::endl;
  algo_obj->set_insert_param_from_json(insert_params_json);

  return true;
}

template <typename T>
void FreshBANG<T>::BatchedInsert(const T* insertvectors, uint32_t batch_size, const uint64_t* ids)
{
  std::cout << "[FreshBANG::BatchedInsert] batch_size=" << batch_size
            << " insertvectors=" << static_cast<const void*>(insertvectors)
            << " ids=" << static_cast<const void*>(ids) << std::endl;

  if (m_pImpl == nullptr) {
    std::cerr << "[FreshBANG::BatchedInsert] Error: CreateAlgo() must be called before "
                 "BatchedInsert()."
              << std::endl;
    return;
  }

  if (insertvectors == nullptr) {
    std::cerr << "[FreshBANG::BatchedInsert] Error: insertvectors must be non-null." << std::endl;
    return;
  }

  auto* impl = static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);
  auto cached = cuvs::bench::detail::get_cached_algo_entry(impl->index_file(),
                                                            impl->algo_name(),
                                                            cuvs::bench::get_dtype_string<T>(),
                                                            impl->dim(),
                                                            std::chrono::minutes(60));
  if (!cached.has_value()) {
    std::cerr << "[FreshBANG::BatchedInsert] Error: algo cache miss for index file "
              << impl->index_file() << ". Call CreateAlgo() again." << std::endl;
    return;
  }
  auto* algo_obj = static_cast<cuvs::bench::algo<T>*>(cached->algo_ptr);
  if (algo_obj == nullptr) {
    std::cerr << "[FreshBANG::BatchedInsert] Error: cached algo pointer is null." << std::endl;
    return;
  }

  // Special case: if no index exists on disk, treat insert as build
  bool has_index_on_disk = cuvs::bench::detail::file_exists(impl->index_file());
  if (!has_index_on_disk) {
    std::cerr << "[FreshBANG::BatchedInsert] Warning: index file '" << impl->index_file()
              << "' does not exist on disk. Treating insert as build operation." << std::endl;
    
    algo_obj->set_build_output_file(impl->index_file());
    std::cout << "[FreshBANG::BatchedInsert] Calling build() with " << batch_size << " vectors"
              << std::endl;
    algo_obj->build(insertvectors, batch_size);
    std::cout << "[FreshBANG::BatchedInsert] build() completed" << std::endl;

    auto index_path = std::filesystem::path(impl->index_file());
    if (index_path.has_parent_path()) {
      std::filesystem::create_directories(index_path.parent_path());
      std::cout << "[FreshBANG::BatchedInsert] Ensured directory exists: " << index_path.parent_path()
                << std::endl;
    }
    std::cout << "[FreshBANG::BatchedInsert] Calling save('" << impl->index_file() << "')" << std::endl;
    algo_obj->save(impl->index_file());
    std::cout << "[FreshBANG::BatchedInsert] save() completed" << std::endl;
    // ToDo: check if we can assign hostside insertvectors like this directly.
        algo_obj->set_search_dataset(insertvectors,
                                 batch_size);
                                 return;
  }

  algo_obj->insert(insertvectors, batch_size, ids);

  std::cout << "[INSERT] Insert completed successfully" << std::endl;

}

template <typename T>
void FreshBANG<T>::BatchedDelete(const uint64_t* ids, uint32_t batch_size)
{
  std::cout << "[FreshBANG::BatchedDelete] batch_size=" << batch_size
            << " ids=" << static_cast<const void*>(ids) << std::endl;
}

template <typename T>
void FreshBANG<T>::Cleanup()
{
  std::cout << "[FreshBANG::Cleanup] called" << std::endl;
  cuvs::bench::detail::clear_cached_algo_entries();
  auto* impl = static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);
  if (impl == nullptr) {
    return;
  }

  // if index file exists, remove it 
  std::filesystem::path index_path(impl->index_file());
  if (std::filesystem::exists(index_path)) {
    std::error_code ec;
    std::filesystem::remove(index_path, ec);
    if (ec) {
      std::cerr << "[FreshBANG::Cleanup] Warning: failed to remove index file " << index_path
                << ": " << ec.message() << std::endl;
    } else {
      std::cout << "[FreshBANG::Cleanup] Removed index file: " << index_path << std::endl;
    }
  }
  delete impl;
  m_pImpl = nullptr;
  
}

// freshbang.hpp already has the explicit instantiation; no duplicate needed here.
