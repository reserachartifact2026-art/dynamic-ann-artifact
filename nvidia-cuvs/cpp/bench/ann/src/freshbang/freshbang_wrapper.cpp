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

inline auto is_device_pointer(const void* ptr) -> bool
{
  if (ptr == nullptr) { return false; }

  cudaPointerAttributes attr{};
  auto status = cudaPointerGetAttributes(&attr, ptr);
  if (status != cudaSuccess) {
    // Host pointers can fail this query on some CUDA/runtime combinations.
    cudaGetLastError();
    return false;
  }

#if CUDART_VERSION >= 10000
  return attr.type == cudaMemoryTypeDevice || attr.type == cudaMemoryTypeManaged;
#else
  return attr.memoryType == cudaMemoryTypeDevice;
#endif
}

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
  std::string data_prefix{"data/"};
  std::string index_prefix{"index/"};
};

auto mode_to_string(FreshBANGMode mode) -> const char*
{
  switch (mode) {
    case FreshBANGMode::kLegacy: return "legacy";
    case FreshBANGMode::kBuild: return "build";
    case FreshBANGMode::kLoad: return "load";
  }
  return "unknown";
}

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

    const auto comment_pos = trimmed_line.find('#');
    if (comment_pos != std::string::npos) {
      trimmed_line = trim_whitespace(trimmed_line.substr(0, comment_pos));
      if (trimmed_line.empty()) { continue; }
    }

    auto separator_pos = trimmed_line.find('=');
    if (separator_pos == std::string::npos) { continue; }

    auto key   = trim_whitespace(trimmed_line.substr(0, separator_pos));
    auto value = trim_whitespace(trimmed_line.substr(separator_pos + 1));

    if (key == "data_prefix") {
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

    auto now     = std::chrono::steady_clock::now();
    auto elapsed = now - entry.timestamp;
    if (std::chrono::duration_cast<std::chrono::minutes>(elapsed) >= max_age) {
      if (entry.algo_ptr != nullptr && entry.deleter) {
        entry.deleter(entry.algo_ptr);
        entry.algo_ptr = nullptr;
      }
      cache_.erase(it);
      return std::nullopt;
    }

    // Sliding TTL: keep frequently-used algo entries alive during long stress runs.
    entry.timestamp = now;

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
  const std::string& conf_path() const { return conf_path_; }
  const std::string& data_prefix() const { return data_prefix_; }
  const std::string& index_prefix() const { return index_prefix_; }

  void set_conf_path(std::string conf_path) { conf_path_ = std::move(conf_path); }

  void set_algo_property(cuvs::bench::algo_property prop) { algo_property_ = prop; }

  void set_prefer_insert_dataset_for_search(bool value) { prefer_insert_dataset_for_search_ = value; }
  bool prefer_insert_dataset_for_search() const { return prefer_insert_dataset_for_search_; }

  void set_has_live_index(bool value) 
  { has_live_index_ = value;
    // print the new value to console for debugging
    std::cout << "[FreshBANGInner] set_has_live_index: " << std::boolalpha << has_live_index_ << std::endl; 
  }
  bool has_live_index() const { return has_live_index_; }

  void set_mode(FreshBANGMode mode) { mode_ = mode; }
  FreshBANGMode mode() const { return mode_; }

 private:
  std::string conf_path_;
  std::string data_prefix_;
  std::string index_prefix_;

  std::string index_file_;
  std::string algo_name_;
  int dim_{0};
  cuvs::bench::algo_property algo_property_{};
  bool prefer_insert_dataset_for_search_{false};
  bool has_live_index_{false};
  FreshBANGMode mode_{FreshBANGMode::kLegacy};
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
  // save the index to disk if it has been built
  if (m_pImpl != nullptr) {
    auto* impl = static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);
    if (impl->has_live_index()) {
      std::cout << "[FreshBANG] Destructor: Saving index to disk before cleanup" << std::endl;
      SaveIndex();
    }
    delete impl;
    m_pImpl = nullptr;
  }
  std::cout << "[FreshBANG] Destructor called" << std::endl;
}

template <typename T>
bool FreshBANG<T>::SetDatasetParams(BuildParams params)
{
  std::cout << "[FreshBANG::SetDatasetParams] dataset_rows=" << params.dataset_rows
            << " dataset_dim=" << params.dataset_dim
            << " distance_measure=" << params.distance_measure << std::endl;
  user_build_params_ = std::move(params);
  return true;
}

template <typename T>
bool FreshBANG<T>::CreateAlgo(const std::string& conf_path, FreshBANGMode mode)
{
  std::lock_guard<std::recursive_mutex> api_lock(cuvs::bench::get_bench_api_mutex());

  std::cout << "[FreshBANG::CreateAlgo] called mode=" << cuvs::bench::detail::mode_to_string(mode)
            << std::endl;
  if (conf_path.empty()) {
    std::cerr << "[FreshBANG::CreateAlgo] Error: conf_path must be non-empty." << std::endl;
    return false;
  }

  constexpr auto config_path = "freshbang.cfg";
  auto config                = cuvs::bench::detail::read_freshbang_config(config_path);

  const auto data_prefix_norm  = cuvs::bench::detail::normalize_prefix(config.data_prefix, "data/");
  const auto index_prefix_norm = cuvs::bench::detail::normalize_prefix(config.index_prefix, "index/");

  std::ifstream conf_stream(conf_path);
  cuvs::bench::ensure_stream_open(conf_stream, conf_path);

  auto& conf                  =
    cuvs::bench::configuration::initialize(conf_stream, data_prefix_norm, index_prefix_norm);
  const std::string index_name = "";
  const std::size_t index_pos  = 0;
  const auto& index            = cuvs::bench::detail::pick_index(conf, index_name, index_pos);

  cuvs::bench::validate_index_file(index.file, true);

  if (m_pImpl == nullptr) {
    m_pImpl =
      new cuvs::bench::detail::FreshBANGInner<T>(conf_path, data_prefix_norm, index_prefix_norm);
  }
  auto* impl = static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);
  impl->set_conf_path(conf_path);
  impl->set_mode(mode);
  impl->set_has_live_index(false);

  std::cout << "[FreshBANG::CreateAlgo] Calling invoke_create_algo for algo='" << index.algo
            << "' distance='" << user_build_params_.distance_measure << "' rows="
            << user_build_params_.dataset_rows << " dim="
            << user_build_params_.dataset_dim << std::endl;

  auto algo = cuvs::bench::detail::invoke_create_algo<T>(index.algo,
                                                          user_build_params_.distance_measure,
                                                          static_cast<int>(user_build_params_.dataset_rows),
                                                          static_cast<int>(user_build_params_.dataset_dim),
                                                          index.build_param);

  std::cout << "[FreshBANG::CreateAlgo] invoke_create_algo returned algo_ptr=" << algo.get()
            << std::endl;

  if (!algo) {
    std::cerr << "[FreshBANG::CreateAlgo] ERROR: create_algo returned null!" << std::endl;
    return false;
  }

  impl->set_index_info(index.file, index.algo, static_cast<int>(user_build_params_.dataset_dim));

  if (mode == FreshBANGMode::kLoad) {
    if (!cuvs::bench::detail::file_exists(index.file)) {
      std::cerr << "[FreshBANG::CreateAlgo] Error: --load mode requires an existing index file at '"
                << index.file << "'." << std::endl;
      return false;
    }

    std::cout << "[FreshBANG::CreateAlgo] Load mode: loading index from disk: " << index.file
              << std::endl;
    try {
      algo->load(index.file);
      impl->set_has_live_index(true);
      std::cout << "[FreshBANG::CreateAlgo] Load mode: load() completed" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "[FreshBANG::CreateAlgo] Error: failed to load index '" << index.file
                << "': " << e.what() << std::endl;
      return false;
    }
  }

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

  const auto index_path = std::filesystem::path(impl->index_file());
  const auto mode = impl->mode();
  const bool has_index_on_disk = cuvs::bench::detail::file_exists(impl->index_file());
  std::cout << "[FreshBANG::BuildIndex] mode=" << cuvs::bench::detail::mode_to_string(mode)
            << " index_file='" << impl->index_file() << "' has_index_on_disk=" << std::boolalpha
            << has_index_on_disk << std::endl;

  auto log_gpu_mem = [](const char* stage) {
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    auto mem_status    = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (mem_status != cudaSuccess) {
      std::cerr << "[FreshBANG::BuildIndex] " << stage
                << " cudaMemGetInfo failed: " << cudaGetErrorString(mem_status) << std::endl;
      return;
    }
    auto used_bytes = total_bytes - free_bytes;
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    std::cout << "[FreshBANG::BuildIndex] " << stage
              << " used_GiB=" << (static_cast<double>(used_bytes) / gib)
              << " free_GiB=" << (static_cast<double>(free_bytes) / gib)
              << " total_GiB=" << (static_cast<double>(total_bytes) / gib) << std::endl;
  };

  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "[FreshBANG::BuildIndex] Warning: cudaDeviceSynchronize failed before load/build."
              << std::endl;
  }
  log_gpu_mem("before_load_or_build");

  if (mode == FreshBANGMode::kLoad) {
    if (!impl->has_live_index()) {
      std::cerr << "[FreshBANG::BuildIndex] Error: load mode expects index to be loaded by "
                   "CreateAlgo(), but no live index is available." << std::endl;
      return false;
    }
    std::cout << "[FreshBANG::BuildIndex] Load mode: skipping build; index already loaded by "
                 "CreateAlgo" << std::endl;
    if (cudaDeviceSynchronize() != cudaSuccess) {
      std::cerr << "[FreshBANG::BuildIndex] Warning: cudaDeviceSynchronize failed in load mode."
                << std::endl;
    }
    log_gpu_mem("after_load_from_create_algo");
    return true;
  }

  if (mode == FreshBANGMode::kLegacy) {
    std::cout << "[FreshBANG::BuildIndex] Legacy mode selected; always building fresh index"
              << std::endl;
  }

  if (mode == FreshBANGMode::kBuild) {
    std::cout << "[FreshBANG::BuildIndex] Build mode selected; forcing build and allowing overwrite"
              << std::endl;
  }

  std::cout << "[FreshBANG::BuildIndex] Calling set_build_output_file('" << impl->index_file() << "')"
            << std::endl;
  algo_obj->set_build_output_file(impl->index_file());

  std::cout << "[FreshBANG::BuildIndex] Calling build() with " << num_basevectors << " vectors..."
            << std::endl;

  std::cout << "[FreshBANG::BuildIndex] Building new index" << std::endl;

  // Note: base_vector can be on host or device. CAGRA wrapper will handle copying if needed.
  algo_obj->build(base_vectors, num_basevectors);

  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::cerr << "[FreshBANG::BuildIndex] Warning: cudaDeviceSynchronize failed after build."
              << std::endl;
  }
  log_gpu_mem("after_build");

  std::cout << "[FreshBANG::BuildIndex] build() completed" << std::endl;

  if (index_path.has_parent_path()) {
    std::filesystem::create_directories(index_path.parent_path());
    std::cout << "[FreshBANG::BuildIndex] Ensured directory exists: " << index_path.parent_path()
              << std::endl;
  }
#ifdef _KVDEBUG
  std::cout << "[FreshBANG::BuildIndex] Calling save('" << "/mnt/ssd_volume/cuvs_benchmarks/index/./datasets/sift10k/index/cuvs_cagra.graph_degree32.intermediate_graph_degree32.graph_build_algoNN_DESCENT.before" << "')" << std::endl;
  algo_obj->save("/mnt/ssd_volume/cuvs_benchmarks/index/./datasets/sift10k/index/cuvs_cagra.graph_degree32.intermediate_graph_degree32.graph_build_algoNN_DESCENT.before");
  std::cout << "[FreshBANG::BuildIndex] save() completed" << std::endl;
#endif
  impl->set_has_live_index(true);

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
  if (impl->mode() == FreshBANGMode::kBuild) {
    std::cout << "[FreshBANG::SetSearchParams] Build mode: no-op" << std::endl;
    return true;
  }

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

  std::ifstream conf_stream(impl->conf_path());
  cuvs::bench::ensure_stream_open(conf_stream, impl->conf_path());

  auto& conf =
    cuvs::bench::configuration::initialize(conf_stream, impl->data_prefix(), impl->index_prefix());
  const std::string index_name = "";
  const std::size_t index_pos  = 0;
  const auto& index            = cuvs::bench::detail::pick_index(conf, index_name, index_pos);

  if (index.search_params.empty()) {
    std::cerr << "[FreshBANG::SetSearchParams] Error: search_params is empty in config for index '"
              << index.algo << "'." << std::endl;
    return false;
  }

  auto sp_json = index.search_params[0];
  // log the search parameters for debugging
  std::cout << "[FreshBANG::SetSearchParams] Original search_params JSON: " << sp_json.dump()
            << std::endl;
  sp_json["k"] = params.recall_at_k;
  
  auto dataset_for_search = cuvs::bench::make_dataset<T>(
    conf.get_dataset_conf(), true, impl->prefer_insert_dataset_for_search());
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



  // We need to explicitly set the dataset, the previous build step wouldn't set it for us
  if (search_param->needs_dataset()) { // returns always true for Cagra
    if (impl->prefer_insert_dataset_for_search() && dataset_for_search->has_insert_set()) {
      algo_obj->set_search_dataset(dataset_for_search->insert_set(algo_property.dataset_memory_type),
                                   dataset_for_search->insert_set_size());
    } else {
      // ToDo: Remove dependency on reading dataset portion of JSON config. Ideally we should only access
      // the index portion
      algo_obj->set_search_dataset(dataset_for_search->base_set(algo_property.dataset_memory_type),
                                   dataset_for_search->base_set_size());
    }
  }


  algo_obj->set_search_param(*search_param,
                             dataset_for_search->filter_bitset(algo_property.dataset_memory_type));
  user_search_params_ = params;

  impl->set_has_live_index(true);

  // Sort the adjacency list (default is a no-op for non-supporting algos).
  //algo_obj->preprocess_built_index();

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
  if (impl->mode() == FreshBANGMode::kBuild) {
    std::cout << "[FreshBANG::BatchedSearch] Build mode: no-op" << std::endl;
    return;
  }

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

  // Accept both host and device query pointers.
  const T* search_queries = query_vectors;
  std::unique_ptr<T, void (*)(T*)> device_query_cleanup(nullptr, [](T* p) {
    if (p != nullptr) { cudaFree(p); }
  });

  bool queries_on_device = cuvs::bench::detail::is_device_pointer(query_vectors);
  if (!queries_on_device) {
    T* device_query_vectors = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&device_query_vectors), elem_count * sizeof(T)) !=
        cudaSuccess) {
      std::cerr << "[FreshBANG::BatchedSearch] Error: cudaMalloc failed for query buffer."
                << std::endl;
      return;
    }
    device_query_cleanup.reset(device_query_vectors);

    if (cudaMemcpy(device_query_vectors,
                   query_vectors,
                   elem_count * sizeof(T),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      std::cerr << "[FreshBANG::BatchedSearch] Error: cudaMemcpyHostToDevice failed."
                << std::endl;
      return;
    }

    search_queries = device_query_vectors;
  }

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
  if (impl->mode() == FreshBANGMode::kBuild) {
    std::cout << "[FreshBANG::SetInsertParams] mode="
              << cuvs::bench::detail::mode_to_string(impl->mode()) << ": no-op" << std::endl;
    return true;
  }

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

  const std::size_t index_pos = 0;

  nlohmann::json insert_params_json = nlohmann::json::object();
  try {
    std::ifstream conf_json_stream(impl->conf_path());
    if (!conf_json_stream) {
      throw std::runtime_error("Cannot open configuration file: " + impl->conf_path());
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
              << impl->conf_path() << ": " << e.what() << std::endl;
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
            << " ids=" << ids[0] << " to " << ids[batch_size - 1] << std::endl;

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

  // Debug: print first 10 components of first insert vector
  std::cout << "[FreshBANG::BatchedInsert] First insert vector (first 10 components): ";
  for (int i = 0; i < std::min(10, static_cast<int>(user_build_params_.dataset_dim)); ++i) {
    std::cout << static_cast<double>(insertvectors[i]) << " ";
  }
  std::cout << std::endl;

  auto* impl = static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);
  if (impl->mode() == FreshBANGMode::kBuild) {
    std::cout << "[FreshBANG::BatchedInsert] mode="
              << cuvs::bench::detail::mode_to_string(impl->mode()) << ": no-op" << std::endl;
    return;
  }

  auto cached = cuvs::bench::detail::get_cached_algo_entry(impl->index_file(),
                                                            impl->algo_name(),
                                                            cuvs::bench::get_dtype_string<T>(),
                                                            impl->dim(),
                                                            std::chrono::minutes(60));
  if (!cached.has_value()) {
    std::cerr << "[FreshBANG::BatchedInsert] Warning: algo cache miss for index file "
              << impl->index_file() << ". Attempting in-place recovery via CreateAlgo()."
              << std::endl;
    auto recover_mode = impl->mode();
    auto recover_conf = impl->conf_path();
    if (!CreateAlgo(recover_conf, recover_mode)) {
      std::cerr << "[FreshBANG::BatchedInsert] Error: recovery CreateAlgo() failed." << std::endl;
      return;
    }
    impl   = static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);
    cached = cuvs::bench::detail::get_cached_algo_entry(impl->index_file(),
                                                         impl->algo_name(),
                                                         cuvs::bench::get_dtype_string<T>(),
                                                         impl->dim(),
                                                         std::chrono::minutes(60));
    if (!cached.has_value()) {
      std::cerr << "[FreshBANG::BatchedInsert] Error: cache miss persists after recovery." << std::endl;
      return;
    }
  }
  auto* algo_obj = static_cast<cuvs::bench::algo<T>*>(cached->algo_ptr);
  if (algo_obj == nullptr) {
    std::cerr << "[FreshBANG::BatchedInsert] Error: cached algo pointer is null." << std::endl;
    return;
  }

  // If an index exists on disk, load it into a fresh algo instance before insert.
  // Skip load when a valid in-process index is already available from BuildIndex/previous ops.
  bool has_index_on_disk = cuvs::bench::detail::file_exists(impl->index_file());
  bool use_build_path    = !has_index_on_disk;
  if (has_index_on_disk && !impl->has_live_index()) {
    try {
      std::cout << "[FreshBANG::BatchedInsert] Loading existing index from '"
                << impl->index_file() << "' before insert" << std::endl;
      algo_obj->load(impl->index_file());
      std::cout << "[FreshBANG::BatchedInsert] load() completed" << std::endl;
      impl->set_has_live_index(true);
    } catch (const std::exception& e) {
      std::cerr << "[FreshBANG::BatchedInsert] Warning: failed to load existing index '"
                << impl->index_file() << "': " << e.what()
                << ". Falling back to build operation." << std::endl;
      use_build_path = true;
    }
  } else if (has_index_on_disk && impl->has_live_index()) {
    std::cout << "[FreshBANG::BatchedInsert] Skipping load(); using in-memory index built in this process"
              << std::endl;
  }

  // Special case: if no index exists on disk (or load failed), treat insert as build.
  if (use_build_path) {
    std::cerr << "[FreshBANG::BatchedInsert] Warning: index unavailable for insert at '"
          << impl->index_file() << "'. Treating insert as build operation." << std::endl;

    algo_obj->set_build_output_file(impl->index_file());
    std::cout << "[FreshBANG::BatchedInsert] Calling build() with " << batch_size
              << " vectors" << std::endl;
    algo_obj->build(insertvectors, batch_size);
    std::cout << "[FreshBANG::BatchedInsert] build() completed" << std::endl;

    auto index_path = std::filesystem::path(impl->index_file());
    if (index_path.has_parent_path()) {
      std::filesystem::create_directories(index_path.parent_path());
      std::cout << "[FreshBANG::BatchedInsert] Ensured directory exists: "
                << index_path.parent_path() << std::endl;
    }
#if _KVDEBUG    
    std::cout << "[FreshBANG::BuildIndex] Calling save('" << "/mnt/ssd_volume/cuvs_benchmarks/index/./datasets/sift10k/index/cuvs_cagra.graph_degree32.intermediate_graph_degree32.graph_build_algoNN_DESCENT.before" << "')" << std::endl;
    algo_obj->save("/mnt/ssd_volume/cuvs_benchmarks/index/./datasets/sift10k/index/cuvs_cagra.graph_degree32.intermediate_graph_degree32.graph_build_algoNN_DESCENT.before");
    std::cout << "[FreshBANG::BatchedInsert] save() completed" << std::endl;
#endif    
    impl->set_has_live_index(true);
    impl->set_prefer_insert_dataset_for_search(true);
    // ToDo: check if we can assign hostside insertvectors like this directly.
/*        algo_obj->set_search_dataset(insertvectors,
                                 batch_size);
  */
  } else {
    impl->set_prefer_insert_dataset_for_search(false);
    algo_obj->insert(insertvectors, batch_size, ids);
  }

  std::cout << "[INSERT] Insert completed successfully" << std::endl;

}

template <typename T>
void FreshBANG<T>::BatchedDelete(const uint64_t* ids, uint32_t batch_size)
{
  std::cout << "[FreshBANG::BatchedDelete] batch_size=" << batch_size
            << " ids=" << ids[0] << " to " << ids[batch_size - 1] << std::endl;

  if (m_pImpl == nullptr) {
    std::cerr << "[FreshBANG::BatchedDelete] Error: CreateAlgo() must be called before "
                 "BatchedDelete()."
              << std::endl;
    return;
  }

  if (batch_size == 0) {
    std::cout << "[FreshBANG::BatchedDelete] batch_size is 0; skipping delete" << std::endl;
    return;
  }

  if (ids == nullptr) {
    std::cerr << "[FreshBANG::BatchedDelete] Error: ids must be non-null when batch_size > 0."
              << std::endl;
    return;
  }

  auto* impl = static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);
  if (impl->mode() == FreshBANGMode::kBuild) {
    std::cout << "[FreshBANG::BatchedDelete] mode="
              << cuvs::bench::detail::mode_to_string(impl->mode()) << ": no-op" << std::endl;
    return;
  }

  auto cached = cuvs::bench::detail::get_cached_algo_entry(impl->index_file(),
                                                            impl->algo_name(),
                                                            cuvs::bench::get_dtype_string<T>(),
                                                            impl->dim(),
                                                            std::chrono::minutes(60));
  if (!cached.has_value()) {
    std::cerr << "[FreshBANG::BatchedDelete] Warning: algo cache miss for index file "
              << impl->index_file() << ". Attempting in-place recovery via CreateAlgo()."
              << std::endl;
    auto recover_mode = impl->mode();
    auto recover_conf = impl->conf_path();
    if (!CreateAlgo(recover_conf, recover_mode)) {
      std::cerr << "[FreshBANG::BatchedDelete] Error: recovery CreateAlgo() failed." << std::endl;
      return;
    }
    impl   = static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);
    cached = cuvs::bench::detail::get_cached_algo_entry(impl->index_file(),
                                                         impl->algo_name(),
                                                         cuvs::bench::get_dtype_string<T>(),
                                                         impl->dim(),
                                                         std::chrono::minutes(60));
    if (!cached.has_value()) {
      std::cerr << "[FreshBANG::BatchedDelete] Error: cache miss persists after recovery." << std::endl;
      return;
    }
  }

  auto* algo_obj = static_cast<cuvs::bench::algo<T>*>(cached->algo_ptr);
  if (algo_obj == nullptr) {
    std::cerr << "[FreshBANG::BatchedDelete] Error: cached algo pointer is null." << std::endl;
    return;
  }

  // Mirror insert behavior: if index exists on disk but is not loaded in-process,
  // load it before mutating ops.
  bool has_index_on_disk = cuvs::bench::detail::file_exists(impl->index_file());
  if (has_index_on_disk && !impl->has_live_index()) {
    try {
      std::cout << "[FreshBANG::BatchedDelete] Loading existing index from '"
                << impl->index_file() << "' before delete" << std::endl;
      algo_obj->load(impl->index_file());
      std::cout << "[FreshBANG::BatchedDelete] load() completed" << std::endl;
      impl->set_has_live_index(true);
    } catch (const std::exception& e) {
      std::cerr << "[FreshBANG::BatchedDelete] Error: failed to load existing index '"
                << impl->index_file() << "': " << e.what() << std::endl;
      return;
    }
  } else if (!has_index_on_disk && !impl->has_live_index()) {
    std::cerr << "[FreshBANG::BatchedDelete] Error: index unavailable at '" << impl->index_file()
              << "' and no live in-memory index exists for delete." << std::endl;
    return;
  }

 algo_obj->delete_vectors(ids, static_cast<size_t>(batch_size));
  std::cout << "[FreshBANG::BatchedDelete] delete() completed" << std::endl;
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

  if (impl->mode() == FreshBANGMode::kLoad || impl->mode() == FreshBANGMode::kBuild) {
    std::cout << "[FreshBANG::Cleanup] mode="
              << cuvs::bench::detail::mode_to_string(impl->mode())
              << ": keeping index file on disk" << std::endl;
    delete impl;
    m_pImpl = nullptr;
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

template <typename T>
void FreshBANG<T>::SaveIndex()
{
  auto* impl = static_cast<cuvs::bench::detail::FreshBANGInner<T>*>(m_pImpl);
  if (impl == nullptr) {
    std::cerr << "[FreshBANG::SaveIndex] Error: CreateAlgo() must be called before SaveIndex()."
              << std::endl;
    return;
  }

  if (impl->mode() == FreshBANGMode::kLegacy) {
    std::cout << "[FreshBANG::SaveIndex] mode=" << cuvs::bench::detail::mode_to_string(impl->mode())
              << " follows legacy behavior: no-op" << std::endl;
    return;
  }

  auto cached = cuvs::bench::detail::get_cached_algo_entry(impl->index_file(),
                                                            impl->algo_name(),
                                                            cuvs::bench::get_dtype_string<T>(),
                                                            impl->dim(),
                                                            std::chrono::minutes(60));
  if (!cached.has_value()) {
    std::cerr << "[FreshBANG::SaveIndex] Error: algo cache miss for index file "
              << impl->index_file() << ". Call CreateAlgo() again." << std::endl;
    return;
  }

  auto* algo_obj = static_cast<cuvs::bench::algo<T>*>(cached->algo_ptr);
  if (algo_obj == nullptr) {
    std::cerr << "[FreshBANG::SaveIndex] Error: cached algo pointer is null." << std::endl;
    return;
  }

  if (impl->mode() == FreshBANGMode::kBuild) {
    algo_obj->save(impl->index_file());
    std::cout << "[FreshBANG::SaveIndex] Build mode: index saved to canonical path: "
              << impl->index_file() << std::endl;
  } else {
    auto temp_path = impl->index_file() + "_temp";
    algo_obj->save(temp_path);
    // In load mode, never overwrite the canonical index file.
    std::cout << "[FreshBANG::SaveIndex] Load mode: index saved to temp path: " << temp_path
              << std::endl;
  }
  std::cout << "[FreshBANG::SaveIndex] save() completed" << std::endl;
}
  

// freshbang.hpp already has the explicit instantiation; no duplicate needed here.
