
#include "../common/blob.hpp"
#include "../common/conf.hpp"
#include "freshbang.hpp"
#include "freshbang_helpers.hpp"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace {

enum class dataset_dtype { kFloat32, kInt32, kUInt8, kInt8, kUnknown };
#define PIPELINE_RUNS 0 // > 1, not working now

struct DriverConfig {
  std::string data_prefix;
  std::string index_prefix;
};

auto read_freshbang_config(const std::string& config_file = "freshbang.cfg")
  -> std::pair<bool, DriverConfig>
{
  std::ifstream conf_stream(config_file);
  DriverConfig config{};

  if (!conf_stream) {
    std::cerr << "[freshbang_driver] Warning: " << config_file << " not found, using current "
                 "directory as defaults"
              << std::endl;
    config.data_prefix  = ".";
    config.index_prefix = ".";
    return {true, config};
  }

  std::string line;
  while (std::getline(conf_stream, line)) {
    // Trim whitespace
    line.erase(0, line.find_first_not_of(" \t\r\n"));
    line.erase(line.find_last_not_of(" \t\r\n") + 1);

    // Skip comments and empty lines
    if (line.empty() || line[0] == '#') { continue; }

    size_t delimiter_pos = line.find('=');
    if (delimiter_pos == std::string::npos) { continue; }

    std::string key   = line.substr(0, delimiter_pos);
    std::string value = line.substr(delimiter_pos + 1);

    // Trim key and value
    key.erase(key.find_last_not_of(" \t") + 1);
    value.erase(0, value.find_first_not_of(" \t"));

    if (key == "data_prefix") {
      config.data_prefix = value;
    } else if (key == "index_prefix") {
      config.index_prefix = value;
    }
  }

  if (config.data_prefix.empty() || config.index_prefix.empty()) {
    std::cerr << "[freshbang_driver] Error: " << config_file
              << " must contain 'data_prefix' and 'index_prefix'" << std::endl;
    return {false, config};
  }

  return {true, config};
}

auto to_lower(std::string s) -> std::string
{
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

auto ends_with(const std::string& value, const std::string& suffix) -> bool
{
  if (value.size() < suffix.size()) { return false; }
  return value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

auto infer_dtype_from_base_file(const std::string& base_file) -> dataset_dtype
{
  auto name = to_lower(base_file);
  if (ends_with(name, ".fbin") || ends_with(name, "fbin")) { return dataset_dtype::kFloat32; }
  if (ends_with(name, ".ibin") || ends_with(name, "ibin")) { return dataset_dtype::kInt32; }
  if (ends_with(name, ".u8bin") || ends_with(name, "u8bin")) { return dataset_dtype::kUInt8; }
  if (ends_with(name, ".i8bin") || ends_with(name, "i8bin")) { return dataset_dtype::kInt8; }
  return dataset_dtype::kUnknown;
}

auto read_dataset_rows_from_json(const std::string& conf_path) -> std::optional<uint32_t>
{
  std::ifstream conf_json_stream(conf_path);
  if (!conf_json_stream) { return std::nullopt; }

  auto conf_json = nlohmann::json::parse(conf_json_stream, nullptr, false);
  if (conf_json.is_discarded() || !conf_json.contains("dataset") ||
      !conf_json.at("dataset").is_object()) {
    return std::nullopt;
  }

  const auto& dataset_json = conf_json.at("dataset");
  if (!dataset_json.contains("rows")) { return std::nullopt; }

  if (dataset_json.at("rows").is_number_unsigned()) {
    auto rows_u64 = dataset_json.at("rows").get<uint64_t>();
    if (rows_u64 <= static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      return static_cast<uint32_t>(rows_u64);
    }
    return std::nullopt;
  }

  if (dataset_json.at("rows").is_number_integer()) {
    auto rows_i64 = dataset_json.at("rows").get<int64_t>();
    if (rows_i64 > 0 && rows_i64 <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
      return static_cast<uint32_t>(rows_i64);
    }
  }

  return std::nullopt;
}

template <typename T>
auto run_workload(const cuvs::bench::configuration::dataset_conf& dataset_conf,
                  const std::string& conf_path,
                  const std::string& data_prefix,
                  const std::string& index_prefix,
                  uint32_t recall_at_k,
                  bool skip_build) -> int
{
  cuvs::bench::blob<T> base_blob(
    dataset_conf.base_file, dataset_conf.subset_first_row, dataset_conf.subset_size);

  BuildParams build_params;
  auto rows_from_config      = read_dataset_rows_from_json(conf_path);
  build_params.dataset_rows  = rows_from_config.value_or(static_cast<uint32_t>(base_blob.n_rows()));
  build_params.dataset_dim      = base_blob.n_cols();
  build_params.distance_measure = dataset_conf.distance;

  FreshBANG<T> freshbang;

  std::cout << "[freshbang_driver] SetDatasetParams(rows=" << build_params.dataset_rows
            << ", dim=" << build_params.dataset_dim
            << ", distance=" << build_params.distance_measure << ")" << std::endl;
  if (!freshbang.SetDatasetParams(build_params)) {
    std::cerr << "[freshbang_driver] SetDatasetParams failed" << std::endl;
    return 1;
  }

  std::cout << "[freshbang_driver] CreateAlgo(conf_path=" << conf_path << ")" << std::endl;
  if (!freshbang.CreateAlgo(conf_path)) {
    std::cerr << "[freshbang_driver] CreateAlgo failed" << std::endl;
    return 1;
  }

  if (!skip_build) {
    std::cout << "[freshbang_driver] BuildIndex(rows=1000 out of " << base_blob.n_rows()
              << ", cols=" << base_blob.n_cols() << ")" << std::endl;
    //if (!freshbang.BuildIndex(base_blob.data(), base_blob.n_rows())) {
 if (!freshbang.BuildIndex(base_blob.data(), 1000)) {
      std::cerr << "[freshbang_driver] BuildIndex failed" << std::endl;
      return 1;
    }
    std::cout << "[freshbang_driver] Build pipeline completed successfully" << std::endl;
  } else {
    std::cout << "[freshbang_driver] --skipbuild enabled: skipping BuildIndex and testing"
                 " insert-as-build path"
              << std::endl;
  }

  std::ifstream conf_stream(conf_path);
  cuvs::bench::ensure_stream_open(conf_stream, conf_path);
  auto& conf = cuvs::bench::configuration::initialize(conf_stream, data_prefix, index_prefix);
  auto dataset = cuvs::bench::make_dataset<T>(conf.get_dataset_conf(), true);

  // search query had to be on device memory for freshbang since it doesn't support host memory input 
  //Not the case anymore.
  const T* query_set = dataset->query_set(cuvs::bench::MemoryType::kDevice);
  const uint32_t batch_size =
    std::min<uint32_t>(static_cast<uint32_t>(dataset->query_set_size()), 10000U);

  SearchParams search_params{};
  search_params.recall_at_k = recall_at_k;

  const auto result_count = static_cast<std::size_t>(batch_size) * search_params.recall_at_k;
  std::vector<uint64_t> ids(result_count, 0ULL);
  std::vector<uint32_t> neighbors(result_count, 0U);
  std::vector<float> distances(result_count, 0.0F);
  uint64_t next_insert_id = 0;
  uint64_t min_insert_id  = 0;
  const std::string results_file = "search_results_ids.txt";

  {
    std::ofstream clear_stream(results_file, std::ios::trunc);
    if (!clear_stream) {
      std::cerr << "[freshbang_driver] Failed to create results file: " << results_file
                << std::endl;
      return 1;
    }
  }

  if (!skip_build) {
    next_insert_id = static_cast<uint64_t>(base_blob.n_rows());
    min_insert_id  = next_insert_id;
    if (next_insert_id > 0) {
      std::cout << "[freshbang_driver] Build consumed base IDs [0.." << (next_insert_id - 1)
                << "]; insert IDs will start at " << next_insert_id << std::endl;
    }
  }

  auto run_search_phase = [&](const char* phase_label) -> double {
    std::cout << "[freshbang_driver] " << phase_label << " (batch_size=" << batch_size
              << ", k=" << search_params.recall_at_k << ")" << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    freshbang.BatchedSearch(query_set, batch_size, ids.data(), neighbors.data(), distances.data());

    auto t1      = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(t1 - t0).count();
    auto qps     = elapsed > 0.0 ? static_cast<double>(batch_size) / elapsed : 0.0;
    std::cout << "[freshbang_driver] " << phase_label << " elapsed_sec=" << elapsed
              << " qps=" << qps << std::endl;
    return elapsed;
  };

  auto dump_ids_to_file = [&](uint32_t query_count) -> bool {
    std::ofstream results_stream(results_file, std::ios::app);
    if (!results_stream) {
      std::cerr << "[freshbang_driver] Failed to open results file for append: "
                << results_file << std::endl;
      return false;
    }

    for (uint32_t query_idx = 0; query_idx < query_count; ++query_idx) {
      const std::size_t row_offset = static_cast<std::size_t>(query_idx) * search_params.recall_at_k;
      for (uint32_t rank = 0; rank < search_params.recall_at_k; ++rank) {
        if (rank > 0) { results_stream << ","; }
        results_stream << ids[row_offset + rank];
      }
      results_stream << '\n';
    }

    return true;
  };

  auto run_insert_phase = [&](std::vector<uint64_t>& inserted_ids_out) -> bool {
    std::cout << "[freshbang_driver] SetInsertParams()" << std::endl;
    if (!freshbang.SetInsertParams()) {
      std::cerr << "[freshbang_driver] SetInsertParams failed" << std::endl;
      return false;
    }

    std::ifstream conf_json_stream(conf_path);
    if (!conf_json_stream) {
      std::cerr << "[freshbang_driver] Cannot open configuration file for insert phase: "
                << conf_path << std::endl;
      return false;
    }
    auto conf_json = nlohmann::json::parse(conf_json_stream);
    if (!conf_json.contains("dataset") || !conf_json.at("dataset").is_object()) {
      std::cerr << "[freshbang_driver] Missing dataset section in config for insert phase"
                << std::endl;
      return false;
    }
    const auto& dataset_json = conf_json.at("dataset");

    if (!dataset_json.contains("insert_file") || !dataset_json.at("insert_file").is_string() ||
        dataset_json.at("insert_file").get<std::string>().empty()) {
      std::cerr << "[freshbang_driver] dataset.insert_file is required in config for insert phase"
                << std::endl;
      return false;
    }

    std::string insert_rel          = dataset_json.at("insert_file").get<std::string>();
    std::string insert_vectors_file = cuvs::bench::combine_path(data_prefix, insert_rel);

    std::cout << "[freshbang_driver] Loading insert vectors from: " << insert_vectors_file
              << std::endl;
    cuvs::bench::blob<T> insert_blob{insert_vectors_file};
    const T* insert_vectors            = insert_blob.data();
    const uint32_t insert_batch_size = static_cast<uint32_t>(insert_blob.n_rows());
    if (insert_batch_size == 0) {
      std::cout << "[freshbang_driver] insert_file is empty; skipping BatchedInsert" << std::endl;
      return true;
    }

    // Keep a running, zero-based insert id sequence for all insert flows.
    // Even though the current cagra insert implementation ignores ids, future
    // implementations are expected to consume them.
    std::vector<uint64_t> insert_ids;
    insert_ids.resize(insert_batch_size);
    for (uint32_t i = 0; i < insert_batch_size; ++i) {
      insert_ids[i] = next_insert_id + static_cast<uint64_t>(i);
    }
    const uint64_t* insert_ids_ptr = insert_ids.data();
    std::cout << "[freshbang_driver] Insert IDs range: [" << next_insert_id << ", "
              << (next_insert_id + static_cast<uint64_t>(insert_batch_size) - 1) << "]"
              << std::endl;

    std::cout << "[freshbang_driver] BatchedInsert(batch_size=" << insert_batch_size << ")"
              << std::endl;

    if (skip_build) {
      const auto& indices = conf.get_indices();
      if (!indices.empty()) {
        const auto& index_file = indices.front().file;
        if (std::filesystem::exists(index_file)) {
          std::cout << "[freshbang_driver] --skipbuild removing existing index file: "
                    << index_file << std::endl;
          std::filesystem::remove(index_file);
        }
      }
    }

    auto t0 = std::chrono::steady_clock::now();
    freshbang.BatchedInsert(insert_vectors, insert_batch_size, insert_ids_ptr);
    auto t1      = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(t1 - t0).count();
    auto qps     = elapsed > 0.0 ? static_cast<double>(insert_batch_size) / elapsed : 0.0;
    std::cout << "[freshbang_driver] BatchedInsert elapsed_sec=" << elapsed
              << " qps=" << qps << std::endl;

    inserted_ids_out = insert_ids;
    next_insert_id += static_cast<uint64_t>(insert_batch_size);
    return true;
  };

  auto run_delete_phase = [&](const std::vector<uint64_t>& ids_to_delete) -> bool {
    if (ids_to_delete.empty()) {
      std::cout << "[freshbang_driver] No inserted IDs; skipping BatchedDelete" << std::endl;
      return true;
    }

    std::cout << "[freshbang_driver] BatchedDelete(batch_size=" << ids_to_delete.size()
              << ") using inserted IDs [" << ids_to_delete.front() << ".."
              << ids_to_delete.back() << "]" << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    freshbang.BatchedDelete(ids_to_delete.data(), static_cast<uint32_t>(ids_to_delete.size()));
    auto t1      = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(t1 - t0).count();
    auto qps     = elapsed > 0.0 ? static_cast<double>(ids_to_delete.size()) / elapsed : 0.0;
    std::cout << "[freshbang_driver] BatchedDelete elapsed_sec=" << elapsed
              << " qps=" << qps << std::endl;

    // Keep insert ID allocation stable across insert-delete cycles.
    auto deleted_count = static_cast<uint64_t>(ids_to_delete.size());
    if (next_insert_id >= deleted_count) {
      next_insert_id -= deleted_count;
    } else {
      next_insert_id = min_insert_id;
    }
    if (next_insert_id < min_insert_id) { next_insert_id = min_insert_id; }

    return true;
  };

  if (!skip_build) {
    std::cout << "[freshbang_driver] SetSearchParams(recall_at_k=" << search_params.recall_at_k
              << ")" << std::endl;
    if (!freshbang.SetSearchParams(search_params)) {
      std::cerr << "[freshbang_driver] SetSearchParams failed" << std::endl;
      return 1;
    }

    // Note: query_set can be on host or device. CAGRA wrapper will handle copying if needed.
    run_search_phase("BatchedSearch");
    if (!dump_ids_to_file(batch_size)) {
      return 1;
    }
    std::cout << "[freshbang_driver] Search pipeline completed" << std::endl;


  }

  constexpr int kPipelineRuns = PIPELINE_RUNS;
  for (int run = 1; run <= kPipelineRuns; ++run) {
    std::cout << "[freshbang_driver] ===== Pipeline Run " << run << "/" << kPipelineRuns
              << " =====" << std::endl;



    std::vector<uint64_t> inserted_ids_for_delete;
    if (!run_insert_phase(inserted_ids_for_delete)) {
      return 1;
    }

    if (skip_build && run == 1) {
      std::cout << "[freshbang_driver] SetSearchParams(recall_at_k=" << search_params.recall_at_k
                << ") after first insert" << std::endl;
      if (!freshbang.SetSearchParams(search_params)) {
        std::cerr << "[freshbang_driver] SetSearchParams failed" << std::endl;
        return 1;
      }
    }

    std::fill(ids.begin(), ids.end(), 0ULL);
    std::fill(neighbors.begin(), neighbors.end(), 0U);
    std::fill(distances.begin(), distances.end(), 0.0F);
    run_search_phase("BatchedSearch after insert");
    if (!dump_ids_to_file(batch_size)) {
      return 1;
    }
    std::cout << "[freshbang_driver] Post-insert search pipeline completed" << std::endl;

    if (!run_delete_phase(inserted_ids_for_delete)) {
      return 1;
    }

    std::fill(ids.begin(), ids.end(), 0ULL);
    std::fill(neighbors.begin(), neighbors.end(), 0U);
    std::fill(distances.begin(), distances.end(), 0.0F);
    run_search_phase("BatchedSearch after delete");
    if (!dump_ids_to_file(batch_size)) {
      return 1;
    }
    std::cout << "[freshbang_driver] Post-delete search pipeline completed" << std::endl;
  } 

  // final cleanup
  //freshbang.Cleanup();
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " <conf_path> <recall_at_k> [--skipbuild]" << std::endl;
    std::cerr << "\nOptional flags:" << std::endl;
    std::cerr << "  --skipbuild     Skip BuildIndex and call BatchedInsert first." << std::endl;
    std::cerr << "                  This is intended to test the special-case path where" << std::endl;
    std::cerr << "                  BatchedInsert behaves like build/save when no index exists." 
              << std::endl;
    std::cerr << "\nDefault phases: SetDatasetParams → CreateAlgo → BuildIndex"
                 " → SetSearchParams → BatchedSearch → SetInsertParams"
                 " → BatchedInsert → BatchedSearch"
              << std::endl;
    std::cerr << "Skip-build phases (--skipbuild): SetDatasetParams → CreateAlgo"
                 " → SetInsertParams → BatchedInsert → SetSearchParams"
                 " → BatchedSearch"
              << std::endl;
    std::cerr << "\nRequired Parameters:" << std::endl;
    std::cerr << "  conf_path       - Path to JSON configuration file" << std::endl;
    std::cerr << "  recall_at_k     - Search K to pass into SetSearchParams" << std::endl;
    std::cerr << "\nConfiguration:" << std::endl;
    std::cerr << "  Data and index prefixes are read from 'freshbang.cfg' in the current "
                 "directory." << std::endl;
    std::cerr << "  Required keys in freshbang.cfg:" << std::endl;
    std::cerr << "    data_prefix     - Prefix prepended to dataset paths in config" << std::endl;
    std::cerr << "    index_prefix    - Prefix prepended to index file paths in config" << std::endl;
    std::cerr << "\nExample freshbang.cfg:" << std::endl;
    std::cerr << "  data_prefix=/path/to/datasets/" << std::endl;
    std::cerr << "  index_prefix=/path/to/indices/" << std::endl;
    std::cerr << "\nSupported Data Types (inferred from base_file extension):" << std::endl;
    std::cerr << "  .fbin           - float32 (✓ supported)" << std::endl;
    std::cerr << "  .ibin           - int32 (requires FreshBANG<int32_t> instantiation)" << std::endl;
    std::cerr << "  .u8bin          - uint8 (requires FreshBANG<uint8_t> instantiation)" << std::endl;
    std::cerr << "  .i8bin          - int8 (requires FreshBANG<int8_t> instantiation)" << std::endl;
    std::cerr << "\nNotes:" << std::endl;
    std::cerr << "  - Configuration is loaded from JSON; index_name='', index_pos=0" << std::endl;
    std::cerr << "  - Base vectors are loaded from dataset_conf.base_file" << std::endl;
    std::cerr << "  - Insert vectors are loaded from dataset.insert_file" << std::endl;
    std::cerr << "  - Output index written to dataset_conf.file with index_prefix" << std::endl;
    std::cerr << "  - With --skipbuild, the initial BatchedSearch before insert is skipped" << std::endl;
    return 2;
  }

  const std::string conf_path = argv[1];
  const bool skip_build       = (argc == 4);

  if (skip_build && std::string(argv[3]) != "--skipbuild") {
    std::cerr << "[freshbang_driver] Unknown argument: '" << argv[3]
              << "'. Did you mean --skipbuild ?" << std::endl;
    return 2;
  }

  uint32_t recall_at_k = 0;

  try {
    recall_at_k = static_cast<uint32_t>(std::stoul(argv[2]));
  } catch (const std::exception&) {
    std::cerr << "[freshbang_driver] Invalid recall_at_k: '" << argv[2]
              << "'. It must be a positive integer." << std::endl;
    return 2;
  }
  if (recall_at_k == 0) {
    std::cerr << "[freshbang_driver] recall_at_k must be greater than zero." << std::endl;
    return 2;
  }

  // Enable debug logging for RAFT without taking a rapids_logger link dependency.
  setenv("RAFT_LOG_LEVEL", "debug", 1);
  setenv("RAPIDS_LOG_LEVEL", "debug", 1);
  setenv("RAPIDS_LOGGER_LOG_LEVEL", "debug", 1);
  setenv("SPDLOG_LEVEL", "debug", 1);

  // Read data_prefix and index_prefix from freshbang.cfg
  auto [config_ok, driver_config] = read_freshbang_config();
  if (!config_ok) {
    std::cerr << "[freshbang_driver] Failed to read freshbang.cfg" << std::endl;
    return 1;
  }

  const std::string data_prefix  = driver_config.data_prefix;
  const std::string index_prefix = driver_config.index_prefix;

  std::cout << "[freshbang_driver] Config loaded:" << std::endl
            << "  conf_path=" << conf_path << std::endl
            << "  data_prefix=" << data_prefix << std::endl
            << "  index_prefix=" << index_prefix << std::endl
            << "  recall_at_k=" << recall_at_k << std::endl
            << "  skip_build=" << (skip_build ? "true" : "false") << std::endl;

  try {
    auto data_prefix_norm  = cuvs::bench::detail::normalize_prefix(data_prefix, "data");
    auto index_prefix_norm = cuvs::bench::detail::normalize_prefix(index_prefix, "index");

    std::ifstream conf_stream(conf_path);
    cuvs::bench::ensure_stream_open(conf_stream, conf_path);

    auto& conf =
      cuvs::bench::configuration::initialize(conf_stream, data_prefix_norm, index_prefix_norm);
    const auto& dataset_conf = conf.get_dataset_conf();

    auto dtype = infer_dtype_from_base_file(dataset_conf.base_file);
    if (dtype == dataset_dtype::kUnknown) {
      std::cerr << "[freshbang_driver] Could not infer data type from base_file: "
                << dataset_conf.base_file << std::endl;
      return 1;
    }

    // Current library instantiation guarantees float support.
    if (dtype == dataset_dtype::kFloat32) {
      return run_workload<float>(
        dataset_conf, conf_path, data_prefix_norm, index_prefix_norm, recall_at_k, skip_build);
    }

    if (dtype == dataset_dtype::kInt32) {
      std::cerr << "[freshbang_driver] base_file indicates int32 (.ibin), but FreshBANG<int32> "
                   "is not instantiated in the current library."
                << std::endl;
      return 1;
    }
    if (dtype == dataset_dtype::kUInt8) {
      std::cerr << "[freshbang_driver] base_file indicates uint8 (.u8bin), but FreshBANG<uint8_t> "
                   "is not instantiated in the current library."
                << std::endl;
      return 1;
    }

    std::cerr << "[freshbang_driver] base_file indicates int8 (.i8bin), but FreshBANG<int8_t> "
                 "is not instantiated in the current library."
              << std::endl;
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "[freshbang_driver] Fatal: " << e.what() << std::endl;
    return 1;
  }
}