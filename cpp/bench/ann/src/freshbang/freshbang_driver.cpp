
#include "../common/blob.hpp"
#include "../common/conf.hpp"
#include "freshbang.hpp"
#include "freshbang_helpers.hpp"

#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace {

enum class dataset_dtype { kFloat32, kInt32, kUInt8, kInt8, kUnknown };
enum class pipeline_op_type { kInsert, kDelete };

struct pipeline_op {
  pipeline_op_type type;
  uint64_t first_id;
  uint64_t last_id;
};

struct DriverConfig {
  std::string data_prefix;
  std::string index_prefix;
  std::vector<pipeline_op> operations;
};

auto trim_copy(std::string s) -> std::string
{
  const auto first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) { return ""; }
  const auto last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

auto parse_id_range(const std::string& value, uint64_t& first_id, uint64_t& last_id) -> bool
{
  const auto comma_pos = value.find(',');
  if (comma_pos == std::string::npos) { return false; }

  auto first_str = trim_copy(value.substr(0, comma_pos));
  auto last_str  = trim_copy(value.substr(comma_pos + 1));
  if (first_str.empty() || last_str.empty()) { return false; }

  try {
    first_id = std::stoull(first_str);
    last_id  = std::stoull(last_str);
  } catch (const std::exception&) {
    return false;
  }

  return first_id <= last_id;
}

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
  uint32_t line_number = 0;
  while (std::getline(conf_stream, line)) {
    ++line_number;
    line = trim_copy(line);

    // Skip comments and empty lines
    if (line.empty() || line[0] == '#') { continue; }

    const auto comment_pos = line.find('#');
    if (comment_pos != std::string::npos) {
      line = trim_copy(line.substr(0, comment_pos));
      if (line.empty()) { continue; }
    }

    size_t delimiter_pos = line.find('=');
    if (delimiter_pos == std::string::npos) { continue; }

    std::string key   = trim_copy(line.substr(0, delimiter_pos));
    std::string value = trim_copy(line.substr(delimiter_pos + 1));

    if (key == "data_prefix") {
      config.data_prefix = value;
    } else if (key == "index_prefix") {
      config.index_prefix = value;
    } else if (key == "insert" || key == "delete") {
      uint64_t first_id = 0;
      uint64_t last_id  = 0;
      if (!parse_id_range(value, first_id, last_id)) {
        std::cerr << "[freshbang_driver] Error: invalid range for '" << key << "' at line "
                  << line_number << ". Expected format: " << key << "=start,end" << std::endl;
        return {false, config};
      }
      config.operations.push_back({key == "insert" ? pipeline_op_type::kInsert
                                                    : pipeline_op_type::kDelete,
                                   first_id,
                                   last_id});
    } else {
      std::cerr << "[freshbang_driver] Warning: ignoring unknown key '" << key
                << "' at line " << line_number << std::endl;
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
                  const std::vector<pipeline_op>& pipeline_ops) -> int
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

  std::cout << "[freshbang_driver] BuildIndex(rows=1000 out of " << base_blob.n_rows()
            << ", cols=" << base_blob.n_cols() << ")" << std::endl;
  if (!freshbang.BuildIndex(base_blob.data(), base_blob.n_rows())) {
    std::cerr << "[freshbang_driver] BuildIndex failed" << std::endl;
    return 1;
  }
  // Save the index after build.
  freshbang.SaveIndex();
  std::cout << "[freshbang_driver] Build pipeline completed successfully" << std::endl;

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
  const std::string results_file = "search_results_ids.txt";

  {
    std::ofstream clear_stream(results_file, std::ios::trunc);
    if (!clear_stream) {
      std::cerr << "[freshbang_driver] Failed to create results file: " << results_file
                << std::endl;
      return 1;
    }
  }

  auto validate_id_range = [&](const char* phase_label,
                               uint64_t first_id,
                               uint64_t count) -> bool {
    if (count == 0) { return true; }

    if (build_params.dataset_rows == 0) {
      std::cerr << "[freshbang_driver] " << phase_label
                << " aborted: dataset_rows is 0, so no valid IDs are available." << std::endl;
      return false;
    }

    auto max_valid_id = static_cast<uint64_t>(build_params.dataset_rows) - 1;
    auto last_id      = first_id + count - 1;
    if (first_id > max_valid_id || last_id > max_valid_id) {
      std::cerr << "[freshbang_driver] " << phase_label << " aborted: requested IDs ["
                << first_id << ", " << last_id << "] exceed valid range [0, "
                << max_valid_id << "] derived from dataset_rows="
                << build_params.dataset_rows << std::endl;
      return false;
    }

    return true;
  };

  auto validate_ids_within_bounds = [&](const char* phase_label,
                                        const std::vector<uint64_t>& ids_to_check) -> bool {
    if (ids_to_check.empty()) { return true; }

    auto [min_it, max_it] = std::minmax_element(ids_to_check.begin(), ids_to_check.end());
    if (*min_it > *max_it) { return true; }

    return validate_id_range(phase_label, *min_it, (*max_it - *min_it) + 1);
  };

  auto has_insert_op = std::any_of(pipeline_ops.begin(), pipeline_ops.end(), [](const pipeline_op& op) {
    return op.type == pipeline_op_type::kInsert;
  });

  std::optional<cuvs::bench::blob<T>> insert_blob;
  const T* insert_vectors_all     = nullptr;
  uint64_t insert_vectors_total   = 0;
  uint64_t insert_vectors_offset  = 0;

  if (has_insert_op) {
    std::ifstream conf_json_stream(conf_path);
    if (!conf_json_stream) {
      std::cerr << "[freshbang_driver] Cannot open configuration file for insert phase: "
                << conf_path << std::endl;
      return 1;
    }
    auto conf_json = nlohmann::json::parse(conf_json_stream);
    if (!conf_json.contains("dataset") || !conf_json.at("dataset").is_object()) {
      std::cerr << "[freshbang_driver] Missing dataset section in config for insert phase"
                << std::endl;
      return 1;
    }
    const auto& dataset_json = conf_json.at("dataset");
    if (!dataset_json.contains("insert_file") || !dataset_json.at("insert_file").is_string() ||
        dataset_json.at("insert_file").get<std::string>().empty()) {
      std::cerr << "[freshbang_driver] dataset.insert_file is required in config for insert phase"
                << std::endl;
      return 1;
    }

    std::string insert_rel          = dataset_json.at("insert_file").get<std::string>();
    std::string insert_vectors_file = cuvs::bench::combine_path(data_prefix, insert_rel);
    std::cout << "[freshbang_driver] Loading insert vectors from: " << insert_vectors_file
              << std::endl;

    insert_blob.emplace(insert_vectors_file);
    insert_vectors_all   = insert_blob->data();
    insert_vectors_total = static_cast<uint64_t>(insert_blob->n_rows());
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

  auto run_insert_phase = [&](uint64_t first_id, uint64_t last_id) -> bool {
    std::cout << "[freshbang_driver] SetInsertParams()" << std::endl;
    if (!freshbang.SetInsertParams()) {
      std::cerr << "[freshbang_driver] SetInsertParams failed" << std::endl;
      return false;
    }

    auto insert_batch_size_u64 = last_id - first_id + 1;
    if (insert_batch_size_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      std::cerr << "[freshbang_driver] BatchedInsert aborted: batch size "
                << insert_batch_size_u64 << " exceeds uint32_t limit" << std::endl;
      return false;
    }
    auto insert_batch_size = static_cast<uint32_t>(insert_batch_size_u64);

    if (!validate_id_range("BatchedInsert", first_id, insert_batch_size_u64)) {
      return false;
    }

    if (insert_vectors_offset + insert_batch_size_u64 > insert_vectors_total) {
      std::cerr << "[freshbang_driver] BatchedInsert aborted: operation requires "
                << insert_batch_size_u64 << " vectors but only "
                << (insert_vectors_total - insert_vectors_offset)
                << " unused vectors remain in dataset.insert_file" << std::endl;
      return false;
    }

    const T* insert_vectors =
      insert_vectors_all + (insert_vectors_offset * static_cast<uint64_t>(build_params.dataset_dim));
    if (insert_batch_size == 0) {
      std::cout << "[freshbang_driver] insert_file is empty; skipping BatchedInsert" << std::endl;
      return true;
    }

    // Use IDs exactly as configured in freshbang.cfg.
    std::vector<uint64_t> insert_ids;
    insert_ids.resize(insert_batch_size);
    for (uint32_t i = 0; i < insert_batch_size; ++i) {
      insert_ids[i] = first_id + static_cast<uint64_t>(i);
    }

    const uint64_t* insert_ids_ptr = insert_ids.data();
    std::cout << "[freshbang_driver] Insert IDs range: [" << first_id << ", " << last_id
              << "]" << std::endl;

    std::cout << "[freshbang_driver] BatchedInsert(batch_size=" << insert_batch_size << ")"
              << std::endl;

    auto t0 = std::chrono::steady_clock::now();
    freshbang.BatchedInsert(insert_vectors, insert_batch_size, insert_ids_ptr);
    auto t1      = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration<double>(t1 - t0).count();
    auto qps     = elapsed > 0.0 ? static_cast<double>(insert_batch_size) / elapsed : 0.0;
    std::cout << "[freshbang_driver] BatchedInsert elapsed_sec=" << elapsed
              << " qps=" << qps << std::endl;

    insert_vectors_offset += insert_batch_size_u64;
    return true;
  };

  auto run_delete_phase = [&](uint64_t first_id, uint64_t last_id) -> bool {
    auto delete_batch_size_u64 = last_id - first_id + 1;
    if (delete_batch_size_u64 > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
      std::cerr << "[freshbang_driver] BatchedDelete aborted: batch size "
                << delete_batch_size_u64 << " exceeds uint32_t limit" << std::endl;
      return false;
    }

    std::vector<uint64_t> ids_to_delete(static_cast<size_t>(delete_batch_size_u64));
    for (uint64_t i = 0; i < delete_batch_size_u64; ++i) {
      ids_to_delete[static_cast<size_t>(i)] = first_id + i;
    }

    if (!validate_ids_within_bounds("BatchedDelete", ids_to_delete)) { return false; }

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

    return true;
  };

  std::cout << "[freshbang_driver] SetSearchParams(recall_at_k=" << search_params.recall_at_k
            << ")" << std::endl;
  if (!freshbang.SetSearchParams(search_params)) {
    std::cerr << "[freshbang_driver] SetSearchParams failed" << std::endl;
    return 1;
  }

  if (pipeline_ops.empty()) {
    std::cout << "[freshbang_driver] No insert/delete operations in freshbang.cfg; stopping "
                 "after initial search"
              << std::endl;
  }

  for (size_t op_index = 0; op_index < pipeline_ops.size(); ++op_index) {
    const auto& op = pipeline_ops[op_index];
    const char* op_name = (op.type == pipeline_op_type::kInsert) ? "insert" : "delete";
    std::cout << "[freshbang_driver] ===== Pipeline Op " << (op_index + 1) << "/"
              << pipeline_ops.size() << ": " << op_name << " [" << op.first_id << ", "
              << op.last_id << "] =====" << std::endl;

    if (op.type == pipeline_op_type::kInsert) {
      if (!run_insert_phase(op.first_id, op.last_id)) { return 1; }
    } else {
      if (!run_delete_phase(op.first_id, op.last_id)) { return 1; }
    }
  }

  std::fill(ids.begin(), ids.end(), 0ULL);
  std::fill(neighbors.begin(), neighbors.end(), 0U);
  std::fill(distances.begin(), distances.end(), 0.0F);
  // Run search 3 times after all configured insert/delete operations complete.
  for (int i = 0; i < 3; ++i) {
    run_search_phase((std::string("BatchedSearch final") + std::to_string(i + 1)).c_str());
  }
  if (!dump_ids_to_file(batch_size)) {
    return 1;
  }
  std::cout << "[freshbang_driver] Final search pipeline completed" << std::endl;
  freshbang.SaveIndex();
  // final cleanup
  //freshbang.Cleanup();
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <conf_path> <recall_at_k>" << std::endl;
    std::cerr << "\nDefault phases: SetDatasetParams → CreateAlgo → BuildIndex"
                 " → SetSearchParams → BatchedSearch → SetInsertParams"
                 " → BatchedInsert → BatchedSearch"
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
    std::cerr << "  Optional repeated operation lines:" << std::endl;
    std::cerr << "    insert=start,end" << std::endl;
    std::cerr << "    delete=start,end" << std::endl;
    std::cerr << "  Lines starting with '#' are treated as comments and ignored." << std::endl;
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
    std::cerr << "  - Operations execute in listed order from freshbang.cfg" << std::endl;
    return 2;
  }

  const std::string conf_path = argv[1];

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
            << "  operations=" << driver_config.operations.size() << std::endl;

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
        dataset_conf,
        conf_path,
        data_prefix_norm,
        index_prefix_norm,
        recall_at_k,
        driver_config.operations);
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