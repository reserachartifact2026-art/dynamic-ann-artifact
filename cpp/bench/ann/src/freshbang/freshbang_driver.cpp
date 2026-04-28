
#include "../common/blob.hpp"
#include "../common/conf.hpp"
#include "freshbang.hpp"
#include "freshbang_helpers.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace {

enum class dataset_dtype { kFloat32, kInt32, kUInt8, kInt8, kUnknown };

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

template <typename T>
auto run_build_pipeline(const cuvs::bench::configuration::dataset_conf& dataset_conf,
                        const std::string& conf_path,
                        const std::string& data_prefix,
                        const std::string& index_prefix,
                        uint32_t recall_at_k,
                        bool skip_build) -> int
{
  cuvs::bench::blob<T> base_blob(
    dataset_conf.base_file, dataset_conf.subset_first_row, dataset_conf.subset_size);

  BuildParams build_params;
  build_params.dataset_dim      = base_blob.n_cols();
  build_params.distance_measure = dataset_conf.distance;

  FreshBANG<T> freshbang;

  std::cout << "[freshbang_driver] SetDatasetParams(dim=" << build_params.dataset_dim
            << ", distance=" << build_params.distance_measure << ")" << std::endl;
  if (!freshbang.SetDatasetParams(build_params)) {
    std::cerr << "[freshbang_driver] SetDatasetParams failed" << std::endl;
    return 1;
  }

  std::cout << "[freshbang_driver] CreateAlgo()" << std::endl;
  if (!freshbang.CreateAlgo()) {
    std::cerr << "[freshbang_driver] CreateAlgo failed" << std::endl;
    return 1;
  }

  if (!skip_build) {
    std::cout << "[freshbang_driver] BuildIndex(rows=" << base_blob.n_rows()
              << ", cols=" << base_blob.n_cols() << ")" << std::endl;
    if (!freshbang.BuildIndex(base_blob.data(), base_blob.n_rows())) {
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

  //const T* query_set = dataset->query_set(cuvs::bench::MemoryType::kDevice);
  const T* query_set = dataset->query_set();
  const uint32_t batch_size =
    std::min<uint32_t>(static_cast<uint32_t>(dataset->query_set_size()), 10000U);

  SearchParams search_params{};
  search_params.recall_at_k = recall_at_k;

  const auto result_count = static_cast<std::size_t>(batch_size) * search_params.recall_at_k;
  std::vector<uint64_t> ids(result_count, 0ULL);
  std::vector<uint32_t> neighbors(result_count, 0U);
  std::vector<float> distances(result_count, 0.0F);

  auto run_search_phase = [&](const char* phase_label) {
    std::cout << "[freshbang_driver] " << phase_label << " (batch_size=" << batch_size
              << ", k=" << search_params.recall_at_k << ")" << std::endl;
    freshbang.BatchedSearch(query_set, batch_size, ids.data(), neighbors.data(), distances.data());
  };

  auto run_insert_phase = [&]() -> bool {
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

    std::cout << "[freshbang_driver] BatchedInsert(batch_size=" << insert_batch_size << ")"
              << std::endl;
    freshbang.BatchedInsert(insert_vectors, insert_batch_size, nullptr);
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
    std::cout << "[freshbang_driver] Search pipeline completed" << std::endl;
  }

  if (!run_insert_phase()) {
    return 1;
  }

  if (skip_build) {
    std::cout << "[freshbang_driver] SetSearchParams(recall_at_k=" << search_params.recall_at_k
              << ") after insert" << std::endl;
    if (!freshbang.SetSearchParams(search_params)) {
      std::cerr << "[freshbang_driver] SetSearchParams failed" << std::endl;
      return 1;
    }
  }

  std::fill(ids.begin(), ids.end(), 0ULL);
  std::fill(neighbors.begin(), neighbors.end(), 0U);
  std::fill(distances.begin(), distances.end(), 0.0F);

  run_search_phase("BatchedSearch after insert");
  std::cout << "[freshbang_driver] Post-insert search pipeline completed" << std::endl;

  // final cleanup
  freshbang.Cleanup();
  return 0;
}

}  // namespace

int main(int argc, char** argv)
{
  if (argc != 5 && argc != 6) {
    std::cerr << "Usage: " << argv[0]
              << " <conf_path> <dataset_prefix> <index_prefix> <recall_at_k> [--skipbuild]"
              << std::endl;
    std::cerr << "\nOptional flags:" << std::endl;
    std::cerr << "  --skipbuild     Skip BuildIndex and call BatchedInsert first." << std::endl;
    std::cerr << "                  This is intended to test the special-case path where" << std::endl;
    std::cerr << "                  BatchedInsert behaves like build/save when no index exists." << std::endl;
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
    std::cerr << "  dataset_prefix  - Prefix prepended to base_file paths in config" << std::endl;
    std::cerr << "  index_prefix    - Prefix prepended to index file paths in config" << std::endl;
    std::cerr << "  recall_at_k     - Search K to pass into SetSearchParams" << std::endl;
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

  const std::string conf_path    = argv[1];
  const std::string data_prefix  = argv[2];
  const std::string index_prefix = argv[3];
  const bool skip_build          = (argc == 6);

  if (skip_build && std::string(argv[5]) != "--skipbuild") {
    std::cerr << "[freshbang_driver] Unknown argument: '" << argv[5]
              << "'. Did you mean --skipbuild ?" << std::endl;
    return 2;
  }

  uint32_t recall_at_k           = 0;

  try {
    recall_at_k = static_cast<uint32_t>(std::stoul(argv[4]));
  } catch (const std::exception&) {
    std::cerr << "[freshbang_driver] Invalid recall_at_k: '" << argv[4]
              << "'. It must be a positive integer." << std::endl;
    return 2;
  }
  if (recall_at_k == 0) {
    std::cerr << "[freshbang_driver] recall_at_k must be greater than zero." << std::endl;
    return 2;
  }

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
      return run_build_pipeline<float>(
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