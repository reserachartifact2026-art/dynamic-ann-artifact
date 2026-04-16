/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0


g++ -std=c++17 -O2 -DBENCH_NS=raft::bench  -g -Icpp/bench/ann/src/common -Icpp/bench/ann/src -Icpp/include -Ic/include   -I$CONDA_PREFIX/targets/x86_64-linux/include   cpp/bench/ann/src/common/build_from_config_driver.cpp   -L$CONDA_PREFIX/lib -lcudart -ldl   -o build_from_config_driver
 */

#include "bench_api.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace ann = cuvs::bench;

int main(int argc, char** argv)
{
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <conf_path> <dataset_prefix> <index_prefix>"
              << std::endl;
    std::cerr << "\nPhases executed: BUILD → INSERT → SEARCH" << std::endl;
    std::cerr << "\nRequired Parameters:" << std::endl;
    std::cerr << "  conf_path       - Path to JSON configuration file" << std::endl;
    std::cerr << "  dataset_prefix  - Prefix prepended to base_file/query_file/insert_file" << std::endl;
    std::cerr << "  index_prefix    - Prefix prepended to index file path" << std::endl;
    std::cerr << "\nNotes:" << std::endl;
    std::cerr << "  - force_overwrite is hardcoded to true" << std::endl;
    std::cerr << "  - index_name='', index_pos=0, search_param_ix=0" << std::endl;
    std::cerr << "  - insert source: dataset.insert_file (required)" << std::endl;
    return 2;
  }

  const std::string conf_path    = argv[1];
  const std::string data_prefix  = argv[2];
  const std::string index_prefix = argv[3];

  const std::string index_name      = "";
  const std::size_t index_pos       = 0;
  const bool force_overwrite        = true;
  std::string insert_params_json = "{}";
  const std::size_t search_param_ix = 0;
//#define SKIP_BUILD  // define this to skip build phase and test insert/search only

#ifdef SKIP_BUILD
std::cout << "build_from_config_float SKIPPED" << std::endl;
#else
  try {
    ann::build_from_config_float(
      conf_path, index_name, index_pos, data_prefix, index_prefix, force_overwrite);
  } catch (const std::exception& e) {
    std::cerr << "build_from_config_float failed: " << e.what() << std::endl;
    return 1;
  }
  std::cout << "build_from_config_float completed" << std::endl;
#endif
  

  std::string insert_vectors_file;
  try {
    std::ifstream conf_json_stream(conf_path);
    if (!conf_json_stream) {
      throw std::runtime_error("Cannot open configuration file: " + conf_path);
    }
    auto conf_json = nlohmann::json::parse(conf_json_stream);
    const auto& dataset_json = conf_json.at("dataset");

    if (!dataset_json.contains("insert_file") || dataset_json.at("insert_file").get<std::string>().empty()) {
      throw std::runtime_error("dataset.insert_file is required in config for insert phase");
    }
    std::string insert_rel = dataset_json.at("insert_file").get<std::string>();

    insert_vectors_file = ann::combine_path(data_prefix, insert_rel);

    // Forward insert params to API so wrapper receives persist/refine/memory options.
    // Support both schema variants:
    //  - index[i].insert_params: [ {...}, ... ]
    //  - index[i].insert_param: { ... }
    if (conf_json.contains("index") && conf_json.at("index").is_array() &&
        conf_json.at("index").size() > index_pos) {
      const auto& idx_json = conf_json.at("index").at(index_pos);
      if (idx_json.contains("insert_params") && idx_json.at("insert_params").is_array() &&
          !idx_json.at("insert_params").empty()) {
        insert_params_json = idx_json.at("insert_params").at(0).dump();
      } else if (idx_json.contains("insert_param") && idx_json.at("insert_param").is_object()) {
        insert_params_json = idx_json.at("insert_param").dump();
      }
    }

    std::cout << "[INSERT] Insert params override: " << insert_params_json << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Failed to resolve insert vectors file from config: " << e.what() << std::endl;
    return 1;
  }

  // Phase 2: Insert
  std::cout << "\n[INSERT] Reading insert vectors from: " << insert_vectors_file << std::endl;
  try {
    auto insert_t0 = std::chrono::steady_clock::now();
    ann::insert_from_config_float(conf_path,
                                  index_name,
                                  index_pos,
                                  insert_vectors_file,
                                  data_prefix,
                                  index_prefix,
                                  insert_params_json);
    auto insert_t1 = std::chrono::steady_clock::now();
    auto insert_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(insert_t1 - insert_t0).count();
    std::cout << "[INSERT] Elapsed: " << insert_ms << " ms ("
              << static_cast<double>(insert_ms) / 1000.0 << " s)" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "insert_from_config_float failed: " << e.what() << std::endl;
    return 1;
  }
  std::cout << "[INSERT] ✓ Insert operation completed successfully" << std::endl;

  // Phase 3: Search
  std::cout << "\n[SEARCH] Starting search phase..." << std::endl;
  try {
    // Load queries from config file (same dataset as build)
    std::cout << "  - Loading queries from config..." << std::endl;
    std::ifstream conf_stream(conf_path);
    auto& conf   = ann::configuration::initialize(conf_stream, data_prefix, index_prefix);
    auto dataset = ann::make_dataset<float>(conf.get_dataset_conf(), true);
    
    const float* query_set         = dataset->query_set(ann::MemoryType::kDevice);
    std::size_t available_queries = dataset->query_set_size();
    std::size_t selected_queries = available_queries;
    std::size_t query_dim = dataset->dim();

    // Resolve output shape first so all output buffers are correctly sized.
    std::size_t out_queries = 0;
    std::size_t out_k       = 0;
    ann::get_search_shape_from_config_float(conf_path,
                        index_name,
                        index_pos,
                        search_param_ix,
                        selected_queries,
                        data_prefix,
                        index_prefix,
                        &out_queries,
                        &out_k);

    // Allocate output buffers.
    std::vector<ann::algo_base::index_type> neighbors(out_queries * out_k);
    std::vector<float> distances(out_queries * out_k);
    std::vector<int64_t> search_ids(out_queries * out_k, -1);

    std::cout << "  - Config: " << conf_path << std::endl;
    std::cout << "  - Index: " << index_name << " (pos=" << index_pos << ")" << std::endl;
    std::cout << "  - Search param index: " << search_param_ix << std::endl;
    std::cout << "  - Queries loaded: " << selected_queries << " x " << query_dim << std::endl;
    std::cout << "  - Precomputed output shape: (" << out_queries << " x " << out_k << ")"
          << std::endl;

    // Use search_from_config_array so it tries to use cached algo from build/insert
    // Run search 5 times and report per-run and average QPS.
    constexpr int kSearchRuns = 5;
    long long total_search_ms = 0;
    for (int run = 0; run < kSearchRuns; ++run) {
      auto search_t0 = std::chrono::steady_clock::now();
      ann::search_from_config_array_float(
          conf_path,
          index_name,
          index_pos,
          search_param_ix,
          query_set,
          selected_queries,
          query_dim,
          data_prefix,
          index_prefix,
          neighbors.data(),
          distances.data(),
          search_ids.data(),
          &out_queries,
          &out_k);
      auto search_t1 = std::chrono::steady_clock::now();
      auto search_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(search_t1 - search_t0).count();
      total_search_ms += search_ms;
      std::cout << "[SEARCH] Run " << (run + 1) << "/" << kSearchRuns
                << " Elapsed: " << search_ms << " ms";
      if (search_ms > 0) {
        auto qps = (static_cast<double>(out_queries) * 1000.0) / static_cast<double>(search_ms);
        std::cout << "  QPS: " << qps;
      }
      std::cout << std::endl;
    }
    auto avg_search_ms = static_cast<double>(total_search_ms) / kSearchRuns;
    std::cout << "[SEARCH] Elapsed (avg): " << avg_search_ms << " ms ("
              << avg_search_ms / 1000.0 << " s)" << std::endl;
    if (avg_search_ms > 0) {
      auto avg_qps = (static_cast<double>(out_queries) * 1000.0) / avg_search_ms;
      std::cout << "[SEARCH] Approx QPS (avg over " << kSearchRuns << " runs): " << avg_qps << std::endl;
    }

    std::cout << "[SEARCH] ✓ Search completed successfully" << std::endl;
    std::cout << "  - Actual queries: " << out_queries << std::endl;
    std::cout << "  - k (neighbors per query): " << out_k << std::endl;
    std::cout << "  - Output shape: (" << out_queries << " x " << out_k << ")" << std::endl;
    if (out_queries > 0) {
      std::cout << "  - Sample neighbors[0]: ";
      for (std::size_t i = 0; i < std::min(out_k, (std::size_t)5); ++i) {
        std::cout << neighbors[i] << " ";
      }
      std::cout << std::endl;
      std::cout << "  - Sample search_ids[0]: ";
      for (std::size_t i = 0; i < std::min(out_k, (std::size_t)5); ++i) {
        std::cout << search_ids[i] << " ";
      }
      std::cout << std::endl;
    }
  } catch (const std::exception& e) {
    std::cerr << "search_from_config_float failed: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "✓ All phases completed successfully!" << std::endl;
  std::cout << "  Build → Insert → Search pipeline executed without errors" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  return 0;
}
