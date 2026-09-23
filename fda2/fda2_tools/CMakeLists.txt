cmake_minimum_required(VERSION 3.18)
project(fda2_vamana_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(OpenMP REQUIRED)

set(VAMANA_TOOLS
    bfs_level_counts
    cluster_incoming_quality
    connected_components_scc
    diff_knn_graphs
    discover_vamana_graph
    gen_vamana_graph
    print_vamana_graph
    test_vamana_bidirectional
    test_vamana_quality
    vamana_indegree_distribution)

foreach(t ${VAMANA_TOOLS})
    add_executable(${t} ${t}.cpp vamana_graph.cpp)
    target_link_libraries(${t} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${t} PRIVATE -O3 -march=native)
endforeach()
