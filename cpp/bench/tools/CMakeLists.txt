# cpp/bench/tools/CMakeLists.txt

add_executable(print_cagra_graph print_cagra_graph.cpp)
add_executable(gen_cagra_graph gen_cagra_graph.cpp)
add_executable(diff_knn_graphs diff_knn_graphs.cpp)
add_executable(connected_components_scc connected_components_scc.cpp)
add_executable(test_knn_quality test_knn_quality.cpp)

# Try to find RAFT and RMM
find_package(raft REQUIRED CONFIG)
find_package(rmm REQUIRED CONFIG)
find_package(CUDAToolkit REQUIRED)
find_package(cuvs REQUIRED)
find_package(OpenMP REQUIRED)

target_link_libraries(print_cagra_graph
    PRIVATE
        raft::raft          # RAFT main library
        rmm::rmm            # RMM memory manager
        CUDA::cudart        # CUDA runtime
)

target_link_libraries(gen_cagra_graph
    PRIVATE
        raft::raft          # RAFT main library
        rmm::rmm            # RMM memory manager
        CUDA::cudart        # CUDA runtime
        cuvs::cuvs          # cuvs
)

target_link_libraries(diff_knn_graphs
    PRIVATE
        raft::raft          # RAFT main library
        rmm::rmm            # RMM memory manager
        CUDA::cudart        # CUDA runtime
)

target_link_libraries(connected_components_scc
    PRIVATE
        raft::raft          # RAFT main library
        rmm::rmm            # RMM memory manager
        CUDA::cudart        # CUDA runtime
)

target_link_libraries(test_knn_quality
    PRIVATE
        raft::raft
        rmm::rmm
        CUDA::cudart
        OpenMP::OpenMP_CXX
)

# Add to the meta target "tools" defined in cpp/CMakeLists.txt
add_dependencies(tools print_cagra_graph)
add_dependencies(tools gen_cagra_graph)
add_dependencies(tools diff_knn_graphs)
add_dependencies(tools connected_components_scc)
add_dependencies(tools test_knn_quality)

