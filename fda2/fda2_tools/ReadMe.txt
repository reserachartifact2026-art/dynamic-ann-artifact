# FDA2 Vamana graph tools

This directory ports the graph-analysis utilities from the artifact
`nvidia-cuvs/cpp/bench/tools` to the FDA2/FreshDiskANN Vamana on-disk format.

## FDA2 Vamana record format

Each node is stored as a fixed-size record:

FP32 graph:
    [D x float32 vector][uint32 degree][R x uint32 neighbors]

UINT8 graph:
    [D x uint8 vector][uint32 degree][R x uint32 neighbors]

For the current SIFT workloads:

    SIFT-1M   : N=500K,  D=128, R=64, FP32
    SIFT-10M  : N=5M,    D=128, R=64, FP32
    SIFT-100M : N=50M,   D=128, R=64, UINT8

Only the first `degree` neighbor slots are valid. The remaining slots are
fixed-capacity storage.

## Tools

- bfs_level_counts
- cluster_incoming_quality
- connected_components_scc
- diff_knn_graphs
- discover_vamana_graph
- gen_vamana_graph
- print_vamana_graph
- test_vamana_bidirectional
- test_vamana_quality
- vamana_indegree_distribution

The first eight correspond to the artifact tools:

    bfs_level_counts.cpp
    cluster_incoming_quality.cpp
    connected_components_scc.cpp
    diff_knn_graphs.cpp
    discover_cagra_graph.cpp
    gen_cagra_graph.cpp
    print_cagra_graph.cpp
    test_knn_quality.cpp

with CAGRA-specific serialization replaced by the FDA2 Vamana reader.

`test_vamana_bidirectional` and `vamana_indegree_distribution` are additional
Vamana-specific analysis tools.

## Build

### Make

    make -j

### CMake

    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    cmake --build . -j

No RAFT/CUDA/CUVS dependency is required by these Vamana ports.

## Examples

### Bidirectional edges

    ./test_vamana_bidirectional \
      --format f32 --nodes 500000 --dim 128 --degree 64 \
      /path/vamana_500k.out --threads 64 --histogram

### In-degree distribution

    ./vamana_indegree_distribution \
      --format u8 --nodes 50000000 --dim 128 --degree 64 \
      /path/vamana_sift_50m_u8.out --csv sift100m_indegree.csv

### Connected components / SCC

    ./connected_components_scc \
      --format f32 --nodes 5000000 --dim 128 --degree 64 \
      /path/vamana_sift_5m.out

### BFS level counts

    ./bfs_level_counts \
      --format f32 --nodes 500000 --dim 128 --degree 64 \
      /path/vamana_500k.out 0

### Print graph rows

    ./print_vamana_graph \
      --format f32 --nodes 500000 --dim 128 --degree 64 \
      /path/vamana_500k.out 0 10

### Discover incoming edges through graph traversal

    ./discover_vamana_graph \
      --format f32 --nodes 500000 --dim 128 --degree 64 \
      --target 100 --scan-range 0 499999 --levels 20 \
      /path/vamana_500k.out

## Notes on scale

The common reader keeps the adjacency graph in host RAM. For a 50M-node,
R=64 graph this is approximately 12.8 GB for adjacency plus the per-node
metadata.

`vamana_indegree_distribution` is intentionally streaming and only needs one
uint32 in-degree counter per node (~200 MB for 50M nodes), so it is preferable
for large graphs.

The cluster-incoming and exact brute-force quality tools require vector data
and are intended primarily for smaller graphs. `test_vamana_quality` requires
`--vectors-fbin` for distance-order/recall modes and disables brute-force
Recall@K automatically for N > 100K.

The SCC implementation retains the artifact's Tarjan-style semantics; for
very large graphs, check available stack/memory before running it.
