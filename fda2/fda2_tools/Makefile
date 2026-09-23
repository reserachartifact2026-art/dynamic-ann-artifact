CXX ?= g++
CXXFLAGS ?= -O3 -march=native -std=c++17 -fopenmp

COMMON = vamana_graph.cpp

TOOLS = \
    bfs_level_counts \
    cluster_incoming_quality \
    connected_components_scc \
    diff_knn_graphs \
    discover_vamana_graph \
    gen_vamana_graph \
    print_vamana_graph \
    test_vamana_bidirectional \
    test_vamana_quality \
    vamana_indegree_distribution

all: $(TOOLS)

bfs_level_counts: bfs_level_counts.cpp $(COMMON) vamana_graph.h
	$(CXX) $(CXXFLAGS) bfs_level_counts.cpp $(COMMON) -o $@

cluster_incoming_quality: cluster_incoming_quality.cpp $(COMMON) vamana_graph.h
	$(CXX) $(CXXFLAGS) cluster_incoming_quality.cpp $(COMMON) -o $@

connected_components_scc: connected_components_scc.cpp $(COMMON) vamana_graph.h
	$(CXX) $(CXXFLAGS) connected_components_scc.cpp $(COMMON) -o $@

diff_knn_graphs: diff_knn_graphs.cpp $(COMMON) vamana_graph.h
	$(CXX) $(CXXFLAGS) diff_knn_graphs.cpp $(COMMON) -o $@

discover_vamana_graph: discover_vamana_graph.cpp $(COMMON) vamana_graph.h
	$(CXX) $(CXXFLAGS) discover_vamana_graph.cpp $(COMMON) -o $@

gen_vamana_graph: gen_vamana_graph.cpp $(COMMON) vamana_graph.h
	$(CXX) $(CXXFLAGS) gen_vamana_graph.cpp $(COMMON) -o $@

print_vamana_graph: print_vamana_graph.cpp $(COMMON) vamana_graph.h
	$(CXX) $(CXXFLAGS) print_vamana_graph.cpp $(COMMON) -o $@

test_vamana_bidirectional: test_vamana_bidirectional.cpp $(COMMON) vamana_graph.h
	$(CXX) $(CXXFLAGS) test_vamana_bidirectional.cpp $(COMMON) -o $@

test_vamana_quality: test_vamana_quality.cpp $(COMMON) vamana_graph.h
	$(CXX) $(CXXFLAGS) test_vamana_quality.cpp $(COMMON) -o $@

vamana_indegree_distribution: vamana_indegree_distribution.cpp $(COMMON) vamana_graph.h
	$(CXX) $(CXXFLAGS) vamana_indegree_distribution.cpp $(COMMON) -o $@

clean:
	rm -f $(TOOLS)
