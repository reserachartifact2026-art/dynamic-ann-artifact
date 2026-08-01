# FreshDiskANN (GPU) — Sequential Model
#
# Single unified Vamana graph; phased INSERT / DELETE / SEARCH (different
# op-types never run concurrently); reverse edges appended into per-node
# P-scratch then RobustPruned R+P -> R; in-place IP-DiskANN delete.
#
# Reuses the BANG primitives in baseline_src/ (greedy search, distances, bloom
# filter, workload parser) UNCHANGED. All sequential-model logic lives in src/.
#
# P (reverse-edge scratch slots per node) is the compile-time P_EXTRA. Override:
#   make compile P=128

OUT := bin/fda2_fresh_diskann

P ?= 64
STORE ?= float
DELLD ?= 128
DELK ?= 50

NVCC_FLAGS := -O3 -std=c++17 -rdc=true \
              -DP_EXTRA=$(P) \
              -DFDA2_STORE_T=$(STORE) \
              -DDEL_LD=$(DELLD) \
              -DDEL_K=$(DELK) \
              -Xcompiler -fopenmp \
              -Xcompiler -static-libstdc++ \
              --extended-lambda

SRCS := src/fda2_main.cu \
        src/fda2_executor.cu \
        src/fda2_graph.cu \
        src/fda2_insert.cu \
        src/fda2_delete.cu \
        src/fda2_search.cu \
        baseline_src/util.cu \
        baseline_src/bloomFilter.cu \
        baseline_src/greedySearch.cu \
        baseline_src/outNeighbors.cu \
        baseline_src/reverseEdge.cu \
        baseline_src/dynamic/workload.cpp

LIBS := -lpthread -lgomp

compile:
	mkdir -p bin
	nvcc $(NVCC_FLAGS) $(SRCS) $(LIBS) -o $(OUT)

compile-debug:
	mkdir -p bin
	nvcc -G -g -std=c++17 -rdc=true -DP_EXTRA=$(P) \
	     -Xcompiler -fopenmp -Xcompiler -static-libstdc++ \
	     --extended-lambda \
	     $(SRCS) $(LIBS) -o $(OUT)

clean:
	rm -rf bin

.PHONY: compile compile-debug clean
