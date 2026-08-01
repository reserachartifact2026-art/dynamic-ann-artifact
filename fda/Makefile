# FreshDiskANN (GPU) — Naive Implementation
#
# Compiles a single binary `bin/fda_fresh_diskann` with the same CLI
# shape as mvcc_fresh_diskann / concurrent_fresh_diskann.
#
# Reuses primitives from baseline_src/ (which is a copy of the BANG
# building blocks: greedy search, RobustPrune, distance kernels,
# bloom filter, workload parser, DeleteList).

OUT := bin/fda_fresh_diskann

NVCC_FLAGS := -O3 -std=c++17 -rdc=true \
              -Xcompiler -fopenmp \
              -Xcompiler -static-libstdc++ \
              --extended-lambda

SRCS := src/fda_main.cu \
        src/fda_executor.cu \
        src/fda_index.cu \
        src/fda_insert.cu \
        src/fda_search.cu \
        src/fda_merge.cu \
        baseline_src/util.cu \
        baseline_src/bloomFilter.cu \
        baseline_src/greedySearch.cu \
        baseline_src/outNeighbors.cu \
        baseline_src/reverseEdge.cu \
        baseline_src/dynamic/deleteList.cu \
        baseline_src/dynamic/workload.cpp

LIBS := -lpthread -lgomp

compile:
	mkdir -p bin
	nvcc $(NVCC_FLAGS) $(SRCS) $(LIBS) -o $(OUT)

compile-debug:
	mkdir -p bin
	nvcc -G -g -std=c++17 -rdc=true \
	     -Xcompiler -fopenmp -Xcompiler -static-libstdc++ \
	     --extended-lambda \
	     $(SRCS) $(LIBS) -o $(OUT)

clean:
	rm -rf bin

.PHONY: compile compile-debug clean
