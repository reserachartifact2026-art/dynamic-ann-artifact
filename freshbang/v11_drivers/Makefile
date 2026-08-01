# Makefile for the shared v11 workload drivers.
#
# v11_loader.hpp is header-only (no deps). The FreshBANG driver links the
# prebuilt libfreshbang.so, which is compiled against the cuvs_bench conda
# env's libstdc++ — so we must build with that env's g++ and rpath its lib.

# ---- paths (override on the command line if they move) ----
FB_DIR   ?= /tmp/freshbang
ENV_DIR  ?= /mnt/ssd_volume/mumba1/envs/cuvs_bench_26_02
CXX_FB   ?= $(ENV_DIR)/bin/x86_64-conda-linux-gnu-g++
CXX      ?= g++

CXXFLAGS := -O2 -std=c++17 -fopenmp
FB_INC   := -I$(FB_DIR) -I.
FB_LINK  := -L$(FB_DIR) -lfreshbang -lpthread -ldl -Wl,-rpath,$(ENV_DIR)/lib -Wl,-rpath,$(FB_DIR)

.PHONY: all clean selftest freshbang debug

all: v11_selftest freshbang_v11_driver

# Standalone parser/GT self-test (plain g++, no freshbang).
v11_selftest: v11_selftest.cpp v11_loader.hpp
	$(CXX) $(CXXFLAGS) v11_selftest.cpp -o v11_selftest

# FreshBANG driver — must use the conda-env g++ for GLIBCXX compatibility.
freshbang_v11_driver: freshbang_v11_driver.cpp v11_loader.hpp
	$(CXX_FB) -O2 -std=c++17 $(FB_INC) freshbang_v11_driver.cpp $(FB_LINK) -o freshbang_v11_driver

selftest: v11_selftest
freshbang: freshbang_v11_driver

clean:
	rm -f v11_selftest freshbang_v11_driver

# ---- DEBUG builds (-O2 -ggdb3): for debugging only, do NOT use for results ----
freshbang_v11_driver_debug: freshbang_v11_driver.cpp v11_loader.hpp
	$(CXX_FB) -O2 -ggdb3 -std=c++17 $(FB_INC) freshbang_v11_driver.cpp $(FB_LINK) -o freshbang_v11_driver_debug

freshbang_v11_driver_u8_debug: freshbang_v11_driver.cpp v11_loader.hpp
	$(CXX_FB) -O2 -ggdb3 -std=c++17 -DFB_DT=uint8_t $(FB_INC) freshbang_v11_driver.cpp $(FB_LINK) -o freshbang_v11_driver_u8_debug

debug: freshbang_v11_driver_debug freshbang_v11_driver_u8_debug
