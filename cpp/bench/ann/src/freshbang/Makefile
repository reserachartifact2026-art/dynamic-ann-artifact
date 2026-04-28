CXX ?= g++
CXXFLAGS ?= -std=c++17 -O2 -ggdb3

# Keep macro consistent with existing manual build command
DEFINES := -DBENCH_NS=raft::bench

# If CONDA_PREFIX is not set, callers can override INC_CUDA/LIB_CUDA manually.
INC_CUDA := -I$(CONDA_PREFIX)/targets/x86_64-linux/include
LIB_CUDA := -L$(CONDA_PREFIX)/lib

INCLUDES := -I../common -I.. -I../../../../include -I../../../../../c/include $(INC_CUDA)
LDFLAGS_SO := $(LIB_CUDA) -lcudart -ldl
LDFLAGS_EXE := $(LIB_CUDA) -lcudart -ldl

SO_TARGET := libfreshbang.so
DRIVER_TARGET := freshbang_driver

SRCS_COMMON := freshbang_wrapper.cpp
SRCS_DRIVER := freshbang_driver.cpp $(SRCS_COMMON)

.PHONY: all clean

all: $(SO_TARGET) $(DRIVER_TARGET)

$(SO_TARGET): $(SRCS_COMMON) freshbang.hpp freshbang_helpers.hpp
	$(CXX) $(CXXFLAGS) -fPIC -shared $(DEFINES) $(INCLUDES) $(SRCS_COMMON) $(LDFLAGS_SO) -o $@

$(DRIVER_TARGET): $(SRCS_DRIVER) freshbang.hpp freshbang_helpers.hpp
	$(CXX) $(CXXFLAGS) $(DEFINES) $(INCLUDES) $(SRCS_DRIVER) $(LDFLAGS_EXE) -o $@

clean:
	rm -f $(SO_TARGET) $(DRIVER_TARGET)
