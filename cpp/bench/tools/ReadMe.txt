# Steps to build the bench/tools

cd <your path>/cuvs/cpp 
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target tools -j
bench/tools/print_cagra_graph <graph index file path>

