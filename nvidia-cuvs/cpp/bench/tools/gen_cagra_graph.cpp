#define RAFT_SYSTEM_LITTLE_ENDIAN 1
//#include <cuvs/distance/distance_types.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/serialize.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <random>
#include <cstdint>

template <typename T>
void write_scalar(std::ofstream& os, const T& value) {
    os.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

/*
CAGRA Graph File Format 
[dtype string]
[version]
[n_rows]
[dim]
[degree]
[metric]
[NUMPY ARRAY HEADER]
[NUMPY ARRAY DATA]
*/




int main() {
    uint32_t n_rows, degree, dim;
    int mode;

    std::cout << "Number of nodes: ";
    std::cin >> n_rows;
    std::cout << "Degree: ";
    std::cin >> degree;
    std::cout << "Vector dim: ";
    std::cin >> dim;
    std::cout << "Fill mode (0=zero, 1=random, 2=sequential): ";
    std::cin >> mode;

    std::ofstream os("dummy_cagra.bin", std::ios::binary);

    // ---- HEADER ----
    char dtype[4] = {'<','f','3','2'};
    os.write(dtype, 4);

    int version = 5;
    //int metric = 0;

    std::cout << "version: " << version << std::endl;
    raft::resources handle;

    raft::serialize_scalar(handle, os, version);
    raft::serialize_scalar(handle, os, n_rows);
    raft::serialize_scalar(handle, os, dim);
    raft::serialize_scalar(handle, os, degree);
    //raft::serialize_scalar(handle, os, metric);
    //cuvs::distance::DistanceType metric =
    int metric = 0;
    //	    cuvs::distance::DistanceType::L2;
    raft::serialize_scalar(handle, os, metric);



    // ---- GRAPH ----
    auto graph = raft::make_host_matrix<uint32_t>(n_rows, degree);
    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(0, n_rows - 1);

    for (uint32_t i = 0; i < n_rows; ++i) {
        for (uint32_t j = 0; j < degree; ++j) {
            uint32_t val = 0;

            if (mode == 1) val = dist(rng);
            else if (mode == 2) val = (i + j) % n_rows;

            graph(i, j) = val;
        }
    }

    raft::serialize_mdspan(handle, os, graph.view());
    uint32_t content_map = 0;  // no dataset, no source indices
    raft::serialize_scalar(handle, os, content_map);
 
    std::cout << "Dummy cagra graph written to dummy_cagra.bin\n";
}

