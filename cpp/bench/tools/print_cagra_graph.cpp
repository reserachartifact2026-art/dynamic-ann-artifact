// Example: Print CAGRA graph from serialized bin file
//#include <raft/core/platform.hpp>
#define RAFT_SYSTEM_LITTLE_ENDIAN 1
#include <raft/core/host_mdarray.hpp>
#include <raft/core/serialize.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdlib>   // std::strtoul
#include <algorithm> // std::min

#include <cmath> // for math
#include <iomanip>


std::vector<float> load_fbin(
    const std::string& path,
    uint32_t& n_vecs,
    uint32_t& dim)
{
    std::ifstream is(path, std::ios::binary);
    if (!is) {
        throw std::runtime_error("Failed to open fbin file");
    }

    is.read(reinterpret_cast<char*>(&n_vecs), sizeof(uint32_t));
    is.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

    std::vector<float> data(static_cast<size_t>(n_vecs) * dim);
    is.read(reinterpret_cast<char*>(data.data()),
            data.size() * sizeof(float));

    if (!is) {
        throw std::runtime_error("Failed to read fbin data");
    }

    return data;
}

float l2_distance(
    const float* a,
    const float* b,
    uint32_t dim)
{
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return std::sqrt(sum);
}



int main(int argc, char** argv) {

    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <cagra_graph.bin> [start_row] [end_row]\n";
        return 1;
    }

    bool print_dist = false;
    std::string fbin_path;

    for (int i = 2; i < argc; ++i) {
	    if (std::string(argv[i]) == "-d") {
		    print_dist = true;
	    } else if (argv[i][0] != '-') {
		    fbin_path = argv[i];
	    }
    }

    if (print_dist && fbin_path.empty()) {
	    std::cerr << "-d specified but no .fbin file provided\n";
	    return 1;
    }

    using IdxT = uint32_t; // Change to uint32_t if needed

    std::ifstream is(argv[1], std::ios::binary);
    if (!is) {
        std::cerr << "Failed to open file\n";
        return 1;
    }

    raft::resources handle;

    // Skip dtype string
    char dtype_string[4];
    is.read(dtype_string, 4);
std::cout << "dtype_string: " << std::string(dtype_string, 4) << std::endl;
if (!is) { std::cerr << "Failed to read dtype_string\n"; return 1; }


    // Read version
    int ver = raft::deserialize_scalar<int>(handle, is);
std::cout << "version: " << ver << std::endl;
if (!is) { std::cerr << "Failed to read version\n"; return 1; }



    // Read shape and graph params
    IdxT n_rows = raft::deserialize_scalar<IdxT>(handle, is);
    std::cout << "n_rows: " << n_rows << std::endl;
    if (!is) { std::cerr << "Failed to read n_rows\n"; return 1; }

    IdxT start_row = 0;
    IdxT end_row   = n_rows;  // exclusive

    if (argc >= 3) {
	    start_row = static_cast<IdxT>(std::strtoul(argv[2], nullptr, 10));
    }

    if (argc >= 4) {
	    end_row = static_cast<IdxT>(std::strtoul(argv[3], nullptr, 10));
    }

    // Clamp to valid range
    start_row = std::min(start_row, n_rows);
    end_row   = std::min(end_row, n_rows);

    if (start_row >= end_row) {
	    std::cerr << "Invalid row range: [" << start_row
		    << ", " << end_row << ")\n";
	    return 1;
    }

    uint32_t dim = raft::deserialize_scalar<uint32_t>(handle, is);
    std::cout << "dim: " << dim << std::endl;
    uint32_t graph_degree = raft::deserialize_scalar<uint32_t>(handle, is);

    uint32_t vec_dim = 0;
    uint32_t n_vecs = 0;
    std::vector<float> vectors;

    if (print_dist) {
	    vectors = load_fbin(fbin_path, n_vecs, vec_dim);

	    if (vec_dim != dim) {
		    std::cerr << "Dimension mismatch: graph dim = "
			    << dim << ", fbin dim = " << vec_dim << "\n";
		    dim = vec_dim;
//		    return 1;
	    }

	    if (n_vecs < n_rows) {
		    std::cerr << "fbin has fewer vectors than graph rows\n";
		    return 1;
	    }
    }



    std::cout << "graph_degree: " << graph_degree << std::endl;
    // Skip metric
    uint metric_dummy = raft::deserialize_scalar<int>(handle, is);
    //uint32_t metric_dummy = raft::deserialize_scalar<uint32_t>(handle, is);

    std::cout << "metric: " << metric_dummy << std::endl;
    // Read the graph
    auto graph = raft::make_host_matrix<IdxT, int64_t>(n_rows, graph_degree);
    raft::deserialize_mdspan(handle, is, graph.view());

    std::cout << "Graph rows [" << start_row << ", " << end_row
	    << ") (" << n_rows << " x " << graph_degree << "):\n";

    for (IdxT i = start_row; i < end_row; ++i) {
	    std::cout << i << ": ";

	    const float* vi =
		    print_dist ? &vectors[static_cast<size_t>(i) * dim] : nullptr;

	    for (uint32_t j = 0; j < graph_degree; ++j) {
		    IdxT nbr = graph(i, j);
		    std::cout << nbr;

		    if (print_dist) {
			    const float* vj =
				    &vectors[static_cast<size_t>(nbr) * dim];

			    float dist = l2_distance(vi, vj, dim);
			    std::cout << "(" << std::fixed << std::setprecision(2)
				    << dist << ")";
		    }

		    std::cout << " ";
	    }
	    std::cout << "\n";
    }
uint32_t content_map = raft::deserialize_scalar<uint32_t>(handle, is);
std::cout << "content_map: " << content_map << "\n";
    return 0;
}
