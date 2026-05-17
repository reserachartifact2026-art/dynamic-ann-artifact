#pragma once

#include <cstdint>
#include <string>

typedef struct _SearchParams
{
    uint32_t recall_at_k;
    // will add more
} SearchParams;

typedef struct _BuildParams
{
     uint32_t dataset_rows;
    uint32_t dataset_dim;
    std::string distance_measure;
} BuildParams;

// Note used 
typedef struct _InsertParams
{
    bool persist_graph = false;
} InsertParams;

template<typename T>
class FreshBANG
{
private:
    void* m_pImpl;
    // user supplied params for build and search, 
    BuildParams user_build_params_;
    SearchParams user_search_params_;

public:
    static constexpr const char* kDistanceMeasure_Euclidean = "euclidean";
    static constexpr const char* kDistanceMeasure_InnerProduct = "inner_product";

    FreshBANG();
    
    // <--- (1) Initialization Sequence of Operations ---> 
    // called only ONCE in the beginning, not before every build/createAlgo
    bool SetDatasetParams(BuildParams params);

    bool CreateAlgo(const std::string& conf_path);

    bool BuildIndex(const T* base_vectors, uint32_t num_base_vectors);

    // called only ONCE in the beginning, not before every search
    bool SetSearchParams(SearchParams param);

    // called only ONCE in the beginning, not before every insert
    bool SetInsertParams();

    // <--- (2)  Dynamic Operations ---> 
    // Pre-req: For all the search/insert/delete APIs, ensure ALL the SetXParams are done before calling them
   
    // Note: Only ONE batched operation can be performed at a time. 
    // Concurrent operations (e.g., search and insert at the same time) are not supported in this version.

    // Important note on IDs: These are semantically slot or row numbers and not running numbers. 
    // Therefore, they are in the range 0 - N-1 where N = BuildParams.dataset_rows. 
    // These are re-used across deletes and inserts. 
    void BatchedSearch(const T* searchvectors, 
                        uint32_t batch_size,
                        uint64_t* ids,
                        uint32_t* neighbors, 
                        float* distances); 
    // Note: The ids need to be contigious     
    void BatchedInsert(const T* insertvectors, 
                        uint32_t batch_size,
                        const uint64_t* ids 
                        );
    // Note: The ids need to be contigious                         
    void BatchedDelete(const uint64_t* ids, 
                        uint32_t batch_size);

    void SaveIndex();
    void Cleanup();
    
    virtual ~FreshBANG();
};

// --- explicit instantiation ---
template class FreshBANG<float>;
// More data types can be instantiated as needed
//template class FreshBANG<uint8_t>;
//template class FreshBANG<int8_t>;