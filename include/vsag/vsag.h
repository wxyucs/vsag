#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include <index.h>
#include <utils.h>
#include <memory_mapper.h>
#include <ann_exception.h>
#include <index_factory.h>
#include <hnswlib/hnswlib.h>
namespace vsag {

class IndexInterface {
public:
    virtual std::priority_queue<std::pair<float, size_t>>
    searchTopK(const void* query_data, size_t k) = 0;

    virtual ~IndexInterface() = default;
};

class HNSW : public IndexInterface {
public:
    HNSW(std::shared_ptr<hnswlib::SpaceInterface> spaceInterface,
         int max_elements,
         int M,
         int ef_construction,
         int ef_runtime);

    void
    addPoint(const void* datapoint, size_t label);

    void
    build(const void* datapoint, size_t data_size, size_t data_dim);

    std::priority_queue<std::pair<float, size_t>>
    searchTopK(const void* query_data, size_t k) override;

    void
    setEfRuntime(size_t ef_runtime);

private:
    std::shared_ptr<hnswlib::HierarchicalNSW> alg_hnsw;
    std::shared_ptr<hnswlib::SpaceInterface> space;
};

class Vamana : public IndexInterface {
public:
    Vamana(diskann::Metric metric, size_t data_dim, size_t data_num, std::string data_type);

    std::priority_queue<std::pair<float, size_t>>
    searchTopK(const void* query_data, size_t k) override;

    void
    build(const void* datapoint, size_t ef_runtime, int M);

    void
    setEfRuntime(size_t ef_runtime);

private:
    std::shared_ptr<diskann::AbstractIndex> index;
    std::shared_ptr<diskann::IndexWriteParametersBuilder> param;
    size_t ef_runtime;
    size_t data_num_;
    size_t data_dim_;
};

float
kmeans_clustering(
    size_t d, size_t n, size_t k, const float* x, float* centroids, const std::string& dis_type);

}  // namespace vsag
