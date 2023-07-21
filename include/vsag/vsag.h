#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "hnswlib/hnswlib.h"
namespace vsag {

class IndexInterface {
public:
    virtual void
    addPoint(const void* datapoint, size_t label) = 0;

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
    addPoint(const void* datapoint, size_t label) override;

    std::priority_queue<std::pair<float, size_t>>
    searchTopK(const void* query_data, size_t k) override;

private:
    std::shared_ptr<hnswlib::HierarchicalNSW> alg_hnsw;
    std::shared_ptr<hnswlib::SpaceInterface> space;
};

float
kmeans_clustering(
    size_t d, size_t n, size_t k, const float* x, float* centroids, const std::string& dis_type);

}  // namespace vsag
