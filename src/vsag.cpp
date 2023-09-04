#include "vsag/vsag.h"

namespace vsag {

HNSW::HNSW(std::shared_ptr<hnswlib::SpaceInterface> spaceInterface,
           int max_elements,
           int M,
           int ef_construction,
           int ef_runtime)
    : space(std::move(spaceInterface)) {
    alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW>(space.get(), max_elements, M, ef_construction);
    setEfRuntime(ef_runtime);
}

void
HNSW::addPoint(const void* datapoint, size_t label) {
    alg_hnsw->addPoint(datapoint, label);
}

void
HNSW::setEfRuntime(size_t ef_runtime) {
    alg_hnsw->setEf(ef_runtime);
}

std::priority_queue<std::pair<float, size_t>>
HNSW::searchTopK(const void* query_data, size_t k) {
    std::priority_queue<std::pair<float, size_t>> results = alg_hnsw->searchKnn(query_data, k);
    return results;
}

Vamana::Vamana(diskann::Metric metric, size_t data_dim, size_t data_num, std::string data_type)
    : data_dim_(data_dim), data_num_(data_num) {
    if (data_type == "float32") {
        index = std::make_shared<diskann::Index<float, uint32_t, uint32_t>>(
            metric, data_dim, data_num, false, false, false, false, 0, false);
    }
}

std::priority_queue<std::pair<float, size_t>>
Vamana::searchTopK(const void* query_data, size_t k) {
    std::priority_queue<std::pair<float, size_t>> results;

    if (!index)
        return results;
    uint32_t labels[k];
    float distances[k];
    index->search((const float*)query_data, k, ef_runtime, labels, distances);
    for (size_t i = 0; i < k; i++) {
        results.emplace(distances[i], labels[i]);
    }

    return results;
}

void
Vamana::build(const void* datapoint, size_t ef_runtime, int M) {
    setEfRuntime(ef_runtime);
    auto index_build_params = diskann::IndexWriteParametersBuilder(ef_runtime, M).build();
    std::vector<uint32_t> tags;
    for (uint32_t i = 0; i < data_num_; ++i) {
        tags.push_back(i);
    }
    index->build((const float*)datapoint, data_num_, index_build_params, tags);
}

void
Vamana::setEfRuntime(size_t ef_runtime) {
    this->ef_runtime = ef_runtime;
}

}  // namespace vsag