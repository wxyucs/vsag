

#include "hnsw.h"

#include <hnswlib/hnswlib.h>

#include <cstdint>
#include <memory>
#include <nlohmann/json.hpp>

namespace vsag {

HNSW::HNSW(std::shared_ptr<hnswlib::SpaceInterface> spaceInterface,
           int max_elements,
           int M,
           int ef_construction)
    : space(std::move(spaceInterface)) {
    alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW>(space.get(), max_elements, M, ef_construction);
}

void
HNSW::Build(const Dataset& base) {
    int64_t num_elements = base.GetNumElements();
    int64_t dim = base.GetDim();
    auto ids = base.GetIds();
    auto vectors = base.GetFloat32Vectors();
    for (int64_t i = 0; i < num_elements; ++i) {
        alg_hnsw->addPoint((const void*)(vectors + i * dim), ids[i]);
    }
}

void
HNSW::Add(const Dataset& base) {
    int64_t num_elements = base.GetNumElements();
    int64_t dim = base.GetDim();
    auto ids = base.GetIds();
    auto vectors = base.GetFloat32Vectors();
    for (int64_t i = 0; i < num_elements; ++i) {
        alg_hnsw->addPoint((const void*)(vectors + i * dim), ids[i]);
    }
}

Dataset
HNSW::KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) {
    nlohmann::json params = nlohmann::json::parse(parameters);
    if (params.contains("ef_runtime")) {
        alg_hnsw->setEf(params["ef_runtime"]);
    }

    int64_t num_elements = query.GetNumElements();
    int64_t dim = query.GetDim();
    auto vectors = query.GetFloat32Vectors();

    Dataset result;
    int64_t* ids = new int64_t[num_elements * k];
    float* dists = new float[num_elements * k];
    for (int64_t i = 0; i < num_elements; ++i) {
        std::priority_queue<std::pair<float, size_t>> results =
            alg_hnsw->searchKnn((const void*)(vectors + i * dim), k);
        for (int64_t j = 0; j < k; ++j) {
            dists[i * k + j] = results.top().first;
            ids[i * k + j] = results.top().second;
            results.pop();
        }
    }
    result.SetIds(ids);
    result.SetDistances(dists);

    return result;
}

void
HNSW::SetEfRuntime(int64_t ef_runtime) {
    alg_hnsw->setEf(ef_runtime);
}

}  // namespace vsag
