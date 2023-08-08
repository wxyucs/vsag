#include "vsag/vsag.h"

#include <hnswlib/hnswlib.h>

#include <utility>

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

}  // namespace vsag