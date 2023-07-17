#include "vsag/vsag.h"

#include "hnswlib/hnswlib.h"

namespace vsag {

HNSW::HNSW(int dim, int max_elements, int M, int ef_construction) {
    space = std::make_shared<hnswlib::InnerProductSpaceInt8>(dim);
    alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW>(
        space.get(), max_elements, M, ef_construction);
}

void
HNSW::addPoint(const void* datapoint, size_t label) {
    alg_hnsw->addPoint(datapoint, label);
}

std::priority_queue<std::pair<float, size_t>>
HNSW::searchTopK(const void* query_data, size_t k) {
    std::priority_queue<std::pair<float, size_t>> results =
        alg_hnsw->searchKnn(query_data, k);
    return results;
}

}  // namespace vsag