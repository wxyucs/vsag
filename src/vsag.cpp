#include "vsag/vsag.h"

#include "hnswlib/hnswlib.h"

namespace vsag {

HNSW::HNSW(int dim, int max_elements, int M, int ef_construction) {
    space = std::make_shared<hnswlib::L2Space>(dim);
    alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW<float>>(
        space.get(), max_elements, M, ef_construction);
}

void
HNSW::addPoint(std::vector<float> datapoint, size_t label) {
    alg_hnsw->addPoint(datapoint.data(), label);
}

std::priority_queue<std::pair<float, size_t>>
HNSW::searchTopK(std::vector<float> query_data, size_t k) {
    std::priority_queue<std::pair<float, size_t>> results =
        alg_hnsw->searchKnn(query_data.data(), k);
    return results;
}

}  // namespace vsag