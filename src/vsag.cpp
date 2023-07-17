#include "vsag/vsag.h"

#include "hnswlib/hnswlib.h"

namespace vsag {

//template <typename DataType>
//HNSW<DataType>::HNSW(int dim, int max_elements, int M, int ef_construction) {
//    space = std::make_shared<hnswlib::L2Space>(dim);
//    alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW<DataType>>(
//        space.get(), max_elements, M, ef_construction);
//}
//
//template <typename DataType> void
//HNSW<DataType>::addPoint(std::shared_ptr<const void> datapoint, size_t label) {
//    alg_hnsw->addPoint(datapoint.get(), label);
//}
//
//template <typename DataType> std::priority_queue<std::pair<float, size_t>>
//HNSW<DataType>::searchTopK(std::shared_ptr<const void> query_data, size_t k) {
//    std::priority_queue<std::pair<float, size_t>> results =
//        alg_hnsw->searchKnn(query_data.get(), k);
//    return results;
//}

}  // namespace vsag