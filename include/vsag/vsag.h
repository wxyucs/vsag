#include <cstdint>
#include <memory>
#include <vector>

#include "hnswlib/hnswlib.h"
namespace vsag {

class IndexInterface {
public:
    virtual void
    addPoint(std::shared_ptr<const void> datapoint, size_t label) = 0;

    virtual std::priority_queue<std::pair<float, size_t>>
    searchTopK(std::shared_ptr<const void> query_data, size_t k) = 0;

    virtual ~IndexInterface() = default;
};


template <typename DataType>
class HNSW : public IndexInterface {
public:
    HNSW(int dim, int max_elements, int M, int ef_construction) {
        space = std::make_shared<hnswlib::L2Space>(dim);
        alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW<DataType>>(
            space.get(), max_elements, M, ef_construction);
    }

    void
    addPoint(std::shared_ptr<const void> datapoint, size_t label) override {
        alg_hnsw->addPoint(datapoint.get(), label);
    }

    std::priority_queue<std::pair<float, size_t>>
    searchTopK(std::shared_ptr<const void> query_data, size_t k) override {
        std::priority_queue<std::pair<float, size_t>> results =
            alg_hnsw->searchKnn(query_data.get(), k);
        return results;
    }

private:
    std::shared_ptr<hnswlib::HierarchicalNSW<DataType>> alg_hnsw;
    std::shared_ptr<hnswlib::L2Space> space;
};

}  // namespace vsag
