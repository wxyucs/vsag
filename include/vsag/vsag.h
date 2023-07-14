#include <cstdint>
#include <memory>
#include <vector>

#include "hnswlib/hnswlib.h"
namespace vsag {

class IndexInterface {
public:
    virtual void
    addPoint(std::vector<float> datapoint, size_t label) = 0;

    virtual std::priority_queue<std::pair<float, size_t>>
    searchTopK(std::vector<float> query_data, size_t k) = 0;

    virtual ~IndexInterface() = default;
};

class HNSW : public IndexInterface {
public:
    HNSW(int dim, int max_elements, int M, int ef_construction);

    void
    addPoint(std::vector<float> datapoint, size_t label) override;

    std::priority_queue<std::pair<float, size_t>>
    searchTopK(std::vector<float> query_data, size_t k) override;

private:
    std::shared_ptr<hnswlib::HierarchicalNSW<float>> alg_hnsw;
    std::shared_ptr<hnswlib::L2Space> space;
};

}  // namespace vsag
