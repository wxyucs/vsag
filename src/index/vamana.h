#pragma once

#include <ann_exception.h>
#include <index.h>
#include <index_factory.h>
#include <memory_mapper.h>
#include <utils.h>

#include <cstdint>
#include <memory>
#include <queue>
#include <utility>
#include <vector>

#include "vsag/index.h"

namespace vsag {

class Vamana : public Index {
public:
    Vamana(diskann::Metric metric, size_t data_dim, size_t data_num, std::string data_type);

    void
    Build(const Dataset& base) override;

    Dataset
    KnnSearch(const Dataset& query, int64_t k, const nlohmann::json& parameters) override;

public:
    using rs = std::pair<float, size_t>;

    std::priority_queue<rs, std::vector<rs>, std::less<rs>>
    searchTopK(const void* query_data, size_t k);

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

}  // namespace vsag
