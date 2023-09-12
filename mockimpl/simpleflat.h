#pragma once

#include <cstdint>
#include <vector>

#include "vsag/index.h"

namespace vsag {

class SimpleFlat : public Index {
public:
    explicit SimpleFlat(const std::string& metric_type);

    void
    Build(const Dataset& base) override;

    void
    Add(const Dataset& base) override;

    Dataset
    KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) override;

public:
    BinarySet
    Serialize() override;

    void
    Deserialize(const BinarySet& binary_set) override;

private:
    using rs = std::pair<float, int64_t>;

    std::vector<rs>
    knn_search(const float* query, int64_t k) const;

    static float
    l2(const float* v1, const float* v2, int64_t dim);

    static float
    ip(const float* v1, const float* v2, int64_t dim);

    static float
    cosine(const float* v1, const float* v2, int64_t dim);

private:
    const std::string metric_type_;
    int64_t num_elements_ = 0;
    int64_t dim_ = 0;
    std::vector<int64_t> ids_;
    std::vector<float> data_;
};

}  // namespace vsag
