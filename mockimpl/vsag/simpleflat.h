#pragma once

#include <cstdint>
#include <vector>

#include "vsag/index.h"

namespace vsag {

class SimpleFlat : public Index {
public:
    explicit SimpleFlat(const std::string& metric_type);

    tl::expected<int64_t, index_error>
    Build(const Dataset& base) override;

    virtual tl::expected<int64_t, index_error>
    Add(const Dataset& base) override;

    tl::expected<Dataset, index_error>
    KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) const override;

    tl::expected<Dataset, index_error>
    RangeSearch(const Dataset& query, float radius, const std::string& parameters) const override;

public:
    tl::expected<BinarySet, index_error>
    Serialize() const override;

    tl::expected<void, index_error>
    Deserialize(const BinarySet& binary_set) override;

    tl::expected<void, index_error>
    Deserialize(const ReaderSet& reader_set) override;

    int64_t
    GetMemoryUsage() const override {
        size_t ids_size = num_elements_ * sizeof(int64_t);
        size_t vector_size = num_elements_ * dim_ * sizeof(float);
        return ids_size + vector_size;
    }

    int64_t
    GetNumElements() const override {
        return num_elements_;
    }

private:
    using rs = std::pair<float, int64_t>;

    std::vector<rs>
    knn_search(const float* query, int64_t k) const;

    std::vector<rs>
    range_search(const float* query, float radius) const;

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
