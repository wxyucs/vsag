#pragma once

#include <cstdint>
#include <vector>

#include "vsag/index.h"

namespace vsag {

class SimpleFlat : public Index {
public:
    explicit SimpleFlat(const std::string& metric_type, int64_t dim);

    tl::expected<std::vector<int64_t>, Error>
    Build(const DatasetPtr& base) override;

    virtual tl::expected<std::vector<int64_t>, Error>
    Add(const DatasetPtr& base) override;

    tl::expected<bool, Error>
    Remove(int64_t id) override;

    tl::expected<DatasetPtr, Error>
    KnnSearch(const DatasetPtr& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override;

    tl::expected<DatasetPtr, Error>
    RangeSearch(const DatasetPtr& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid = nullptr) const override;

public:
    tl::expected<BinarySet, Error>
    Serialize() const override;

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override;

    tl::expected<void, Error>
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

    std::string
    GetStats() const override;

private:
    using rs = std::pair<float, int64_t>;

    std::vector<rs>
    knn_search(const float* query, int64_t k, BitsetPtr invalid) const;

    std::vector<rs>
    range_search(const float* query, float radius, BitsetPtr invalid) const;

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
