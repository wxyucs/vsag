#pragma once

#include <hnswlib/hnswlib.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../utils.h"
#include "vsag/errors.h"
#include "vsag/index.h"
#include "vsag/readerset.h"

namespace vsag {

#define SAFE_CALL(stmt)                                                                       \
    try {                                                                                     \
        stmt;                                                                                 \
    } catch (const std::exception& e) {                                                       \
        LOG_ERROR_AND_RETURNS(index_error_type::internal_error, "unknown error: ", e.what()); \
    }

class HNSW : public Index {
public:
    HNSW(std::shared_ptr<hnswlib::SpaceInterface> space_interface, int M, int ef_construction);

    tl::expected<std::vector<int64_t>, index_error>
    Build(const Dataset& base) override {
        SAFE_CALL(return this->build(base));
    }

    tl::expected<std::vector<int64_t>, index_error>
    Add(const Dataset& base) override {
        SAFE_CALL(return this->add(base));
    }

    tl::expected<Dataset, index_error>
    KnnSearch(const Dataset& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        SAFE_CALL(return this->knn_search(query, k, parameters, invalid));
    }

    tl::expected<Dataset, index_error>
    RangeSearch(const Dataset& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid = nullptr) const override {
        SAFE_CALL(return this->range_search(query, radius, parameters, invalid));
    }

public:
    tl::expected<BinarySet, index_error>
    Serialize() const override {
        SAFE_CALL(return this->serialize());
    }

    tl::expected<void, index_error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(return this->deserialize(binary_set));
    }

    tl::expected<void, index_error>
    Deserialize(const ReaderSet& reader_set) override {
        SAFE_CALL(return this->deserialize(reader_set));
    }

public:
    int64_t
    GetNumElements() const override {
        return alg_hnsw->cur_element_count;
    }

    int64_t
    GetMemoryUsage() const override {
        return alg_hnsw->calcSerializeSize();
    }

    std::string
    GetStats() const override;

public:
    void
    SetEfRuntime(int64_t ef_runtime);

private:
    tl::expected<std::vector<int64_t>, index_error>
    build(const Dataset& base);

    tl::expected<std::vector<int64_t>, index_error>
    add(const Dataset& base);

    tl::expected<Dataset, index_error>
    knn_search(const Dataset& query,
               int64_t k,
               const std::string& parameters,
               BitsetPtr invalid = nullptr) const;

    tl::expected<Dataset, index_error>
    range_search(const Dataset& query,
                 float radius,
                 const std::string& parameters,
                 BitsetPtr invalid = nullptr) const;

    tl::expected<BinarySet, index_error>
    serialize() const;

    tl::expected<void, index_error>
    deserialize(const BinarySet& binary_set);

    tl::expected<void, index_error>
    deserialize(const ReaderSet& binary_set);

private:
    std::shared_ptr<hnswlib::HierarchicalNSW> alg_hnsw;
    std::shared_ptr<hnswlib::SpaceInterface> space;

    int64_t dim_;

    mutable std::mutex stats_mutex_;
    mutable std::map<std::string, WindowResultQueue> result_queues_;
};

}  // namespace vsag
