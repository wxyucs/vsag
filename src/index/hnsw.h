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

#include "../common.h"
#include "../utils.h"
#include "vsag/errors.h"
#include "vsag/index.h"
#include "vsag/readerset.h"

namespace vsag {
class HNSW : public Index {
public:
    HNSW(std::shared_ptr<hnswlib::SpaceInterface> space_interface, int M, int ef_construction);

    tl::expected<std::vector<int64_t>, Error>
    Build(const Dataset& base) override {
        SAFE_CALL(return this->build(base));
    }

    tl::expected<std::vector<int64_t>, Error>
    Add(const Dataset& base) override {
        SAFE_CALL(return this->add(base));
    }

    tl::expected<Dataset, Error>
    KnnSearch(const Dataset& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        SAFE_CALL(return this->knn_search(query, k, parameters, invalid));
    }

    tl::expected<Dataset, Error>
    RangeSearch(const Dataset& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid = nullptr) const override {
        SAFE_CALL(return this->range_search(query, radius, parameters, invalid));
    }

public:
    tl::expected<BinarySet, Error>
    Serialize() const override {
        SAFE_CALL(return this->serialize());
    }

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(return this->deserialize(binary_set));
    }

    tl::expected<void, Error>
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
    tl::expected<std::vector<int64_t>, Error>
    build(const Dataset& base);

    tl::expected<std::vector<int64_t>, Error>
    add(const Dataset& base);

    tl::expected<Dataset, Error>
    knn_search(const Dataset& query,
               int64_t k,
               const std::string& parameters,
               BitsetPtr invalid = nullptr) const;

    tl::expected<Dataset, Error>
    range_search(const Dataset& query,
                 float radius,
                 const std::string& parameters,
                 BitsetPtr invalid = nullptr) const;

    tl::expected<BinarySet, Error>
    serialize() const;

    tl::expected<void, Error>
    deserialize(const BinarySet& binary_set);

    tl::expected<void, Error>
    deserialize(const ReaderSet& binary_set);

private:
    std::shared_ptr<hnswlib::HierarchicalNSW> alg_hnsw;
    std::shared_ptr<hnswlib::SpaceInterface> space;

    int64_t dim_;

    mutable std::mutex stats_mutex_;
    mutable std::map<std::string, WindowResultQueue> result_queues_;
};

}  // namespace vsag
