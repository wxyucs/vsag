#pragma once

#include <hnswlib/hnswlib.h>

#include <cstdint>
#include <memory>
#include <queue>
#include <vector>

#include "vsag/index.h"

namespace vsag {

class HNSW : public Index {
public:
    HNSW(std::shared_ptr<hnswlib::SpaceInterface> spaceInterface,
         int max_elements,
         int M,
         int ef_construction);

    tl::expected<int64_t, index_error>
    Build(const Dataset& base) override;

    tl::expected<int64_t, index_error>
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

public:
    int64_t
    GetNumElements() const override {
        return alg_hnsw->cur_element_count;
    }

    int64_t
    GetMemoryUsage() const override {
        return alg_hnsw->calcSerializeSize();
    }

public:
    void
    SetEfRuntime(int64_t ef_runtime);

private:
    std::shared_ptr<hnswlib::HierarchicalNSW> alg_hnsw;
    std::shared_ptr<hnswlib::SpaceInterface> space;
};

}  // namespace vsag
