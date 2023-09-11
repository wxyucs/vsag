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

public:
    void
    SetEfRuntime(int64_t ef_runtime);

private:
    std::shared_ptr<hnswlib::HierarchicalNSW> alg_hnsw;
    std::shared_ptr<hnswlib::SpaceInterface> space;
};

}  // namespace vsag
