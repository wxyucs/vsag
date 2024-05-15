#pragma once

#include <nlohmann/json.hpp>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "../common.h"
#include "../utils.h"
#include "vsag/errors.h"
#include "vsag/index.h"

namespace vsag {

class ConjugateGraph {
public:
    ConjugateGraph();

    tl::expected<bool, Error>
    AddNeighbor(int64_t from_tag_id, int64_t to_tag_id);

    tl::expected<uint32_t, Error>
    EnhanceResult(std::priority_queue<std::pair<float, size_t>>& results,
                  const std::function<float(int64_t)>& distance_of_tag) const;

public:
    tl::expected<BinarySet, Error>
    Serialize() const {
        throw std::runtime_error("Conjugate graph doesn't support serialize");
    };

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) {
        throw std::runtime_error("Conjugate graph doesn't support deserialize");
    };

    size_t
    GetMemoryUsage() const;

private:
    const std::unordered_set<int64_t>&
    get_neighbors(int64_t from_tag_id) const;

private:
    uint32_t memory_usage_;

    std::unordered_map<int64_t, std::unordered_set<int64_t>> conjugate_graph_;
};

}  // namespace vsag
