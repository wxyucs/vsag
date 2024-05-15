#pragma once

#include <hnswlib/hnswlib.h>

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../common.h"
#include "../conjugate_graph/conjugate_graph.h"
#include "../logger.h"
#include "../utils.h"
#include "vsag/binaryset.h"
#include "vsag/errors.h"
#include "vsag/index.h"
#include "vsag/readerset.h"

namespace vsag {

class HNSW : public Index {
public:
    HNSW(std::shared_ptr<hnswlib::SpaceInterface> space_interface,
         int M,
         int ef_construction,
         bool use_static = false,
         bool use_reversed_edges = false,
         bool use_conjugate_graph = false);

    tl::expected<std::vector<int64_t>, Error>
    Build(const Dataset& base) override {
        SAFE_CALL(return this->build(base));
    }

    tl::expected<std::vector<int64_t>, Error>
    Add(const Dataset& base) override {
        SAFE_CALL(return this->add(base));
    }

    tl::expected<bool, Error>
    Remove(int64_t id) override {
        SAFE_CALL(return this->remove(id));
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

    tl::expected<uint32_t, Error>
    Feedback(const Dataset& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id = std::numeric_limits<int64_t>::max()) override {
        SAFE_CALL(return this->feedback(query, k, parameters, global_optimum_tag_id));
    };

public:
    tl::expected<BinarySet, Error>
    Serialize() const override {
        SAFE_CALL(return this->serialize());
    }

    tl::expected<void, Error>
    Serialize(std::ostream& out_stream) override {
        SAFE_CALL(return this->serialize(out_stream));
    }

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(return this->deserialize(binary_set));
    }

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        SAFE_CALL(return this->deserialize(reader_set));
    }

    tl::expected<void, Error>
    Deserialize(std::istream& in_stream, int64_t length) override {
        SAFE_CALL(return this->deserialize(in_stream, length));
    }

public:
    int64_t
    GetNumElements() const override {
        return alg_hnsw->getCurrentElementCount() - alg_hnsw->getDeletedCount();
    }

    int64_t
    GetMemoryUsage() const override {
        if (use_conjugate_graph_)
            return alg_hnsw->calcSerializeSize() + conjugate_graph_->GetMemoryUsage();
        else
            return alg_hnsw->calcSerializeSize();
    }

    std::string
    GetStats() const override;

    // used to test the integrity of graphs, used only in UT.
    bool
    CheckGraphIntegrity() const;

private:
    tl::expected<std::vector<int64_t>, Error>
    build(const Dataset& base);

    tl::expected<std::vector<int64_t>, Error>
    add(const Dataset& base);

    tl::expected<bool, Error>
    remove(int64_t id);

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

    tl::expected<uint32_t, Error>
    feedback(const Dataset& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id);

    tl::expected<uint32_t, Error>
    feedback(const Dataset& result, int64_t global_optimum_tag_id, int64_t k);

    tl::expected<Dataset, Error>
    brute_force(const Dataset& query, int64_t k);

    tl::expected<BinarySet, Error>
    serialize() const;

    tl::expected<void, Error>
    serialize(std::ostream& out_stream);

    tl::expected<void, Error>
    deserialize(const BinarySet& binary_set);

    tl::expected<void, Error>
    deserialize(const ReaderSet& binary_set);

    tl::expected<void, Error>
    deserialize(std::istream& in_stream, int64_t length);

    BinarySet
    empty_binaryset() const;

private:
    std::shared_ptr<hnswlib::AlgorithmInterface<float>> alg_hnsw;
    std::shared_ptr<hnswlib::SpaceInterface> space;

    bool use_conjugate_graph_;
    std::shared_ptr<ConjugateGraph> conjugate_graph_;

    int64_t dim_;
    bool use_static_ = false;
    bool empty_index_ = false;
    bool use_reversed_edges_ = false;

    mutable std::mutex stats_mutex_;
    mutable std::map<std::string, WindowResultQueue> result_queues_;
};

}  // namespace vsag
