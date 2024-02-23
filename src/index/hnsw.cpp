
#include "hnsw.h"

#include <fmt/format-inl.h>
#include <hnswlib/hnswlib.h>
#include <spdlog/spdlog.h>

#include <exception>
#include <new>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "../utils.h"
#include "vsag/binaryset.h"
#include "vsag/constants.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"

namespace vsag {

const static int64_t EXPANSION_NUM = 1000000;
const static int64_t DEFAULT_MAX_ELEMENT = 10000;
const static int MINIMAL_M = 8;
const static int MAXIMAL_M = 64;

class Filter : public hnswlib::BaseFilterFunctor {
public:
    Filter(BitsetPtr bitset) : bitset_(bitset) {
    }

    bool
    operator()(hnswlib::labeltype id) override {
        return not bitset_->Get(id);
    }

private:
    BitsetPtr bitset_;
};

inline int64_t
random_integer(int64_t lower_bound, int64_t upper_bound) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distribution(lower_bound, upper_bound);
    return distribution(generator);
}

HNSW::HNSW(std::shared_ptr<hnswlib::SpaceInterface> space_interface, int M, int ef_construction)
    : space(std::move(space_interface)) {
    dim_ = *((size_t*)space->get_dist_func_param());

    M = std::min(std::max(M, MINIMAL_M), MAXIMAL_M);

    if (ef_construction <= 0) {
        throw std::runtime_error(MESSAGE_PARAMETER);
    }

    alg_hnsw = std::make_shared<hnswlib::HierarchicalNSW>(
        space.get(), DEFAULT_MAX_ELEMENT, M, ef_construction);
}

tl::expected<std::vector<int64_t>, index_error>
HNSW::build(const Dataset& base) {
    SlowTaskTimer t("hnsw build");
    std::vector<int64_t> failed_ids;
    spdlog::debug(fmt::format("index.dim={}, base.dim={}", this->dim_, base.GetDim()));
    if (base.GetDim() != dim_) {
        LOG_ERROR_AND_RETURNS(
            index_error_type::dimension_not_equal,
            fmt::format("dimension not equal: add({}) index({})", base.GetDim(), dim_));
    }
    int64_t num_elements = base.GetNumElements();
    int64_t max_elements_ = alg_hnsw->getMaxElements();
    if (max_elements_ < num_elements) {
        max_elements_ = num_elements;
        // noexcept even cannot alloc memory
        alg_hnsw->resizeIndex(max_elements_);
    }

    auto ids = base.GetIds();
    auto vectors = base.GetFloat32Vectors();
    for (int64_t i = 0; i < num_elements; ++i) {
        // noexcept runtime
        if (!alg_hnsw->addPoint((const void*)(vectors + i * dim_), ids[i])) {
            spdlog::debug("duplicate point: {}", ids[i]);
            failed_ids.push_back(ids[i]);
        }
    }

    return failed_ids;
}

tl::expected<std::vector<int64_t>, index_error>
HNSW::add(const Dataset& base) {
    SlowTaskTimer t("hnsw add", 10);
    std::vector<int64_t> failed_ids;
    int64_t num_elements = base.GetNumElements();

    if (base.GetDim() != dim_) {
        LOG_ERROR_AND_RETURNS(
            index_error_type::dimension_not_equal,
            fmt::format("dimension not equal: add({}) index({})", base.GetDim(), dim_));
    }

    auto ids = base.GetIds();
    auto vectors = base.GetFloat32Vectors();
    int64_t max_elements_ = alg_hnsw->getMaxElements();
    if (num_elements + GetNumElements() > max_elements_) {
        if (max_elements_ > EXPANSION_NUM) {
            max_elements_ += EXPANSION_NUM;
        } else {
            max_elements_ *= 2;
        }
        // noexcept even cannot alloc memory
        alg_hnsw->resizeIndex(max_elements_);
    }

    for (int64_t i = 0; i < num_elements; ++i) {
        // noexcept runtime
        if (!alg_hnsw->addPoint((const void*)(vectors + i * dim_), ids[i])) {
            spdlog::debug("duplicate point: {}", i);
            failed_ids.push_back(ids[i]);
        }
    }

    return std::move(failed_ids);
}

tl::expected<Dataset, index_error>
HNSW::knn_search(const Dataset& query,
                 int64_t k,
                 const std::string& parameters,
                 BitsetPtr invalid) const {
    SlowTaskTimer t("hnsw knnsearch", 10);
    nlohmann::json params = nlohmann::json::parse(parameters);

    if (k <= 0) {
        LOG_ERROR_AND_RETURNS(index_error_type::invalid_parameter,
                              fmt::format("invalid parameter: k ({})", k));
    }

    if (!params.contains(INDEX_HNSW) || !params[INDEX_HNSW].contains(HNSW_PARAMETER_EF_RUNTIME)) {
        LOG_ERROR_AND_RETURNS(index_error_type::invalid_parameter,
                              "missing parameter: ",
                              INDEX_HNSW,
                              ".",
                              HNSW_PARAMETER_EF_RUNTIME);
    }

    alg_hnsw->setEf(params[INDEX_HNSW][HNSW_PARAMETER_EF_RUNTIME]);

    k = std::min(k, GetNumElements());

    std::shared_ptr<Filter> filter = nullptr;
    if (invalid != nullptr) {
        if (invalid->Capcity() < GetNumElements()) {
            LOG_ERROR_AND_RETURNS(index_error_type::internal_error,
                                  "number of invalid is less than the size of index");
        }
        filter = std::make_shared<Filter>(invalid);
    }

    int64_t num_elements = query.GetNumElements();
    if (num_elements != 1) {
        LOG_ERROR_AND_RETURNS(index_error_type::internal_error,
                              "number of elements is NOT 1 in query database");
    }
    int64_t dim = query.GetDim();
    if (dim != dim_) {
        LOG_ERROR_AND_RETURNS(index_error_type::dimension_not_equal,
                              fmt::format("dimension not equal: query({}) index({})", dim, dim_));
    }
    auto vectors = query.GetFloat32Vectors();

    std::priority_queue<std::pair<float, size_t>> results;
    for (int64_t i = 0; i < num_elements; ++i) {
        try {
            double time_cost;
            {
                Timer t(time_cost);
                results = alg_hnsw->searchKnn((const void*)(vectors + i * dim_), k, filter.get());
            }

            // stats
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);

                result_queues_[STATSTIC_KNN_TIME].Push(time_cost);
            }
        } catch (std::runtime_error& e) {
            LOG_ERROR_AND_RETURNS(
                index_error_type::internal_error, "failed to call searchKnn: ", e.what());
        }
    }

    Dataset result;
    if (results.size() == 0) {
        result.Dim(0).NumElements(1);
        return result;
    }
    int64_t* ids = new int64_t[results.size()];
    float* dists = new float[results.size()];
    result.Dim(results.size()).NumElements(1).Ids(ids).Distances(dists);
    for (int64_t j = results.size() - 1; j >= 0; --j) {
        dists[j] = results.top().first;
        ids[j] = results.top().second;
        results.pop();
    }

    return std::move(result);
}

tl::expected<Dataset, index_error>
HNSW::range_search(const Dataset& query,
                   float radius,
                   const std::string& parameters,
                   BitsetPtr invalid) const {
    SlowTaskTimer t("hnsw rangesearch", 10);
    nlohmann::json params = nlohmann::json::parse(parameters);

    if (radius < 0) {
        LOG_ERROR_AND_RETURNS(index_error_type::invalid_parameter,
                              fmt::format("invalid parameter: radius ({})", radius));
    }

    if (!params.contains(INDEX_HNSW) || !params[INDEX_HNSW].contains(HNSW_PARAMETER_EF_RUNTIME)) {
        LOG_ERROR_AND_RETURNS(
            index_error_type::invalid_parameter, "missing parameter: ", HNSW_PARAMETER_EF_RUNTIME);
    }

    alg_hnsw->setEf(params[INDEX_HNSW][HNSW_PARAMETER_EF_RUNTIME]);

    int64_t num_elements = query.GetNumElements();
    if (num_elements != 1) {
        LOG_ERROR_AND_RETURNS(index_error_type::internal_error,
                              "number of elements is NOT 1 in query database");
    }
    int64_t dim = query.GetDim();
    if (dim != dim_) {
        LOG_ERROR_AND_RETURNS(index_error_type::dimension_not_equal,
                              fmt::format("dimension not equal: query({}) index({})", dim, dim_));
    }

    std::shared_ptr<Filter> filter = nullptr;
    if (invalid != nullptr) {
        if (invalid->Capcity() < GetNumElements()) {
            LOG_ERROR_AND_RETURNS(index_error_type::internal_error,
                                  "number of invalid is less than the size of index");
        }
        filter = std::make_shared<Filter>(invalid);
    }

    auto vector = query.GetFloat32Vectors();

    std::priority_queue<std::pair<float, size_t>> results;
    try {
        double time_cost;
        {
            Timer timer(time_cost);
            results = alg_hnsw->searchRange((const void*)(vector), radius, filter.get());
        }
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            result_queues_[STATSTIC_RANGE_TIME].Push(time_cost);
        }
    } catch (std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(index_error_type::internal_error, e.what());
    }

    Dataset result;
    if (results.size() == 0) {
        result.Dim(0).NumElements(1);
        return result;
    }
    int64_t* ids = new int64_t[results.size()];
    float* dists = new float[results.size()];
    result.Dim(results.size()).NumElements(1).Ids(ids).Distances(dists);
    for (int64_t j = results.size() - 1; j >= 0; --j) {
        dists[+j] = results.top().first;
        ids[j] = results.top().second;
        results.pop();
    }

    return std::move(result);
}

tl::expected<BinarySet, index_error>
HNSW::serialize() const {
    if (GetNumElements() == 0) {
        LOG_ERROR_AND_RETURNS(index_error_type::index_empty,
                              "failed to serialize: hnsw index is empty");
    }
    SlowTaskTimer t("hnsw serialize");
    size_t num_bytes = alg_hnsw->calcSerializeSize();
    try {
        std::shared_ptr<int8_t[]> bin(new int8_t[num_bytes]);
        alg_hnsw->saveIndex(bin.get());
        Binary b{
            .data = bin,
            .size = num_bytes,
        };
        BinarySet bs;
        bs.Set(HNSW_DATA, b);

        return bs;
    } catch (const std::bad_alloc& e) {
        LOG_ERROR_AND_RETURNS(
            index_error_type::no_enough_memory, "failed to save index: ", e.what());
    }
}

tl::expected<void, index_error>
HNSW::deserialize(const BinarySet& binary_set) {
    SlowTaskTimer t("hnsw deserialize");
    if (this->alg_hnsw->getCurrentElementCount() > 0) {
        LOG_ERROR_AND_RETURNS(index_error_type::index_not_empty,
                              "failed to deserialize: index is not empty");
    }

    Binary b = binary_set.Get(HNSW_DATA);
    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        std::memcpy(dest, b.data.get() + offset, len);
    };

    try {
        alg_hnsw->loadIndex(func, this->space.get());
    } catch (std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(index_error_type::read_error, "failed to deserialize: ", e.what());
    }

    return {};
}

tl::expected<void, index_error>
HNSW::deserialize(const ReaderSet& reader_set) {
    SlowTaskTimer t("hnsw deserialize");
    if (this->alg_hnsw->getCurrentElementCount() > 0) {
        LOG_ERROR_AND_RETURNS(index_error_type::index_not_empty,
                              "failed to deserialize: index is not empty");
    }

    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        reader_set.Get(HNSW_DATA)->Read(offset, len, dest);
    };

    try {
        alg_hnsw->loadIndex(func, this->space.get());
    } catch (std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(index_error_type::read_error, "failed to deserialize: ", e.what());
    }

    return {};
}

void
HNSW::SetEfRuntime(int64_t ef_runtime) {
    alg_hnsw->setEf(ef_runtime);
}

std::string
HNSW::GetStats() const {
    nlohmann::json j;
    j[STATSTIC_DATA_NUM] = GetNumElements();
    j[STATSTIC_INDEX_NAME] = INDEX_HNSW;
    j[STATSTIC_MEMORY] = GetMemoryUsage();

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        for (auto& item : result_queues_) {
            j[item.first] = item.second.GetAvgResult();
        }
    }
    return j.dump();
}

}  // namespace vsag
