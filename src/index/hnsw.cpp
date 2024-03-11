
#include "hnsw.h"

#include <fmt/format-inl.h>
#include <hnswlib/hnswlib.h>
#include <spdlog/spdlog.h>

#include <exception>
#include <new>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "../common.h"
#include "../utils.h"
#include "./hnsw_zparameters.h"
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
        return not bitset_->Get(id & ROW_ID_MASK);
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

tl::expected<std::vector<int64_t>, Error>
HNSW::build(const Dataset& base) {
    SlowTaskTimer t("hnsw build");

    try {
        spdlog::debug("index.dim={}, base.dim={}", this->dim_, base.GetDim());

        auto base_dim = base.GetDim();
        CHECK_ARGUMENT(base_dim == dim_,
                       fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));

        int64_t num_elements = base.GetNumElements();
        int64_t max_elements_ = alg_hnsw->getMaxElements();
        if (max_elements_ < num_elements) {
            spdlog::debug("max_elements_={}, num_elements={}", max_elements_, num_elements);
            max_elements_ = num_elements;
            // noexcept even cannot alloc memory
            alg_hnsw->resizeIndex(max_elements_);
        }

        auto ids = base.GetIds();
        auto vectors = base.GetFloat32Vectors();
        std::vector<int64_t> failed_ids;
        for (int64_t i = 0; i < num_elements; ++i) {
            // noexcept runtime
            if (!alg_hnsw->addPoint((const void*)(vectors + i * dim_), ids[i])) {
                spdlog::debug("duplicate point: {}", ids[i]);
                failed_ids.emplace_back(ids[i]);
            }
        }

        return failed_ids;
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::INVALID_ARGUMENT, "failed to build(invalid argument): ", e.what());
    }
}

tl::expected<std::vector<int64_t>, Error>
HNSW::add(const Dataset& base) {
    SlowTaskTimer t("hnsw add", 10);

    try {
        auto base_dim = base.GetDim();
        CHECK_ARGUMENT(base_dim == dim_,
                       fmt::format("base.dim({}) must be equal to index.dim({})", base_dim, dim_));

        int64_t num_elements = base.GetNumElements();
        int64_t max_elements_ = alg_hnsw->getMaxElements();
        if (num_elements + GetNumElements() > max_elements_) {
            spdlog::debug("num_elements={}, index.num_elements, max_elements_={}",
                          num_elements,
                          GetNumElements(),
                          max_elements_);
            if (max_elements_ > EXPANSION_NUM) {
                max_elements_ += EXPANSION_NUM;
            } else {
                max_elements_ *= 2;
            }
            // noexcept even cannot alloc memory
            alg_hnsw->resizeIndex(max_elements_);
        }

        auto ids = base.GetIds();
        auto vectors = base.GetFloat32Vectors();
        std::vector<int64_t> failed_ids;
        for (int64_t i = 0; i < num_elements; ++i) {
            // noexcept runtime
            if (!alg_hnsw->addPoint((const void*)(vectors + i * dim_), ids[i])) {
                spdlog::debug("duplicate point: {}", i);
                failed_ids.push_back(ids[i]);
            }
        }

        return failed_ids;
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::INVALID_ARGUMENT, "failed to add(invalid argument): ", e.what());
    }
}

tl::expected<Dataset, Error>
HNSW::knn_search(const Dataset& query,
                 int64_t k,
                 const std::string& parameters,
                 BitsetPtr invalid) const {
    SlowTaskTimer t("hnsw knnsearch", 10);

    try {
        // check query vector
        CHECK_ARGUMENT(query.GetNumElements() == 1, "query dataset should contain 1 vector only");
        auto vector = query.GetFloat32Vectors();
        int64_t query_dim = query.GetDim();
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));

        // check k
        CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k))
        k = std::min(k, GetNumElements());

        // check search parameters
        auto params = HnswSearchParameters::FromJson(parameters);
        alg_hnsw->setEf(params.ef_search);

        // check filter
        std::shared_ptr<Filter> filter = nullptr;
        if (invalid != nullptr) {
            CHECK_ARGUMENT(
                invalid->Capacity() >= GetNumElements(),
                fmt::format("invalid.capcity({}) must be greater equal than index.size({})",
                            invalid->Capacity(),
                            GetNumElements()));
            filter = std::make_shared<Filter>(invalid);
        }

        // perform search
        std::priority_queue<std::pair<float, size_t>> results;
        double time_cost;
        try {
            Timer t(time_cost);
            results = alg_hnsw->searchKnn((const void*)(vector), k, filter.get());
        } catch (const std::runtime_error& e) {
            LOG_ERROR_AND_RETURNS(ErrorType::INTERNAL_ERROR,
                                  "failed to perofrm knn_search(internalError): ",
                                  e.what());
        }

        // update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            result_queues_[STATSTIC_KNN_TIME].Push(time_cost);
        }

        // return result
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
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              "failed to perform knn_search(invalid argument): ",
                              e.what());
    }
}

tl::expected<Dataset, Error>
HNSW::range_search(const Dataset& query,
                   float radius,
                   const std::string& parameters,
                   BitsetPtr invalid) const {
    SlowTaskTimer t("hnsw rangesearch", 10);

    try {
        // check query vector
        CHECK_ARGUMENT(query.GetNumElements() == 1, "query dataset should contain 1 vector only");
        auto vector = query.GetFloat32Vectors();
        int64_t query_dim = query.GetDim();
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));

        // check radius
        CHECK_ARGUMENT(radius >= 0, fmt::format("radius({}) must be greater equal than 0", radius))

        // check search parameters
        auto params = HnswSearchParameters::FromJson(parameters);
        alg_hnsw->setEf(params.ef_search);

        // check filter
        std::shared_ptr<Filter> filter = nullptr;
        if (invalid != nullptr) {
            CHECK_ARGUMENT(
                invalid->Capacity() >= GetNumElements(),
                fmt::format("invalid.capcity({}) must be greater equal than index.size({})",
                            invalid->Capacity(),
                            GetNumElements()));
            filter = std::make_shared<Filter>(invalid);
        }

        // perform search
        std::priority_queue<std::pair<float, size_t>> results;
        double time_cost;
        try {
            Timer timer(time_cost);
            results = alg_hnsw->searchRange((const void*)(vector), radius, filter.get());
        } catch (std::runtime_error& e) {
            LOG_ERROR_AND_RETURNS(ErrorType::INTERNAL_ERROR,
                                  "failed to perofrm range_search(internalError): ",
                                  e.what());
        }

        // update stats
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            result_queues_[STATSTIC_KNN_TIME].Push(time_cost);
        }

        // return result
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
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              "failed to perform range_search(invalid argument): ",
                              e.what());
    }
}

tl::expected<BinarySet, Error>
HNSW::serialize() const {
    if (GetNumElements() == 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_EMPTY, "failed to serialize: hnsw index is empty");
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
            ErrorType::NO_ENOUGH_MEMORY, "failed to serialize(bad alloc): ", e.what());
    }
}

tl::expected<void, Error>
HNSW::deserialize(const BinarySet& binary_set) {
    SlowTaskTimer t("hnsw deserialize");
    if (this->alg_hnsw->getCurrentElementCount() > 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to deserialize: index is not empty");
    }

    Binary b = binary_set.Get(HNSW_DATA);
    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        std::memcpy(dest, b.data.get() + offset, len);
    };

    try {
        alg_hnsw->loadIndex(func, this->space.get());
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
    }

    return {};
}

tl::expected<void, Error>
HNSW::deserialize(const ReaderSet& reader_set) {
    SlowTaskTimer t("hnsw deserialize");
    if (this->alg_hnsw->getCurrentElementCount() > 0) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to deserialize: index is not empty");
    }

    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        reader_set.Get(HNSW_DATA)->Read(offset, len, dest);
    };

    try {
        alg_hnsw->loadIndex(func, this->space.get());
    } catch (const std::runtime_error& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::READ_ERROR, "failed to deserialize: ", e.what());
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
