
#include "hnsw.h"

#include <hnswlib/hnswlib.h>

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

#include "../utils.h"
#include "spdlog/spdlog.h"
#include "vsag/binaryset.h"
#include "vsag/constants.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/utils.h"

const static int64_t EXPANSION_NUM = 1000000;

namespace vsag {

inline int64_t
random_integer(int64_t lower_bound, int64_t upper_bound) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distribution(lower_bound, upper_bound);
    return distribution(generator);
}

HNSW::HNSW(std::shared_ptr<hnswlib::SpaceInterface> spaceInterface,
           int max_elements,
           int M,
           int ef_construction)
    : space(std::move(spaceInterface)) {
    alg_hnsw =
        std::make_shared<hnswlib::HierarchicalNSW>(space.get(), max_elements, M, ef_construction);
}

tl::expected<int64_t, index_error>
HNSW::Build(const Dataset& base) {
    SlowTaskTimer t("hnsw build");
    int64_t num_elements = base.GetNumElements();
    int64_t dim = base.GetDim();
    int64_t max_elements_ = alg_hnsw->getMaxElements();
    if (max_elements_ < num_elements) {
        max_elements_ = num_elements;
        try {
            alg_hnsw->resizeIndex(max_elements_);
        } catch (std::runtime_error e) {
            spdlog::error(std::string("failed to resize index: ") + e.what());
            return tl::unexpected(index_error::no_enough_memory);
        }
    }

    auto ids = base.GetIds();
    auto vectors = base.GetFloat32Vectors();
    for (int64_t i = 0; i < num_elements; ++i) {
        try {
            alg_hnsw->addPoint((const void*)(vectors + i * dim), ids[i]);
        } catch (std::runtime_error e) {
            spdlog::error(std::string("failed to add points: ") + e.what());
            return tl::unexpected(index_error::internal_error);
        }
    }

    return this->GetNumElements();
}

tl::expected<int64_t, index_error>
HNSW::Add(const Dataset& base) {
    SlowTaskTimer t("hnsw add", 10);
    int64_t num_elements = base.GetNumElements();
    int64_t dim = base.GetDim();
    int64_t index_dim = *((size_t*)alg_hnsw->dist_func_param_);
    if (dim != index_dim) {
        spdlog::error("dimension not equal: add(" + std::to_string(dim) + ") index(" +
                      std::to_string(index_dim) + ")");
        return tl::unexpected(index_error::dimension_not_equal);
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
        try {
            alg_hnsw->resizeIndex(max_elements_);
        } catch (std::runtime_error e) {
            spdlog::error(std::string("failed to resize index: ") + e.what());
            return tl::unexpected(index_error::no_enough_memory);
        }
    }

    for (int64_t i = 0; i < num_elements; ++i) {
        try {
            alg_hnsw->addPoint((const void*)(vectors + i * dim), ids[i]);
        } catch (std::runtime_error e) {
            spdlog::error(std::string("failed to add points: ") + e.what());
            return tl::unexpected(index_error::internal_error);
        }
    }

    return base.GetNumElements();
}

tl::expected<Dataset, index_error>
HNSW::KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) const {
    SlowTaskTimer t("hnsw search", 10);
    nlohmann::json params = nlohmann::json::parse(parameters);
    if (params.contains("hnsw") and params["hnsw"].contains("ef_runtime")) {
        alg_hnsw->setEf(params["hnsw"]["ef_runtime"]);
    }
    k = std::min(k, GetNumElements());

    int64_t num_elements = query.GetNumElements();
    int64_t dim = query.GetDim();
    int64_t index_dim = *((size_t*)alg_hnsw->dist_func_param_);
    if (dim != index_dim) {
        spdlog::error("dimension not equal: query(" + std::to_string(dim) + ") index(" +
                      std::to_string(index_dim) + ")");
        return tl::unexpected(index_error::dimension_not_equal);
    }
    auto vectors = query.GetFloat32Vectors();

    Dataset result;
    int64_t* ids = new int64_t[num_elements * k];
    float* dists = new float[num_elements * k];
    for (int64_t i = 0; i < num_elements; ++i) {
        try {
            std::priority_queue<std::pair<float, size_t>> results =
                alg_hnsw->searchKnn((const void*)(vectors + i * dim), k);
            for (int64_t j = k - 1; j >= 0; --j) {
                dists[i * k + j] = results.top().first;
                ids[i * k + j] = results.top().second;
                results.pop();
            }
        } catch (std::runtime_error e) {
            return tl::unexpected(index_error::internal_error);
        }
    }
    result.SetDim(k);
    result.SetNumElements(num_elements);
    result.SetIds(ids);
    result.SetDistances(dists);

    return std::move(result);
}

tl::expected<BinarySet, index_error>
HNSW::Serialize() const {
    SlowTaskTimer t("hnsw serialize");
    size_t num_bytes = alg_hnsw->calcSerializeSize();
    // std::cout << "num_bytes: " << std::to_string(num_bytes) << std::endl;
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
        return tl::unexpected(index_error::no_enough_memory);
    }
}

tl::expected<void, index_error>
HNSW::Deserialize(const BinarySet& binary_set) {
    SlowTaskTimer t("hnsw deserialize");
    if (this->alg_hnsw->getCurrentElementCount() > 0) {
        spdlog::error("failed to deserialize: index is not empty");
        return tl::unexpected(index_error::index_not_empty);
    }

    Binary b = binary_set.Get(HNSW_DATA);
    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        std::memcpy(dest, b.data.get() + offset, len);
    };

    alg_hnsw->loadIndex(func, this->space.get());

    return {};
}

tl::expected<void, index_error>
HNSW::Deserialize(const ReaderSet& reader_set) {
    SlowTaskTimer t("hnsw deserialize");
    if (this->alg_hnsw->getCurrentElementCount() > 0) {
        spdlog::error("failed to deserialize: index is not empty");
        return tl::unexpected(index_error::index_not_empty);
    }

    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        reader_set.Get(HNSW_DATA)->Read(offset, len, dest);
    };

    alg_hnsw->loadIndex(func, this->space.get());

    return {};
}

void
HNSW::SetEfRuntime(int64_t ef_runtime) {
    alg_hnsw->setEf(ef_runtime);
}

}  // namespace vsag
