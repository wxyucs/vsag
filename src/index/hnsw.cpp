
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
#include "vsag/binaryset.h"
#include "vsag/constants.h"
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

void
HNSW::Build(const Dataset& base) {
    SlowTaskTimer t("hnsw build");
    int64_t num_elements = base.GetNumElements();
    int64_t dim = base.GetDim();
    int64_t max_elements_ = alg_hnsw->getMaxElements();
    if (max_elements_ < num_elements) {
        max_elements_ = num_elements;
        alg_hnsw->resizeIndex(max_elements_);
    }

    auto ids = base.GetIds();
    auto vectors = base.GetFloat32Vectors();
    for (int64_t i = 0; i < num_elements; ++i) {
        alg_hnsw->addPoint((const void*)(vectors + i * dim), ids[i]);
    }
}

void
HNSW::Add(const Dataset& base) {
    SlowTaskTimer t("hnsw add", 10);
    int64_t num_elements = base.GetNumElements();
    int64_t dim = base.GetDim();
    auto ids = base.GetIds();
    auto vectors = base.GetFloat32Vectors();
    int64_t max_elements_ = alg_hnsw->getMaxElements();
    if (num_elements + GetNumElements() > max_elements_) {
        if (max_elements_ > EXPANSION_NUM) {
            max_elements_ += EXPANSION_NUM;
        } else {
            max_elements_ *= 2;
        }
        alg_hnsw->resizeIndex(max_elements_);
    }

    for (int64_t i = 0; i < num_elements; ++i) {
        alg_hnsw->addPoint((const void*)(vectors + i * dim), ids[i]);
    }
}

Dataset
HNSW::KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) {
    SlowTaskTimer t("hnsw search", 10);
    nlohmann::json params = nlohmann::json::parse(parameters);
    if (params.contains("hnsw") and params["hnsw"].contains("ef_runtime")) {
        alg_hnsw->setEf(params["hnsw"]["ef_runtime"]);
    }
    k = std::min(k, GetNumElements());

    int64_t num_elements = query.GetNumElements();
    int64_t dim = query.GetDim();
    auto vectors = query.GetFloat32Vectors();

    Dataset result;
    int64_t* ids = new int64_t[num_elements * k];
    float* dists = new float[num_elements * k];
    for (int64_t i = 0; i < num_elements; ++i) {
        std::priority_queue<std::pair<float, size_t>> results =
            alg_hnsw->searchKnn((const void*)(vectors + i * dim), k);
        for (int64_t j = k - 1; j >= 0; --j) {
            dists[i * k + j] = results.top().first;
            ids[i * k + j] = results.top().second;
            results.pop();
        }
    }
    result.SetDim(k);
    result.SetNumElements(num_elements);
    result.SetIds(ids);
    result.SetDistances(dists);

    return std::move(result);
}

BinarySet
HNSW::Serialize() {
    SlowTaskTimer t("hnsw serialize");
    size_t num_bytes = alg_hnsw->calcSerializeSize();
    // std::cout << "num_bytes: " << std::to_string(num_bytes) << std::endl;
    std::shared_ptr<int8_t[]> bin(new int8_t[num_bytes]);
    alg_hnsw->saveIndex(bin.get());

    Binary b{
        .data = bin,
        .size = num_bytes,
    };
    BinarySet bs;
    bs.Set(HNSW_DATA, b);

    return bs;
}

void
HNSW::Deserialize(const BinarySet& binary_set) {
    SlowTaskTimer t("hnsw deserialize");
    if (this->alg_hnsw->getCurrentElementCount() > 0) {
        throw std::runtime_error("deserialize on existed index");
    }

    Binary b = binary_set.Get(HNSW_DATA);
    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        std::memcpy(dest, b.data.get() + offset, len);
    };

    alg_hnsw->loadIndex(func, this->space.get());
}

void
HNSW::Deserialize(const ReaderSet& reader_set) {
    SlowTaskTimer t("hnsw deserialize");
    if (this->alg_hnsw->getCurrentElementCount() > 0) {
        throw std::runtime_error("deserialize on existed index");
    }

    auto func = [&](uint64_t offset, uint64_t len, void* dest) -> void {
        reader_set.Get(HNSW_DATA)->Read(offset, len, dest);
    };

    alg_hnsw->loadIndex(func, this->space.get());
}

void
HNSW::SetEfRuntime(int64_t ef_runtime) {
    alg_hnsw->setEf(ef_runtime);
}

}  // namespace vsag
