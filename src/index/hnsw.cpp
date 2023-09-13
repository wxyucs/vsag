
#include "hnsw.h"

#include <hnswlib/hnswlib.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>

#include "vsag/binaryset.h"
#include "vsag/utils.h"

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
    int64_t num_elements = base.GetNumElements();
    int64_t dim = base.GetDim();
    auto ids = base.GetIds();
    auto vectors = base.GetFloat32Vectors();
    for (int64_t i = 0; i < num_elements; ++i) {
        alg_hnsw->addPoint((const void*)(vectors + i * dim), ids[i]);
    }
}

void
HNSW::Add(const Dataset& base) {
    int64_t num_elements = base.GetNumElements();
    int64_t dim = base.GetDim();
    auto ids = base.GetIds();
    auto vectors = base.GetFloat32Vectors();
    for (int64_t i = 0; i < num_elements; ++i) {
        alg_hnsw->addPoint((const void*)(vectors + i * dim), ids[i]);
    }
}

Dataset
HNSW::KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) {
    nlohmann::json params = nlohmann::json::parse(parameters);
    if (params.contains("ef_runtime")) {
        alg_hnsw->setEf(params["ef_runtime"]);
    }

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
    result.SetNumElements(num_elements);
    result.SetIds(ids);
    result.SetDistances(dists);

    return std::move(result);
}

BinarySet
HNSW::Serialize() {
    // FIXME: index should save to memory buffer directly
    std::string filename = "/tmp/hnsw-" + std::to_string(random_integer(1'000'000, 9'000'000));
    alg_hnsw->saveIndex(filename);

    std::ifstream file(filename, std::ios::binary);
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::shared_ptr<int8_t[]> fileContents(new int8_t[fileSize]);
    file.read(reinterpret_cast<char*>(fileContents.get()), fileSize);
    file.close();
    std::remove(filename.c_str());

    Binary b{
        .data = fileContents,
        .size = fileSize,
    };
    BinarySet bs;
    bs.Set(HNSW_DATA, b);

    return bs;
}

void
HNSW::Deserialize(const BinarySet& binary_set) {
    // FIXME: index should load directly
    if (this->alg_hnsw->getCurrentElementCount() > 0) {
        throw std::runtime_error("deserialize on existed index");
    }
    std::string filename = "/tmp/hnsw-" + std::to_string(random_integer(1'000'000, 9'000'000));
    std::ofstream file(filename, std::ios::binary);
    Binary b = binary_set.Get(HNSW_DATA);
    file.write((const char*)b.data.get(), b.size);
    file.close();
    alg_hnsw->loadIndex(filename, this->space.get());
    std::remove(filename.c_str());
}

void
HNSW::SetEfRuntime(int64_t ef_runtime) {
    alg_hnsw->setEf(ef_runtime);
}

}  // namespace vsag
