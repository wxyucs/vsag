#include "hnsw.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

#include "spdlog/spdlog.h"
#include "vsag/bitset.h"
#include "vsag/errors.h"

namespace {

void
load_float_data(const char* filename,
                float*& data,
                unsigned& num,
                unsigned& dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new float[num * dim * sizeof(float)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
}

void
load_int_data(const char* filename,
              int*& data,
              unsigned& num,
              unsigned& dim) {  // load data with sift10K pattern
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char*)&dim, 4);
    std::cout << "data dimension: " << dim << std::endl;
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    num = (unsigned)(fsize / (dim + 1) / 4);
    data = new int[num * dim * sizeof(int)];

    in.seekg(0, std::ios::beg);
    for (size_t i = 0; i < num; i++) {
        in.seekg(4, std::ios::cur);
        in.read((char*)(data + i * dim), dim * 4);
    }
    in.close();
}

static __attribute__((always_inline)) inline float
naive_l2_dist_calc(const float* p, const float* q, const unsigned dim) {
    float ans = 0;
    for (unsigned i = 0; i < dim; i++) {
        ans += (p[i] - q[i]) * (p[i] - q[i]);
    }
    return ans;
}

void
compute_knn_brute_force(float* base,
                        unsigned base_num,
                        float* query,
                        unsigned query_num,
                        unsigned dim,
                        unsigned k,
                        int*& ground) {
    ground = new int[k * query_num];
#pragma omp parallel for
    for (int i = 0; i < query_num; i++) {
        std::priority_queue<std::pair<float, int>> result_queue;
        for (int j = 0; j < base_num; j++) {
            float dist = naive_l2_dist_calc(base + j * dim, query + i * dim, dim);
            if (result_queue.size() < k)
                result_queue.emplace(dist, j);
            else if (result_queue.top().first > dist) {
                result_queue.pop();
                result_queue.emplace(dist, j);
            }
        }
        int cnt = 0;
        while (!result_queue.empty()) {
            ground[i * k + cnt] = result_queue.top().second;
            result_queue.pop();
            cnt++;
        }
    }
}

bool
isFileExists_ifstream(const char* name) {
    std::ifstream f(name);
    return f.good();
}

std::tuple<std::vector<int64_t>, std::vector<float>>
generate_ids_and_vectors(int64_t num_elements, int64_t dim) {
    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    std::vector<int64_t> ids(num_elements);
    std::vector<float> vectors(dim * num_elements);
    for (int64_t i = 0; i < num_elements; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_elements; ++i) {
        vectors[i] = distrib_real(rng);
    }
    return {ids, vectors};
}
};  // namespace

TEST_CASE("build & add", "[hnsw][ut]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    std::vector<int64_t> ids(1);
    int64_t incorrect_dim = 63;
    std::vector<float> vectors(incorrect_dim);

    vsag::Dataset dataset;
    dataset.Dim(incorrect_dim)
        .NumElements(1)
        .Ids(ids.data())
        .Float32Vectors(vectors.data())
        .Owner(false);

    SECTION("build with incorrect dim") {
        auto result = index->Build(dataset);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("add with incorrect dim") {
        auto result = index->Add(dataset);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("knn_search", "[hnsw][ut]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    const int64_t num_elements = 10;
    auto [ids, vectors] = ::generate_ids_and_vectors(num_elements, dim);

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(1).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    vsag::Dataset query;
    query.NumElements(1).Dim(dim).Float32Vectors(vectors.data()).Owner(false);
    int64_t k = 10;
    nlohmann::json params{
        {"hnsw", {{"ef_search", 100}}},
    };

    SECTION("invalid parameters k is 0") {
        auto result = index->KnnSearch(query, 0, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters k less than 0") {
        auto result = index->KnnSearch(query, -1, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters hnsw not found") {
        nlohmann::json invalid_params{};
        auto result = index->KnnSearch(query, k, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters ef_search not found") {
        nlohmann::json invalid_params{
            {"hnsw", {}},
        };
        auto result = index->KnnSearch(query, k, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid bitset length is less than the size of index") {
        auto invalid_bitset = std::make_shared<vsag::Bitset>(1);
        auto result = index->KnnSearch(query, k, params.dump(), invalid_bitset);
        REQUIRE(result.has_value());
    }

    SECTION("query length is not 1") {
        vsag::Dataset query;
        query.NumElements(2).Dim(dim).Float32Vectors(vectors.data()).Owner(false);
        auto result = index->KnnSearch(query, k, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("dimension not equal") {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim - 1).Float32Vectors(vectors.data()).Owner(false);
        auto result = index->KnnSearch(query, k, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("range_search", "[hnsw][ut]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    const int64_t num_elements = 10;
    auto [ids, vectors] = ::generate_ids_and_vectors(num_elements, dim);

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(1).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    vsag::Dataset query;
    query.NumElements(1).Dim(dim).Float32Vectors(vectors.data()).Owner(false);
    float radius = 9.9f;
    nlohmann::json params{
        {"hnsw", {{"ef_search", 100}}},
    };

    SECTION("invalid parameter radius equals to 0") {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(vectors.data()).Owner(false);
        auto result = index->RangeSearch(query, 0, params.dump());
        REQUIRE(result.has_value());
    }

    SECTION("invalid parameter radius less than 0") {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(vectors.data()).Owner(false);
        auto result = index->RangeSearch(query, -1, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters hnsw not found") {
        nlohmann::json invalid_params{};
        auto result = index->RangeSearch(query, radius, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters ef_search not found") {
        nlohmann::json invalid_params{
            {"hnsw", {}},
        };
        auto result = index->RangeSearch(query, radius, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("query length is not 1") {
        vsag::Dataset query;
        query.NumElements(2).Dim(dim).Float32Vectors(vectors.data()).Owner(false);
        auto result = index->RangeSearch(query, radius, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("serialize empty index", "[hnsw][ut]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    auto result = index->Serialize();
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().type == vsag::ErrorType::INDEX_EMPTY);
}

TEST_CASE("deserialize on not empty index", "[hnsw][ut]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    const int64_t num_elements = 10;
    auto [ids, vectors] = ::generate_ids_and_vectors(num_elements, dim);

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(1).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    auto binary_set = index->Serialize();
    REQUIRE(binary_set.has_value());

    auto voidresult = index->Deserialize(binary_set.value());
    REQUIRE_FALSE(voidresult.has_value());
    REQUIRE(voidresult.error().type == vsag::ErrorType::INDEX_NOT_EMPTY);
}

TEST_CASE("static hnsw", "[hnsw][ut]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction, true);

    const int64_t num_elements = 10;
    auto [ids, vectors] = ::generate_ids_and_vectors(num_elements, dim);

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(9).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    vsag::Dataset one_vector;
    one_vector.Dim(dim)
        .NumElements(1)
        .Ids(ids.data() + 9)
        .Float32Vectors(vectors.data() + 9 * dim)
        .Owner(false);
    result = index->Add(one_vector);
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    nlohmann::json params{
        {"hnsw", {{"ef_search", 100}}},
    };

    auto knn_result = index->KnnSearch(one_vector, 1, params.dump());
    REQUIRE(knn_result.has_value());

    auto range_result = index->RangeSearch(one_vector, 1, params.dump());
    REQUIRE_FALSE(range_result.has_value());
    REQUIRE(range_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    REQUIRE_THROWS(std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(127), max_degree, ef_construction, true));
}