#include "hnsw.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

#include "spdlog/spdlog.h"
#include "vsag/bitset.h"
#include "vsag/errors.h"

namespace {
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
