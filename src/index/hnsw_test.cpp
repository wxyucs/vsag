#include "hnsw.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

#include "fixtures.h"
#include "spdlog/spdlog.h"
#include "vsag/bitset.h"
#include "vsag/errors.h"

TEST_CASE("build & add", "[ut][hnsw]") {
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

TEST_CASE("knn_search", "[ut][hnsw]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    const int64_t num_elements = 10;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

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

TEST_CASE("range_search", "[ut][hnsw]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    const int64_t num_elements = 10;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

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

TEST_CASE("serialize empty index", "[ut][hnsw]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    SECTION("serialize to binaryset") {
        auto result = index->Serialize();
        REQUIRE(result.has_value());
        REQUIRE(result.value().Contains(vsag::BLANK_INDEX));
    }

    SECTION("serialize to fstream") {
        fixtures::temp_dir dir("hnsw_test_serialize_empty_index");
        std::fstream out_stream(dir.path + "empty_index.bin", std::ios::out | std::ios::binary);
        auto result = index->Serialize(out_stream);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INDEX_EMPTY);
    }
}

TEST_CASE("deserialize on not empty index", "[ut][hnsw]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    const int64_t num_elements = 10;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(1).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    SECTION("serialize to binaryset") {
        auto binary_set = index->Serialize();
        REQUIRE(binary_set.has_value());

        auto voidresult = index->Deserialize(binary_set.value());
        REQUIRE_FALSE(voidresult.has_value());
        REQUIRE(voidresult.error().type == vsag::ErrorType::INDEX_NOT_EMPTY);
    }

    SECTION("serialize to fstream") {
        fixtures::temp_dir dir("hnsw_test_deserialize_on_not_empty_index");
        std::fstream out_stream(dir.path + "index.bin", std::ios::out | std::ios::binary);
        auto serialize_result = index->Serialize(out_stream);
        REQUIRE(serialize_result.has_value());
        out_stream.close();

        std::fstream in_stream(dir.path + "index.bin", std::ios::in | std::ios::binary);
        in_stream.seekg(0, std::ios::end);
        auto length = in_stream.tellg();
        in_stream.seekg(0, std::ios::beg);
        auto voidresult = index->Deserialize(in_stream, length);
        REQUIRE_FALSE(voidresult.has_value());
        REQUIRE(voidresult.error().type == vsag::ErrorType::INDEX_NOT_EMPTY);
        in_stream.close();
    }
}

TEST_CASE("static hnsw", "[ut][hnsw]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction, true);

    const int64_t num_elements = 10;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

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
