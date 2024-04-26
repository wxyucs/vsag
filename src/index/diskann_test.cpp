#include "diskann.h"

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <tuple>
#include <vector>

#include "distance.h"
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

TEST_CASE("build", "[diskann][ut]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t ef_construction = 100;
    int64_t max_degree = 12;
    float pq_sample_rate = 1.0f;
    size_t pq_dims = 16;
    auto index = std::make_shared<vsag::DiskANN>(diskann::Metric::L2,
                                                 "float32",
                                                 ef_construction,
                                                 max_degree,
                                                 pq_sample_rate,
                                                 pq_dims,
                                                 dim,
                                                 false,
                                                 false,
                                                 false);

    int64_t num_elements = 10;
    auto [ids, vectors] = generate_ids_and_vectors(num_elements, dim);

    SECTION("build with incorrect dim") {
        int64_t incorrect_dim = dim - 1;
        vsag::Dataset dataset;
        dataset.Dim(incorrect_dim)
            .NumElements(num_elements)
            .Ids(ids.data())
            .Float32Vectors(vectors.data())
            .Owner(false);
        auto result = index->Build(dataset);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("number of elements less than 2") {
        vsag::Dataset dataset;
        dataset.Dim(dim).NumElements(1).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);
        auto result = index->Build(dataset);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("build twice") {
        vsag::Dataset dataset;
        dataset.Dim(dim)
            .NumElements(10)
            .Ids(ids.data())
            .Float32Vectors(vectors.data())
            .Owner(false);
        REQUIRE(index->Build(dataset).has_value());

        auto result = index->Build(dataset);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::BUILD_TWICE);
    }
}

TEST_CASE("knn_search", "[diskann][ut]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t ef_construction = 100;
    int64_t max_degree = 12;
    float pq_sample_rate = 1.0f;
    size_t pq_dims = 16;
    auto index = std::make_shared<vsag::DiskANN>(diskann::Metric::L2,
                                                 "float32",
                                                 ef_construction,
                                                 max_degree,
                                                 pq_sample_rate,
                                                 pq_dims,
                                                 dim,
                                                 false,
                                                 false,
                                                 false);

    int64_t num_elements = 100;
    auto [ids, vectors] = generate_ids_and_vectors(num_elements, dim);

    vsag::Dataset dataset;
    dataset.Dim(dim)
        .NumElements(num_elements)
        .Ids(ids.data())
        .Float32Vectors(vectors.data())
        .Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    vsag::Dataset query;
    query.Dim(dim).NumElements(1).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);
    int64_t k = 10;
    nlohmann::json params{{"diskann", {{"ef_search", 100}, {"beam_search", 4}, {"io_limit", 200}}}};

    SECTION("index empty") {
        auto empty_index = std::make_shared<vsag::DiskANN>(diskann::Metric::L2,
                                                           "float32",
                                                           ef_construction,
                                                           max_degree,
                                                           pq_sample_rate,
                                                           pq_dims,
                                                           dim,
                                                           false,
                                                           false,
                                                           false);
        auto result = empty_index->KnnSearch(query, k, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INDEX_EMPTY);
    }

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

    SECTION("dimension not equal") {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim - 1).Float32Vectors(vectors.data()).Owner(false);
        auto result = index->KnnSearch(query, k, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid bitset length is less than the size of index") {
        auto invalid_bitset = std::make_shared<vsag::Bitset>(1);
        auto result = index->KnnSearch(query, k, params.dump(), invalid_bitset);
        REQUIRE(result.has_value());
    }

    SECTION("invalid parameters diskann not found") {
        nlohmann::json invalid_params{};
        auto result = index->KnnSearch(query, k, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters beam_search not found") {
        nlohmann::json invalid_params{{"diskann", {{"ef_search", 100}, {"io_limit", 200}}}};

        auto result = index->KnnSearch(query, k, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters io_limit not found") {
        nlohmann::json invalid_params{{"diskann", {{"ef_search", 100}, {"beam_search", 4}}}};

        auto result = index->KnnSearch(query, k, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters ef_search not found") {
        nlohmann::json invalid_params{{"diskann", {{"beam_search", 4}, {"io_limit", 200}}}};

        auto result = index->KnnSearch(query, k, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("range_search", "[diskann][ut]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t ef_construction = 100;
    int64_t max_degree = 12;
    float pq_sample_rate = 1.0f;
    size_t pq_dims = 16;
    auto index = std::make_shared<vsag::DiskANN>(diskann::Metric::L2,
                                                 "float32",
                                                 ef_construction,
                                                 max_degree,
                                                 pq_sample_rate,
                                                 pq_dims,
                                                 dim,
                                                 false,
                                                 false,
                                                 false);

    int64_t num_elements = 100;
    auto [ids, vectors] = generate_ids_and_vectors(num_elements, dim);

    vsag::Dataset dataset;
    dataset.Dim(dim)
        .NumElements(num_elements)
        .Ids(ids.data())
        .Float32Vectors(vectors.data())
        .Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    vsag::Dataset query;
    query.Dim(dim).NumElements(1).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);
    float radius = 9.9f;
    nlohmann::json params{{"diskann", {{"ef_search", 100}, {"beam_search", 4}, {"io_limit", 200}}}};

    SECTION("index empty") {
        auto empty_index = std::make_shared<vsag::DiskANN>(diskann::Metric::L2,
                                                           "float32",
                                                           ef_construction,
                                                           max_degree,
                                                           pq_sample_rate,
                                                           pq_dims,
                                                           dim,
                                                           false,
                                                           false,
                                                           false);
        auto result = empty_index->RangeSearch(query, radius, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INDEX_EMPTY);
    }

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

    SECTION("dimension not equal") {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim - 1).Float32Vectors(vectors.data()).Owner(false);
        auto result = index->RangeSearch(query, radius, params.dump());
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

    SECTION("invalid bitset length is less than the size of index") {
        auto invalid_bitset = std::make_shared<vsag::Bitset>(1);
        auto result = index->RangeSearch(query, radius, params.dump(), invalid_bitset);
        REQUIRE(result.has_value());
    }

    SECTION("invalid parameters diskann not found") {
        nlohmann::json invalid_params{};
        auto result = index->RangeSearch(query, radius, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters beam_search not found") {
        nlohmann::json invalid_params{{"diskann", {{"ef_search", 100}, {"io_limit", 200}}}};

        auto result = index->RangeSearch(query, radius, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters io_limit not found") {
        nlohmann::json invalid_params{{"diskann", {{"ef_search", 100}, {"beam_search", 4}}}};

        auto result = index->RangeSearch(query, radius, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters ef_search not found") {
        nlohmann::json invalid_params{{"diskann", {{"beam_search", 4}, {"io_limit", 200}}}};

        auto result = index->RangeSearch(query, radius, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("serialize empty index", "[diskann][ut]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t ef_construction = 100;
    int64_t max_degree = 12;
    float pq_sample_rate = 1.0f;
    size_t pq_dims = 16;
    auto index = std::make_shared<vsag::DiskANN>(diskann::Metric::L2,
                                                 "float32",
                                                 ef_construction,
                                                 max_degree,
                                                 pq_sample_rate,
                                                 pq_dims,
                                                 dim,
                                                 false,
                                                 false,
                                                 false);

    auto result = index->Serialize();
    REQUIRE(result.has_value());
}

TEST_CASE("deserialize on not empty index", "[diskann][ut]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t ef_construction = 100;
    int64_t max_degree = 12;
    float pq_sample_rate = 1.0f;
    size_t pq_dims = 16;
    auto index = std::make_shared<vsag::DiskANN>(diskann::Metric::L2,
                                                 "float32",
                                                 ef_construction,
                                                 max_degree,
                                                 pq_sample_rate,
                                                 pq_dims,
                                                 dim,
                                                 false,
                                                 false,
                                                 false);

    int64_t num_elements = 100;
    auto [ids, vectors] = generate_ids_and_vectors(num_elements, dim);

    vsag::Dataset dataset;
    dataset.Dim(dim)
        .NumElements(num_elements)
        .Ids(ids.data())
        .Float32Vectors(vectors.data())
        .Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    auto binary_set = index->Serialize();
    REQUIRE(binary_set.has_value());

    auto voidresult = index->Deserialize(binary_set.value());
    REQUIRE_FALSE(voidresult.has_value());
    REQUIRE(voidresult.error().type == vsag::ErrorType::INDEX_NOT_EMPTY);
}

TEST_CASE("split building process", "[diskann][ut]") {
    spdlog::set_level(spdlog::level::debug);
    int64_t dim = 128;
    int64_t ef_construction = 100;
    int64_t max_degree = 12;
    float pq_sample_rate = 1.0f;
    size_t pq_dims = 16;

    int64_t num_elements = 1000;
    auto [ids, vectors] = generate_ids_and_vectors(num_elements, dim);

    vsag::Dataset dataset;
    dataset.Dim(dim)
        .NumElements(num_elements)
        .Ids(ids.data())
        .Float32Vectors(vectors.data())
        .Owner(false);

    vsag::BinarySet binary_set;
    vsag::BuildStatus status;
    std::shared_ptr<vsag::DiskANN> partial_index;
    double partial_time = 0;
    {
        vsag::Timer timer(partial_time);
        do {
            partial_index = std::make_shared<vsag::DiskANN>(diskann::Metric::L2,
                                                            "float32",
                                                            ef_construction,
                                                            max_degree,
                                                            pq_sample_rate,
                                                            pq_dims,
                                                            dim,
                                                            false,
                                                            false,
                                                            false);
            binary_set = partial_index->ContinueBuild(dataset, binary_set).value();
            vsag::Binary status_binary = binary_set.Get("status");
            memcpy(&status, status_binary.data.get(), status_binary.size);
        } while (status != vsag::BuildStatus::FINISH);
    }

    nlohmann::json parameters{
        {"diskann", {{"ef_search", 10}, {"beam_search", 4}, {"io_limit", 20}}}};
    float correct = 0;
    for (int i = 0; i < num_elements; i++) {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(vectors.data() + i * dim).Owner(false);
        int64_t k = 2;
        if (auto result = partial_index->KnnSearch(query, k, parameters.dump());
            result.has_value()) {
            if (result->GetNumElements() == 1) {
                REQUIRE(!std::isinf(result->GetDistances()[0]));
                if (result->GetIds()[0] == i) {
                    correct++;
                }
            }
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to search on index: internalError" << std::endl;
            exit(-1);
        }
    }
    float recall_partial = correct / 1000;

    double full_time = 0;
    {
        vsag::Timer timer(full_time);
        std::shared_ptr<vsag::DiskANN> full_index =
            std::make_shared<vsag::DiskANN>(diskann::Metric::L2,
                                            "float32",
                                            ef_construction,
                                            max_degree,
                                            pq_sample_rate,
                                            pq_dims,
                                            dim,
                                            false,
                                            false,
                                            false);
        full_index->Build(dataset);
    }
    correct = 0;
    for (int i = 0; i < num_elements; i++) {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(vectors.data() + i * dim).Owner(false);
        int64_t k = 2;
        if (auto result = partial_index->KnnSearch(query, k, parameters.dump());
            result.has_value()) {
            if (result->GetNumElements() == 1) {
                REQUIRE(!std::isinf(result->GetDistances()[0]));
                if (result->GetIds()[0] == i) {
                    correct++;
                }
            }
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to search on index: internalError" << std::endl;
            exit(-1);
        }
    }
    float recall_full = correct / 1000;
    std::cout << "Recall: " << recall_full << std::endl;
    REQUIRE(recall_full == recall_partial);
}
