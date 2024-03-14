#include <spdlog/spdlog.h>

#include <catch2/catch_test_macros.hpp>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>

#include "vsag/errors.h"
#include "vsag/vsag.h"

using namespace std;

unsigned int
Factorial(unsigned int number) {
    return number <= 1 ? number : Factorial(number - 1) * number;
}

TEST_CASE("Factorials are computed", "[factorial]") {
    REQUIRE(Factorial(1) == 1);
    REQUIRE(Factorial(2) == 2);
    REQUIRE(Factorial(3) == 6);
    REQUIRE(Factorial(10) == 3628800);
}

TEST_CASE("HNSW Float Recall", "[hnsw][test]") {
    spdlog::set_level(spdlog::level::debug);
    int dim = 128;
    int max_elements = 1000;
    int max_degree = 64;
    int ef_construction = 200;
    int ef_search = 200;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_degree", max_degree},
        {"ef_construction", ef_construction},
        {"ef_search", ef_search},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};

    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    hnsw = index.value();

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];
    for (int i = 0; i < max_elements; i++) {
        ids[i] = i;
    }
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(max_elements).Ids(ids).Float32Vectors(data);
    hnsw->Add(dataset);

    // Query the elements for themselves and measure recall 1@1
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(data + i * dim).Owner(false);
        nlohmann::json parameters{
            {"hnsw", {{"ef_search", ef_search}}},
        };
        int64_t k = 10;
        if (auto result = hnsw->KnnSearch(query, k, parameters.dump()); result.has_value()) {
            if (result->GetIds()[0] == i) {
                correct++;
            }
            REQUIRE(result->GetDim() == k);
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to perform knn search on index" << std::endl;
        }
    }
    float recall = correct / max_elements;

    REQUIRE(recall == 1);
}

TEST_CASE("HNSW IP Search", "[hnsw][test]") {
    spdlog::set_level(spdlog::level::debug);
    int dim = 128;
    int max_elements = 1000;
    int max_degree = 64;
    int ef_construction = 200;
    int ef_search = 200;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_degree", max_degree},
        {"ef_construction", ef_construction},
        {"ef_search", ef_search},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "ip"}, {"dim", dim}, {"hnsw", hnsw_parameters}};
    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    hnsw = index.value();
    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];
    for (int i = 0; i < max_elements; i++) {
        ids[i] = i;
    }
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(max_elements).Ids(ids).Float32Vectors(data);
    hnsw->Build(dataset);

    // Query the elements for themselves and measure recall 1@1
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(data + i * dim).Owner(false);

        nlohmann::json parameters{
            {"hnsw", {{"ef_search", ef_search}}},
        };
        int64_t k = 10;
        if (auto result = hnsw->KnnSearch(query, k, parameters.dump()); result.has_value()) {
            if (result->GetIds()[0] == i) {
                correct++;
            }
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to perform knn search on index" << std::endl;
        }
    }
    float recall = correct / max_elements;

    REQUIRE(recall > 0.85);
}

TEST_CASE("Two HNSW", "[hnsw][test]") {
    spdlog::set_level(spdlog::level::debug);
    int dim = 128;
    int max_elements = 1000;
    int max_degree = 64;
    int ef_construction = 200;
    int ef_search = 200;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_degree", max_degree},
        {"ef_construction", ef_construction},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};
    std::shared_ptr<vsag::Index> hnsw;
    std::shared_ptr<vsag::Index> hnsw2;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    auto index2 = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    REQUIRE(index2.has_value());
    hnsw = index.value();
    hnsw2 = index2.value();
    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];
    for (int i = 0; i < max_elements; i++) {
        ids[i] = i;
    }
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(max_elements).Ids(ids).Float32Vectors(data);
    hnsw->Build(dataset);
    hnsw2->Build(dataset);

    // Query the elements for themselves and measure recall 1@1
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(data + i * dim).Owner(false);

        nlohmann::json parameters{
            {"hnsw", {{"ef_search", ef_search}}},
        };
        int64_t k = 10;

        auto result = hnsw->KnnSearch(query, k, parameters.dump());
        REQUIRE(result.has_value());
        if (result->GetIds()[0] == i) {
            REQUIRE(!std::isinf(result->GetDistances()[0]));
            correct++;
        }
    }
    float recall = correct / max_elements;

    REQUIRE(recall == 1);
}

TEST_CASE("HNSW build test", "[hnsw][test]") {
    spdlog::set_level(spdlog::level::debug);
    int dim = 128;
    int max_elements = 1000;
    int max_degree = 64;
    int ef_construction = 200;
    int ef_search = 200;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_degree", max_degree},
        {"ef_construction", ef_construction},
        {"ef_search", ef_search},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};

    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    hnsw = index.value();

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    int64_t* ids = new int64_t[2 * max_elements];
    float* data = new float[2 * dim * max_elements];
    for (int i = 0; i < max_elements; i++) {
        ids[i] = i;
    }
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }
    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(max_elements).Ids(ids).Float32Vectors(data);
    hnsw->Build(dataset);

    for (int i = max_elements; i < 2 * max_elements; i++) {
        ids[i] = i + max_elements;
    }
    for (int i = max_elements; i < 2 * dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    for (int i = max_elements; i < 2 * max_elements; i++) {
        vsag::Dataset dataset;
        dataset.Dim(dim).NumElements(1).Ids(ids + i).Float32Vectors(data + i * dim).Owner(false);
        hnsw->Add(dataset);
    }

    REQUIRE(hnsw->GetNumElements() == max_elements * 2);
}

TEST_CASE("HNSW range search", "[hnsw][test]") {
    spdlog::set_level(spdlog::level::debug);
    int dim = 71;
    int max_elements = 10000;
    int max_degree = 16;
    int ef_construction = 100;
    int ef_search = 100;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_elements", max_elements},
        {"max_degree", max_degree},
        {"ef_construction", ef_construction},
        {"ef_search", ef_search},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};

    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    hnsw = index.value();

    // Generate random data
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> distrib_real;
    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];
    for (int64_t i = 0; i < max_elements; i++) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * max_elements; ++i) {
        data[i] = distrib_real(rng);
    }

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(max_elements).Ids(ids).Float32Vectors(data);
    hnsw->Build(dataset);

    REQUIRE(hnsw->GetNumElements() == max_elements);

    float radius = 12.0f;
    float* query_data = new float[dim];
    for (int64_t i = 0; i < dim; ++i) {
        query_data[i] = distrib_real(rng);
    }
    vsag::Dataset query;
    query.Dim(dim).NumElements(1).Float32Vectors(query_data);
    nlohmann::json parameters{
        {"hnsw", {{"ef_search", ef_search}}},
    };
    auto result = hnsw->RangeSearch(query, radius, parameters.dump());
    REQUIRE(result.has_value());
    REQUIRE(result->GetNumElements() == 1);

    auto expected = vsag::l2_and_filtering(dim, max_elements, data, query_data, radius);
    if (expected->CountOnes() != result->GetDim()) {
        std::cout << "not 100% recall: expect " << expected->CountOnes() << " return "
                  << result->GetDim() << std::endl;
    }

    // check no false recall
    for (int64_t i = 0; i < result->GetDim(); ++i) {
        auto offset = result->GetIds()[i];
        CHECK(expected->Get(offset));
    }

    // recall > 99%
    CHECK((expected->CountOnes() - result->GetDim()) * 100 < max_elements);
}

TEST_CASE("HNSW filtering knn search", "[hnsw][test]") {
    spdlog::set_level(spdlog::level::debug);
    int dim = 17;
    int max_elements = 1000;
    int label_num = 100;
    int max_degree = 16;
    int ef_construction = 100;
    int ef_search = 100;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_degree", max_degree},
        {"ef_construction", ef_construction},
        {"ef_search", ef_search},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};

    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    hnsw = index.value();

    // Generate random data
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> distrib_real;
    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];

    for (int64_t i = 0; i < max_elements; i++) {
        int64_t array_id = i / label_num;
        int64_t label = i % label_num;
        ids[i] = label | (array_id << 32);
    }
    for (int64_t i = 0; i < dim * max_elements; ++i) {
        data[i] = distrib_real(rng);
    }

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(max_elements).Ids(ids).Float32Vectors(data);
    hnsw->Build(dataset);

    REQUIRE(hnsw->GetNumElements() == max_elements);

    // Query the elements for themselves and measure recall 1@1
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(data + i * dim).Owner(false);

        nlohmann::json parameters{
            {"hnsw", {{"ef_search", ef_search}}},
        };
        int64_t k = max_elements;

        vsag::BitsetPtr filter = vsag::Bitset::Random(label_num);
        int64_t num_deleted = filter->CountOnes();

        auto result = hnsw->KnnSearch(query, k, parameters.dump(), filter);
        REQUIRE(result.has_value());
        REQUIRE(result->GetDim() == max_elements - num_deleted * (max_elements / label_num));
        for (int64_t j = 0; j < result->GetDim(); ++j) {
            // deleted ids NOT in result
            REQUIRE(filter->Get(result->GetIds()[j] & 0xFFFFFFFFLL) == false);
        }
    }
}

TEST_CASE("HNSW Filtering Test", "[hnsw][test]") {
    spdlog::set_level(spdlog::level::debug);
    int dim = 17;
    int max_elements = 1000;
    int max_degree = 16;
    int ef_construction = 100;
    int ef_search = 1000;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_degree", max_degree},
        {"ef_construction", ef_construction},
        {"ef_search", ef_search},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};

    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    hnsw = index.value();

    // Generate random data
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<> distrib_real;
    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];
    for (int64_t i = 0; i < max_elements; i++) {
        ids[i] = max_elements - i - 1;
    }
    for (int64_t i = 0; i < dim * max_elements; ++i) {
        data[i] = distrib_real(rng);
    }

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(max_elements).Ids(ids).Float32Vectors(data);
    hnsw->Build(dataset);

    REQUIRE(hnsw->GetNumElements() == max_elements);

    // Query the elements for themselves and measure recall 1@1
    float correct_knn = 0.0f;
    float recall_knn = 0.0f;
    float correct_range = 0.0f;
    float recall_range = 0.0f;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(data + i * dim).Owner(false);
        nlohmann::json parameters{
            {"hnsw", {{"ef_search", ef_search}}},
        };
        float radius = 9.87f;
        int64_t k = 10;

        vsag::BitsetPtr filter = vsag::Bitset::Random(max_elements);
        int64_t num_deleted = filter->CountOnes();

        if (auto result = hnsw->RangeSearch(query, radius, parameters.dump(), filter);
            result.has_value()) {
            REQUIRE(result->GetDim() == max_elements - num_deleted);
            for (int64_t j = 0; j < result->GetDim(); ++j) {
                // deleted ids NOT in result
                REQUIRE(filter->Get(result->GetIds()[j]) == false);
            }
        } else {
            std::cerr << "failed to range search on index: internalError" << std::endl;
            exit(-1);
        }

        if (auto result = hnsw->KnnSearch(query, k, parameters.dump(), filter);
            result.has_value()) {
            REQUIRE(result.has_value());
            for (int64_t j = 0; j < result->GetDim(); ++j) {
                // deleted ids NOT in result
                REQUIRE(filter->Get(result->GetIds()[j]) == false);
            }
        } else {
            std::cerr << "failed to knn search on index: internalError" << std::endl;
            exit(-1);
        }

        size_t bytes_count = max_elements / 4 + 1;
        auto bits_ones = new uint8_t[bytes_count];
        std::memset(bits_ones, 0xFF, bytes_count);
        vsag::BitsetPtr ones = std::make_shared<vsag::Bitset>(bits_ones, bytes_count);
        if (auto result = hnsw->RangeSearch(query, radius, parameters.dump(), ones);
            result.has_value()) {
            REQUIRE(result->GetDim() == 0);
            REQUIRE(result->GetDistances() == nullptr);
            REQUIRE(result->GetIds() == nullptr);
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to range search on index: internalError" << std::endl;
            exit(-1);
        }

        if (auto result = hnsw->KnnSearch(query, k, parameters.dump(), ones); result.has_value()) {
            REQUIRE(result->GetDim() == 0);
            REQUIRE(result->GetDistances() == nullptr);
            REQUIRE(result->GetIds() == nullptr);
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to knn search on index: internalError" << std::endl;
            exit(-1);
        }

        auto bits_zeros = new uint8_t[bytes_count];
        std::memset(bits_zeros, 0, bytes_count);
        vsag::BitsetPtr zeros = std::make_shared<vsag::Bitset>(bits_zeros, bytes_count);

        if (auto result = hnsw->KnnSearch(query, k, parameters.dump(), zeros); result.has_value()) {
            correct_knn += vsag::knn_search_recall(
                data, ids, max_elements, data + i * dim, dim, result->GetIds(), result->GetDim());
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to knn search on index: internalError" << std::endl;
            exit(-1);
        }

        if (auto result = hnsw->RangeSearch(query, radius, parameters.dump(), zeros);
            result.has_value()) {
            if (result->GetNumElements() == 1) {
                if (result->GetDim() != 0 && result->GetIds()[0] == ids[i]) {
                    correct_range++;
                }
            }
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to range search on index: internalError" << std::endl;
            exit(-1);
        }
        delete[] bits_ones;
        delete[] bits_zeros;
    }
    recall_knn = correct_knn / max_elements;
    recall_range = correct_range / max_elements;

    REQUIRE(recall_range == 1);
    REQUIRE(recall_knn == 1);
}

TEST_CASE("HNSW small dimension", "[hnsw][test]") {
    spdlog::set_level(spdlog::level::debug);
    int dim = 3;
    int max_elements = 1000;
    int max_degree = 24;
    int ef_construction = 100;
    int ef_search = 100;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_degree", max_degree},
        {"ef_construction", ef_construction},
        {"ef_search", ef_search},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};

    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    hnsw = index.value();

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];
    for (int i = 0; i < max_elements; i++) {
        ids[i] = i;
    }
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(max_elements).Ids(ids).Float32Vectors(data);
    hnsw->Add(dataset);
    return;

    // Query the elements for themselves and measure recall 1@1
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(data + i * dim).Owner(false);
        nlohmann::json parameters{
            {"hnsw", {{"ef_search", ef_search}}},
        };
        int64_t k = 10;
        if (auto result = hnsw->KnnSearch(query, k, parameters.dump()); result.has_value()) {
            if (result->GetIds()[0] == i) {
                correct++;
            }
            REQUIRE(result->GetDim() == k);
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to perform knn search on index" << std::endl;
        }
    }
    float recall = correct / max_elements;

    REQUIRE(recall == 1);
}

TEST_CASE("HNSW Random Id", "[hnsw][test]") {
    spdlog::set_level(spdlog::level::debug);
    int dim = 128;
    int max_elements = 1000;
    int max_degree = 64;
    int ef_construction = 200;
    int ef_search = 200;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_degree", max_degree},
        {"ef_construction", ef_construction},
        {"ef_search", ef_search},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};

    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    hnsw = index.value();

    // Generate random data
    std::mt19937 rng;
    std::uniform_real_distribution<> distrib_real;
    std::uniform_int_distribution<> ids_random(0, max_elements - 1);
    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];
    for (int i = 0; i < max_elements; i++) {
        ids[i] = ids_random(rng);
        if (i == 1 || i == 2) {
            ids[i] = std::numeric_limits<int64_t>::max();
        } else if (i == 3 || i == 4) {
            ids[i] = std::numeric_limits<int64_t>::min();
        } else if (i == 5 || i == 6) {
            ids[i] = 1;
        } else if (i == 7 || i == 8) {
            ids[i] = -1;
        } else if (i == 9 || i == 10) {
            ids[i] = 0;
        }
    }
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(max_elements).Ids(ids).Float32Vectors(data);
    auto failed_ids = hnsw->Build(dataset);

    float rate = hnsw->GetNumElements() / (float)max_elements;
    // 1 - 1 / e
    REQUIRE((rate > 0.60 && rate < 0.65));

    REQUIRE(failed_ids->size() + hnsw->GetNumElements() == max_elements);

    // Query the elements for themselves and measure recall 1@1
    float correct = 0;
    std::set<int64_t> unique_ids;
    for (int i = 0; i < max_elements; i++) {
        if (unique_ids.find(ids[i]) != unique_ids.end()) {
            continue;
        }
        unique_ids.insert(ids[i]);
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(data + i * dim).Owner(false);
        nlohmann::json parameters{
            {"hnsw", {{"ef_search", ef_search}}},
        };
        int64_t k = 10;
        if (auto result = hnsw->KnnSearch(query, k, parameters.dump()); result.has_value()) {
            if (result->GetIds()[0] == ids[i]) {
                correct++;
            }
            REQUIRE(result->GetDim() == k);
        } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
            std::cerr << "failed to perform knn search on index" << std::endl;
        }
    }
    float recall = correct / hnsw->GetNumElements();
    REQUIRE(recall == 1);
}
