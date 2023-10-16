#include <hnswlib/hnswlib.h>

#include <spdlog/spdlog.h>
#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>
#include <random>

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

TEST_CASE("HNSW Float Recall", "[hnsw]") {
    int dim = 128;
    int max_elements = 1000;
    int M = 64;
    int ef_construction = 200;
    int ef_runtime = 200;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_elements", max_elements},
        {"M", M},
        {"ef_construction", ef_construction},
        {"ef_runtime", ef_runtime},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};
    auto hnsw = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());

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
    dataset.SetDim(dim);
    dataset.SetNumElements(max_elements);
    dataset.SetIds(ids);
    dataset.SetFloat32Vectors(data);
    hnsw->Build(dataset);

    // Query the elements for themselves and measure recall 1@1
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.SetNumElements(1);
        query.SetDim(dim);
        query.SetFloat32Vectors(data + i * dim);
        query.SetOwner(false);
        nlohmann::json parameters;
        int64_t k = 10;
        if (auto result = hnsw->KnnSearch(query, k, parameters.dump()); result.has_value()) {
            if (result->GetIds()[0] == i) {
                correct++;
            }
        } else if (result.error() == vsag::index_error::internal_error) {
	    std::cerr << "failed to perform knn search on index" << std::endl;
        }
    }
    float recall = correct / max_elements;

    REQUIRE(recall == 1);
}

TEST_CASE("Two HNSW", "[hnsw]") {
    int dim = 128;
    int max_elements = 1000;
    int M = 64;
    int ef_construction = 200;
    int ef_runtime = 200;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_elements", max_elements},
        {"M", M},
        {"ef_construction", ef_construction},
        {"ef_runtime", ef_runtime},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};
    auto hnsw = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    auto hnsw2 = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());

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
    dataset.SetDim(dim);
    dataset.SetNumElements(max_elements);
    dataset.SetIds(ids);
    dataset.SetFloat32Vectors(data);
    hnsw->Build(dataset);
    hnsw2->Build(dataset);

    // Query the elements for themselves and measure recall 1@1
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.SetNumElements(1);
        query.SetDim(dim);
        query.SetFloat32Vectors(data + i * dim);
        query.SetOwner(false);
        nlohmann::json parameters;
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

TEST_CASE("HNSW build test", "[hnsw build]") {
    int dim = 128;
    int max_elements = 1000;
    int M = 64;
    int ef_construction = 200;
    int ef_runtime = 200;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_elements", 0},
        {"M", M},
        {"ef_construction", ef_construction},
        {"ef_runtime", ef_runtime},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};
    auto hnsw = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());

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
    dataset.SetDim(dim);
    dataset.SetNumElements(max_elements);
    dataset.SetIds(ids);
    dataset.SetFloat32Vectors(data);
    hnsw->Build(dataset);

    for (int i = max_elements; i < 2 * max_elements; i++) {
        ids[i] = i + max_elements;
    }
    for (int i = max_elements; i < 2 * dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    for (int i = max_elements; i < 2 * max_elements; i++) {
        vsag::Dataset dataset;
        dataset.SetOwner(false);
        dataset.SetDim(dim);
        dataset.SetNumElements(1);
        dataset.SetIds(ids + i);
        dataset.SetFloat32Vectors(data + i * dim);
        hnsw->Add(dataset);
    }

    REQUIRE(hnsw->GetNumElements() == max_elements * 2);
}
