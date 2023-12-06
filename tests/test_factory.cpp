
#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <nlohmann/json.hpp>

#include "vsag/vsag.h"

TEST_CASE("index params", "[factory]") {
    int dim = 16;
    int max_elements = 1000;
    int M = 16;
    int ef_construction = 100;
    int ef_runtime = 100;

    nlohmann::json hnsw_parameters{
        {"max_elements", max_elements},
        {"M", M},
        {"ef_construction", ef_construction},
        {"ef_runtime", ef_runtime},
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
    hnsw->Build(dataset);

    // Query the elements for themselves and measure recall 1@1
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(data + i * dim).Owner(false);

        nlohmann::json parameters{
            {"hnsw", {{"ef_runtime", ef_runtime}}},
        };
        int64_t k = 10;
        if (auto result = hnsw->KnnSearch(query, k, parameters.dump()); result.has_value()) {
            if (result->GetIds()[0] == i) {
                correct++;
            }
        } else if (result.error() == vsag::index_error::internal_error) {
            std::cerr << "failed to search on index: internal error" << std::endl;
        }
    }
    float recall = correct / max_elements;

    REQUIRE(recall == 1);
}
