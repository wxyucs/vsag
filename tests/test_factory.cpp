
#include <catch2/catch_test_macros.hpp>
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
        auto result = hnsw->KnnSearch(query, k, parameters.dump());
        if (result.GetIds()[0] == i) {
            correct++;
        }
    }
    float recall = correct / max_elements;

    REQUIRE(recall == 1);
}
