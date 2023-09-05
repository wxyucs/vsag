#include <catch2/catch_test_macros.hpp>
#include <random>
#include <hnswlib/hnswlib.h>

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
    int max_elements = 10000;
    int M = 64;
    int ef_construction = 200;
    int ef_runtime = 200;
    // Initing index
    nlohmann::json index_parameters{
	{"dtype", "float32"},
	{"metric_type", "l2"},
	{"dim", dim},
        {"max_elements", max_elements},
        {"M", M},
        {"ef_construction", ef_construction},
        {"ef_runtime", ef_runtime},
    };
    auto hnsw = vsag::Factory::create("hnsw", index_parameters);

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

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
	vsag::Dataset query;
	query.SetNumElements(1);
	query.SetDim(dim);
	query.SetFloat32Vectors(data + i * dim);
	query.SetOwner(false);
	nlohmann::json parameters;
	auto result = hnsw->KnnSearch(query, 1, parameters);
	if (result.GetIds()[0] == i) {
	    correct++;
	}
    }
    float recall = correct / max_elements;

    REQUIRE(recall == 1);
}

/*
TEST_CASE("HNSW Int8 Recall", "[hnsw]") {
    int dim = 256;
    int max_elements = 10000;
    int M = 64;
    int ef_construction = 200;
    int ef_runtime = 200;
    // Initing index
    vsag::HNSW hnsw(std::make_shared<hnswlib::InnerProductSpaceInt8>(dim),
                    max_elements,
                    M,
                    ef_construction,
                    ef_runtime = 200);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    int8_t* data = new int8_t[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng) * 256;
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        hnsw.addPoint(data + i * dim, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            hnsw.searchTopK(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i)
            correct++;
    }
    float recall = correct / max_elements;

    REQUIRE(recall == 1);
}
*/
