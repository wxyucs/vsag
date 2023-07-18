#include <catch2/catch_test_macros.hpp>
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

TEST_CASE("Float Recall", "[hnsw]") {
    int dim = 128;
    int max_elements = 10000;
    int M = 64;
    int ef_construction = 200;
    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW* hnsw = new hnswlib::HierarchicalNSW(&space, max_elements, M, ef_construction);

    // generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    for (int i = 0; i < max_elements; i++) {
        hnsw->addPoint(data + i * dim, i);
    }

    hnsw->setEf(200);
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i)
            correct++;
    }
    float recall = correct / max_elements;

    REQUIRE(recall == 1);
}

TEST_CASE("Int8 Recall", "[hnsw]") {

    int dim = 256;
    int max_elements = 10000;
    int M = 64;
    int ef_construction = 200;
    // Initing index
    vsag::HNSW hnsw(dim, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    int8_t * data = new int8_t[dim * max_elements];
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

