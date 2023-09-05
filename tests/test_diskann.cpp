//
// Created by inabao on 2023/8/22.
//
#include <catch2/catch_test_macros.hpp>
#include "vsag/vsag.h"
#include <omp.h>
#include <cstring>

#include <ann_exception.h>
#include <index.h>
#include <index_factory.h>
#include <memory_mapper.h>
#include <utils.h>

TEST_CASE("DiskAnn Float Recall", "[diskann]") {

    diskann::Metric metric = diskann::Metric::L2;

    size_t data_num = 10000, data_dim = 256;
    int L = 200;
    int R = 64;
    std::string data_type = "float";
    std::string label_type = "uint32";

    auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
                                  .build();

    diskann::Index<float, uint32_t, uint32_t> index(metric, data_dim, data_num, false, false, false, false, 4, false);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float * data = new float[data_dim * data_num];
    std::vector<uint32_t> tags;
    for (uint32_t i = 0; i < data_dim * data_num; i++) {
        data[i] = distrib_real(rng);
        tags.push_back(i);
    }
    index.build(data, data_num, index_build_params, tags);
    auto results = new uint32_t[20];
    double correct = 0;
    for (size_t i = 0; i < data_num; i++) {
        index.search(data + i * data_dim, 1, L, results);
        if (results[0] == i) {
            correct += 1;
        }
    }
    REQUIRE((correct / data_num) == 1);
}
