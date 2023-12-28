//
// Created by root on 11/15/23.
//
#include <hnswlib/hnswlib.h>
#include <spdlog/spdlog.h>

#include <catch2/catch_test_macros.hpp>
#include <nlohmann/json.hpp>

#include "vsag/vsag.h"

using namespace std;

TEST_CASE("Random Index Test", "[random]") {
    std::random_device rd;
    std::mt19937 rng(rd());

    std::uniform_int_distribution<int> dim_generate(1, 500);
    std::uniform_int_distribution<int> num_generate(
        2, 1000);  // DiskANN does not allow building a graph with fewer than 2 points.
    std::uniform_int_distribution<int> m_generate(
        2, 64);  // When the number of edges is less than 2, connectivity cannot be guaranteed.
    std::uniform_int_distribution<int> construct_generate(1, 500);
    std::uniform_int_distribution<int> search_generate(1, 500);
    std::uniform_int_distribution<int> k_generate(1, 200);
    std::uniform_int_distribution<int> io_limit_generate(1, 500);
    std::uniform_real_distribution<float> threshold_generate(1, 500);
    std::uniform_real_distribution<float> chunks_num_generate(
        1, 512);  // DiskANN does not allow the number of PQ buckets to be greater than 512.
    std::uniform_real_distribution<float> preload_generate;
    std::uniform_real_distribution<float> mold_generate(-1000, 1000);

    int dim = dim_generate(rng);
    int max_elements = num_generate(rng);
    int M = m_generate(rng);
    int ef_construction = construct_generate(rng);
    int ef_runtime = search_generate(rng);
    int64_t k = k_generate(rng);

    int io_limit = io_limit_generate(rng);
    float threshold = threshold_generate(rng);
    int chunks_num = chunks_num_generate(rng);
    bool preload = preload_generate(rng) > 0.5;
    float mold = mold_generate(rng);

    std::uniform_int_distribution<int> seed_random;
    int seed = seed_random(rng);
    rng.seed(seed);

    spdlog::info(
        "seed: {}, dim: {}, max_elements: {}, M: {}, ef_construction: {}, ef_runtime: {}, k: {}, "
        "io_limit: {}, threshold: {}, chunks_num: {}, preload: {}, mold: {}",
        seed,
        dim,
        max_elements,
        M,
        ef_construction,
        ef_runtime,
        k,
        io_limit,
        threshold,
        chunks_num,
        preload,
        mold);
    float p_val = 0.5;
    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_elements", max_elements},
        {"M", M},
        {"ef_construction", ef_construction},
        {"ef_runtime", ef_runtime},
    };

    nlohmann::json diskann_parameters{{"R", M},
                                      {"L", ef_construction},
                                      {"p_val", p_val},
                                      {"disk_pq_dims", chunks_num},
                                      {"preload", preload}};
    nlohmann::json index_parameters{{"dtype", "float32"},
                                    {"metric_type", "l2"},
                                    {"dim", dim},
                                    {"diskann", diskann_parameters},
                                    {"hnsw", hnsw_parameters}};

    nlohmann::json parameters{
        {"diskann", {{"ef_search", ef_runtime}, {"beam_search", 4}, {"io_limit", io_limit}}},
        {"hnsw", {{"ef_runtime", ef_runtime}}}};

    // Generate random data
    std::uniform_real_distribution<> distrib_real;
    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];
    for (int i = 0; i < max_elements; i++) {
        ids[i] = i;
    }
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng) * mold;
    }

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(max_elements).Ids(ids).Float32Vectors(data);

    std::shared_ptr<vsag::Index> hnsw;
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
    REQUIRE(index.has_value());
    hnsw = index.value();
    hnsw->Build(dataset);

    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(data + i * dim).Owner(false);
        auto knn_result = hnsw->KnnSearch(query, k, parameters.dump());
        REQUIRE(knn_result.has_value());

        REQUIRE(knn_result.value().GetDim() == std::min(k, (int64_t)max_elements));
        auto range_result = hnsw->RangeSearch(query, k, parameters.dump());
        REQUIRE(range_result.has_value());
    }

    std::shared_ptr<vsag::Index> diskann;
    index = vsag::Factory::CreateIndex("diskann", index_parameters.dump());
    REQUIRE(index.has_value());
    diskann = index.value();

    diskann->Build(dataset);

    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(data + i * dim).Owner(false);
        auto knn_result = diskann->KnnSearch(query, k, parameters.dump());
        REQUIRE(knn_result.has_value());
        REQUIRE(knn_result.value().GetDim() == std::min(k, (int64_t)max_elements));
        auto range_result = diskann->RangeSearch(query, k, parameters.dump());
        REQUIRE(range_result.has_value());
    }
}
