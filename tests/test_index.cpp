
#include <fstream>

#include "catch2/catch_message.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/generators/catch_generators.hpp"
#include "fixtures/fixtures.h"
#include "nlohmann/json.hpp"
#include "spdlog/common.h"
#include "spdlog/spdlog.h"
#include "vsag/vsag.h"

TEST_CASE("check build parameters", "[index][test]") {
    auto json_string = R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 512,
            "hnsw": {
                "max_degree": 16,
                "ef_construction": 100
            },
            "diskann": {
                "max_degree": 16,
                "ef_construction": 200,
                "pq_dims": 32,
                "pq_sample_rate": 0.5
            }
        }
        )";
    auto res = vsag::check_diskann_hnsw_build_parameters(json_string);
    REQUIRE(res.has_value());
}

TEST_CASE("check search parameters", "[index][test]") {
    auto json_string = R"(
        {
            "hnsw": {
                "ef_search": 100
            },
            "diskann": {
                "ef_search": 200, 
                "beam_search": 4, 
                "io_limit": 200,
                "use_reorder": true
           }
        }
        )";
    auto res = vsag::check_diskann_hnsw_search_parameters(json_string);
    REQUIRE(res.has_value());
}

TEST_CASE("generate build parameters", "[index][test]") {
    spdlog::set_level(spdlog::level::debug);

    auto metric_type = GENERATE("l2", "IP");
    auto num_elements = GENERATE(1'000'000,
                                 2'000'000,
                                 3'000'000,
                                 4'000'000,
                                 5'000'000,
                                 6'000'000,
                                 7'000'000,
                                 8'000'000,
                                 9'000'000,
                                 10'000'000,
                                 11'000'000);
    auto dim = GENERATE(32, 64, 96, 128, 256, 512, 768, 1024, 1536, 2048, 4096);

    auto parameters = vsag::generate_build_parameters(metric_type, num_elements, dim);

    REQUIRE(parameters.has_value());
    auto json = nlohmann::json::parse(parameters.value());
    REQUIRE(json["dim"] == dim);
    REQUIRE(json["diskann"]["pq_dims"] == dim / 4);
}

TEST_CASE("generate build parameters with invalid num_elements", "[index][test]") {
    spdlog::set_level(spdlog::level::debug);

    auto metric_type = GENERATE("l2", "IP");
    auto num_elements = GENERATE(-1'000'000, -1, 0, 17'000'001, 1'000'000'000);
    int64_t dim = 128;

    auto parameters = vsag::generate_build_parameters(metric_type, num_elements, dim);

    REQUIRE(not parameters.has_value());
    REQUIRE(parameters.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("generate build parameters with invalid dim", "[index][test]") {
    spdlog::set_level(spdlog::level::debug);

    auto metric_type = GENERATE("l2", "IP");
    int64_t num_elements = 1'000'000;
    int64_t dim = GENERATE(1, 3, 42, 61, 90);

    auto parameters = vsag::generate_build_parameters(metric_type, num_elements, dim);

    REQUIRE(not parameters.has_value());
    REQUIRE(parameters.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("build index with generated_build_parameters", "[index][test]") {
    spdlog::set_level(spdlog::level::debug);

    int64_t num_vectors = 10000;
    int64_t dim = 64;

    auto index = vsag::Factory::CreateIndex(
                     "hnsw", vsag::generate_build_parameters("l2", num_vectors, dim).value())
                     .value();

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_vectors, dim);

    vsag::Dataset base;
    base.NumElements(num_vectors)
        .Dim(dim)
        .Ids(ids.data())
        .Float32Vectors(vectors.data())
        .Owner(false);
    REQUIRE(index->Build(base).has_value());

    auto search_parameters = R"(
    {
	"hnsw": {
	    "ef_search": 100
	},
	"diskann": {
	    "ef_search": 100, 
	    "beam_search": 4, 
	    "io_limit": 100,
	    "use_reorder": false
	}
    }
    )";

    int64_t correct = 0;
    for (int64_t i = 0; i < num_vectors; ++i) {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(vectors.data() + i * dim).Owner(false);
        auto result = index->KnnSearch(query, 10, search_parameters).value();
        for (int64_t j = 0; j < result.GetDim(); ++j) {
            if (i == result.GetIds()[j]) {
                ++correct;
                break;
            }
        }
    }

    float recall = 1.0 * correct / num_vectors;
    spdlog::debug("recall: {}", recall);
    REQUIRE(recall > 0.95);
}

TEST_CASE("serialize/deserialize with file stream", "[index][test]") {
    spdlog::set_level(spdlog::level::debug);

    int64_t num_vectors = 10000;
    int64_t dim = 64;
    // auto index_name = GENERATE("hnsw", "diskann");
    auto index_name = GENERATE("hnsw");
    auto metric_type = GENERATE("l2", "ip");

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_vectors, dim);
    auto index = fixtures::generate_index(index_name, metric_type, num_vectors, dim, ids, vectors);

    auto search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        },
        "diskann": {
            "ef_search": 100,
            "beam_search": 4,
            "io_limit": 100,
            "use_reorder": false
        }
    }
    )";

    {
        fixtures::temp_dir dir("test_index_serialize_via_stream");

        // serialize to file stream
        std::fstream out_file(dir.path + "index.bin", std::ios::out | std::ios::binary);
        REQUIRE(index->Serialize(out_file).has_value());
        out_file.close();

        // deserialize from file stream
        std::fstream in_file(dir.path + "index.bin", std::ios::in | std::ios::binary);
        in_file.seekg(0, std::ios::end);
        int64_t length = in_file.tellg();
        in_file.seekg(0, std::ios::beg);
        auto new_index =
            vsag::Factory::CreateIndex(
                index_name, vsag::generate_build_parameters(metric_type, num_vectors, dim).value())
                .value();
        REQUIRE(new_index->Deserialize(in_file, length).has_value());

        // compare recall
        auto before_serialize_recall =
            fixtures::test_knn_recall(index, search_parameters, num_vectors, dim, ids, vectors);
        auto aftet_serialize_recall =
            fixtures::test_knn_recall(new_index, search_parameters, num_vectors, dim, ids, vectors);

        REQUIRE(before_serialize_recall == aftet_serialize_recall);
    }
}

TEST_CASE("serialize/deserialize hnswstatic with file stream", "[index][test]") {
    spdlog::set_level(spdlog::level::debug);

    int64_t num_vectors = 10000;
    int64_t dim = 64;
    auto index_name = GENERATE("hnsw");
    auto metric_type = GENERATE("l2", "ip");

    auto build_parameters = R"(
        {
            "dtype": "float32",
            "metric_type": "l2",
            "dim": 64,
            "hnsw": {
                "max_degree": 16,
                "ef_construction": 100,
                "use_static": true
            },
            "diskann": {
                "max_degree": 16,
                "ef_construction": 200,
                "pq_dims": 32,
                "pq_sample_rate": 0.5
            }
        }
        )";

    auto index = vsag::Factory::CreateIndex(index_name, build_parameters).value();

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_vectors, dim);
    vsag::Dataset base;
    base.NumElements(num_vectors)
        .Dim(dim)
        .Ids(ids.data())
        .Float32Vectors(vectors.data())
        .Owner(false);
    REQUIRE(index->Build(base).has_value());

    auto search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        },
        "diskann": {
            "ef_search": 100,
            "beam_search": 4,
            "io_limit": 100,
            "use_reorder": false
        }
    }
    )";

    {
        fixtures::temp_dir dir("test_index_serialize_via_stream");

        // serialize to file stream
        std::fstream out_file(dir.path + "index.bin", std::ios::out | std::ios::binary);
        REQUIRE(index->Serialize(out_file).has_value());
        out_file.close();

        // deserialize from file stream
        std::fstream in_file(dir.path + "index.bin", std::ios::in | std::ios::binary);
        in_file.seekg(0, std::ios::end);
        int64_t length = in_file.tellg();
        in_file.seekg(0, std::ios::beg);
        auto new_index = vsag::Factory::CreateIndex(index_name, build_parameters).value();
        REQUIRE(new_index->Deserialize(in_file, length).has_value());

        // compare recall
        auto before_serialize_recall =
            fixtures::test_knn_recall(index, search_parameters, num_vectors, dim, ids, vectors);
        auto aftet_serialize_recall =
            fixtures::test_knn_recall(new_index, search_parameters, num_vectors, dim, ids, vectors);

        REQUIRE(before_serialize_recall == aftet_serialize_recall);
    }
}
