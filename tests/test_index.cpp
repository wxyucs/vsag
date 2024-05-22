
#include <spdlog/spdlog.h>

#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <fstream>
#include <nlohmann/json.hpp>

#include "fixtures/fixtures.h"
#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "vsag/dataset.h"
#include "vsag/errors.h"
#include "vsag/logger.h"
#include "vsag/options.h"
#include "vsag/vsag.h"

/////////////////////////////////////////////////////////
// index->build
/////////////////////////////////////////////////////////

TEST_CASE("hnsw build test", "[ft][index][hnsw]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

    int64_t num_vectors = 10000;
    int64_t dim = 57;
    auto metric_type = GENERATE("l2", "ip");

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_vectors * 2, dim);
    auto createindex = vsag::Factory::CreateIndex(
        "hnsw", fixtures::generate_hnsw_build_parameters_string(metric_type, dim));
    REQUIRE(createindex.has_value());
    auto index = createindex.value();

    // build index
    vsag::Dataset base;
    base.NumElements(num_vectors)
        .Dim(dim)
        .Ids(ids.data())
        .Float32Vectors(vectors.data())
        .Owner(false);
    auto buildindex = index->Build(base);
    REQUIRE(buildindex.has_value());

    // check the number of vectors in index
    REQUIRE(index->GetNumElements() == num_vectors);

    for (int64_t i = 0; i < num_vectors; ++i) {
        vsag::Dataset one_vector;
        one_vector.NumElements(1)
            .Dim(dim)
            .Ids(ids.data() + num_vectors + i)
            .Float32Vectors(vectors.data() + (num_vectors + i) * dim)
            .Owner(false);

        REQUIRE(index->Add(one_vector).has_value());
    }

    // check the number of vectors in index
    REQUIRE(index->GetNumElements() == num_vectors * 2);
}

/////////////////////////////////////////////////////////
// index->search
/////////////////////////////////////////////////////////

TEST_CASE("hnsw float32 recall", "[ft][index][hnsw]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

    int64_t num_vectors = 10000;
    int64_t dim = 104;
    auto metric_type = GENERATE("l2", "ip");

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_vectors, dim);
    auto createindex = vsag::Factory::CreateIndex(
        "hnsw", fixtures::generate_hnsw_build_parameters_string(metric_type, dim));
    REQUIRE(createindex.has_value());
    auto index = createindex.value();

    // build index
    vsag::Dataset base;
    base.NumElements(num_vectors)
        .Dim(dim)
        .Ids(ids.data())
        .Float32Vectors(vectors.data())
        .Owner(false);
    auto buildindex = index->Build(base);
    REQUIRE(buildindex.has_value());

    auto search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";

    float recall =
        fixtures::test_knn_recall(index, search_parameters, num_vectors, dim, ids, vectors);
    REQUIRE(recall > 0.99);
}

TEST_CASE("create two hnsw index in the same time", "[ft][index][hnsw]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

    int64_t num_vectors = 10000;
    int64_t dim = 49;
    auto metric_type = GENERATE("l2", "ip");

    auto [ids1, vectors1] = fixtures::generate_ids_and_vectors(num_vectors, dim);
    auto [ids2, vectors2] = fixtures::generate_ids_and_vectors(num_vectors, dim);

    // index1
    auto createindex1 = vsag::Factory::CreateIndex(
        "hnsw", fixtures::generate_hnsw_build_parameters_string(metric_type, dim));
    REQUIRE(createindex1.has_value());
    auto index1 = createindex1.value();
    // index2
    auto createindex2 = vsag::Factory::CreateIndex(
        "hnsw", fixtures::generate_hnsw_build_parameters_string(metric_type, dim));
    REQUIRE(createindex2.has_value());
    auto index2 = createindex2.value();

    vsag::Dataset base1;
    base1.NumElements(num_vectors)
        .Dim(dim)
        .Ids(ids1.data())
        .Float32Vectors(vectors1.data())
        .Owner(false);
    auto buildindex1 = index1->Build(base1);
    REQUIRE(buildindex1.has_value());
    vsag::Dataset base2;
    base2.NumElements(num_vectors)
        .Dim(dim)
        .Ids(ids2.data())
        .Float32Vectors(vectors2.data())
        .Owner(false);
    auto buildindex2 = index2->Build(base1);
    REQUIRE(buildindex2.has_value());

    auto search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";

    float recall1 =
        fixtures::test_knn_recall(index1, search_parameters, num_vectors, dim, ids1, vectors1);
    REQUIRE(recall1 > 0.99);
    float recall2 =
        fixtures::test_knn_recall(index2, search_parameters, num_vectors, dim, ids2, vectors2);
    REQUIRE(recall2 > 0.99);
}

/////////////////////////////////////////////////////////
// index->serialize/deserialize
/////////////////////////////////////////////////////////

TEST_CASE("serialize/deserialize with file stream", "[ft][index]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

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
        auto after_serialize_recall =
            fixtures::test_knn_recall(new_index, search_parameters, num_vectors, dim, ids, vectors);
        REQUIRE(before_serialize_recall == after_serialize_recall);
    }
}

TEST_CASE("serialize/deserialize hnswstatic with file stream", "[ft][index]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

    int64_t num_vectors = 10000;
    int64_t dim = 64;
    auto index_name = GENERATE("hnsw");
    auto metric_type = GENERATE("l2");  // hnswstatic does not support ip

    auto build_parameter_json = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": 64,
        "hnsw": {{
            "max_degree": 16,
            "ef_construction": 100,
            "use_static": true
        }},
        "diskann": {{
            "max_degree": 16,
            "ef_construction": 200,
            "pq_dims": 32,
            "pq_sample_rate": 0.5
        }}
    }}
    )";
    auto build_parameters = fmt::format(build_parameter_json, metric_type, dim);

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

TEST_CASE("search on a deserialized empty index", "[ft][index]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

    int64_t num_vectors = 10000;
    int64_t dim = 64;
    auto index_name = GENERATE("hnsw", "diskann");
    auto metric_type = GENERATE("l2", "ip");

    auto index =
        vsag::Factory::CreateIndex(
            index_name, vsag::generate_build_parameters(metric_type, num_vectors, dim).value())
            .value();

    vsag::Dataset base;
    REQUIRE(index->Build(base).has_value());

    auto serializeindex = index->Serialize();
    REQUIRE(serializeindex.has_value());

    auto bs = serializeindex.value();

    index = nullptr;
    index = vsag::Factory::CreateIndex(
                index_name, vsag::generate_build_parameters(metric_type, num_vectors, dim).value())
                .value();
    auto deserializeindex = index->Deserialize(bs);
    REQUIRE(deserializeindex.has_value());

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(1, dim);
    vsag::Dataset one_vector;
    one_vector.NumElements(1).Dim(dim).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);

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

    auto knnsearch = index->KnnSearch(one_vector, 10, search_parameters);
    REQUIRE(knnsearch.has_value());
    REQUIRE(knnsearch.value().GetNumElements() == 1);
    REQUIRE(knnsearch.value().GetDim() == 0);

    auto rangesearch = index->RangeSearch(one_vector, 10, search_parameters);
    REQUIRE(rangesearch.has_value());
    REQUIRE(rangesearch.value().GetNumElements() == 1);
    REQUIRE(rangesearch.value().GetDim() == 0);
}

TEST_CASE("remove vectors from the index", "[ft][index]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);
    int64_t num_vectors = 10000;
    int64_t dim = 64;
    auto index_name = GENERATE("fresh_hnsw", "hnsw", "diskann");
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

    if (index_name != "diskann") {  // index that supports remove
        // remove half data

        int correct = 0;
        for (int i = 0; i < num_vectors; i++) {
            vsag::Dataset query;
            query.NumElements(1).Dim(dim).Float32Vectors(vectors.data() + i * dim).Owner(false);

            int64_t k = 10;
            auto result = index->KnnSearch(query, k, search_parameters);
            REQUIRE(result.has_value());
            if (result->GetIds()[0] == ids[i]) {
                correct += 1;
            }
        }
        float recall_before = ((float)correct) / num_vectors;

        for (int i = 0; i < num_vectors / 2; ++i) {
            auto result = index->Remove(i);
            REQUIRE(result.has_value());
            REQUIRE(result.value());
        }
        auto wrong_result = index->Remove(-1);
        REQUIRE(wrong_result.has_value());
        REQUIRE_FALSE(wrong_result.value());

        REQUIRE(index->GetNumElements() == num_vectors / 2);

        // test recall for half data
        correct = 0;
        for (int i = 0; i < num_vectors; i++) {
            vsag::Dataset query;
            query.NumElements(1).Dim(dim).Float32Vectors(vectors.data() + i * dim).Owner(false);

            int64_t k = 10;
            auto result = index->KnnSearch(query, k, search_parameters);
            REQUIRE(result.has_value());
            if (i < num_vectors / 2) {
                REQUIRE(result->GetIds()[0] != ids[i]);
            } else {
                if (result->GetIds()[0] == ids[i]) {
                    correct += 1;
                }
            }
        }
        float recall = ((float)correct) / (num_vectors / 2);
        REQUIRE(recall >= 0.99);

        // remove all data
        for (int i = num_vectors / 2; i < num_vectors; ++i) {
            auto result = index->Remove(i);
            REQUIRE(result.has_value());
            REQUIRE(result.value());
        }

        // add data into index again
        correct = 0;
        vsag::Dataset dataset;
        dataset.NumElements(num_vectors)
            .Dim(dim)
            .Float32Vectors(vectors.data())
            .Ids(ids.data())
            .Owner(false);
        auto result = index->Add(dataset);

        for (int i = 0; i < num_vectors; i++) {
            vsag::Dataset query;
            query.NumElements(1).Dim(dim).Float32Vectors(vectors.data() + i * dim).Owner(false);

            int64_t k = 10;
            auto result = index->KnnSearch(query, k, search_parameters);
            REQUIRE(result.has_value());
            if (result->GetIds()[0] == ids[i]) {
                correct += 1;
            }
        }
        float recall_after = ((float)correct) / num_vectors;
        REQUIRE(abs(recall_before - recall_after) < 0.001);
    } else {  // index that does not supports remove
        REQUIRE_THROWS(index->Remove(-1));
    }
}

TEST_CASE("index with bsa", "[ft][index]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);
    int64_t num_vectors = 1000;
    int64_t dim = 128;
    auto index_name = GENERATE("diskann");
    auto metric_type = "l2";

    auto build_parameter_json = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "hnsw": {{
            "max_degree": 16,
            "ef_construction": 100
        }},
        "diskann": {{
            "max_degree": 16,
            "ef_construction": 100,
            "pq_dims": 32,
            "pq_sample_rate": 0.5,
            "use_pq_search": true
        }}
    }}
    )";
    auto build_parameters = fmt::format(build_parameter_json, metric_type, dim);
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
            "use_reorder": true,
            "use_bsa": true
        }
    }
    )";
    float recall =
        fixtures::test_knn_recall(index, search_parameters, num_vectors, dim, ids, vectors);
    REQUIRE(recall > 0.99);
}

/////////////////////////////////////////////////////////
// utility functions
/////////////////////////////////////////////////////////

TEST_CASE("check correct build parameters", "[ft][index]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

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

TEST_CASE("check incorrect build parameters", "[ft][index]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

    // dtype is missing
    auto json_string = R"(
    {
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
    REQUIRE_FALSE(res.has_value());
    REQUIRE(res.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("check correct search parameters", "[ft][index]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

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

TEST_CASE("check incorrect search parameters", "[ft][index]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

    auto json_string = R"(
        {
            "hhhhhhhhhhhhhh": {
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
    REQUIRE_FALSE(res.has_value());
    REQUIRE(res.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("generate build parameters", "[ft][index]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

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

TEST_CASE("generate build parameters with invalid num_elements", "[ft][index]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

    auto metric_type = GENERATE("l2", "IP");
    auto num_elements = GENERATE(-1'000'000, -1, 0, 17'000'001, 1'000'000'000);
    int64_t dim = 128;

    auto parameters = vsag::generate_build_parameters(metric_type, num_elements, dim);

    REQUIRE(not parameters.has_value());
    REQUIRE(parameters.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("generate build parameters with invalid dim", "[ft][index]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

    auto metric_type = GENERATE("l2", "IP");
    int64_t num_elements = 1'000'000;
    int64_t dim = GENERATE(1, 3, 42, 61, 90);

    auto parameters = vsag::generate_build_parameters(metric_type, num_elements, dim);

    REQUIRE(not parameters.has_value());
    REQUIRE(parameters.error().type == vsag::ErrorType::INVALID_ARGUMENT);
}

TEST_CASE("build index with generated_build_parameters", "[ft][index]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

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
    std::cout << "recall: " << recall << std::endl;
    REQUIRE(recall > 0.95);
}

TEST_CASE("hnsw + feedback with global optimum id", "[ft][index][hnsw]") {
    auto logger = vsag::Options::Instance().logger();
    logger->SetLevel(vsag::Logger::Level::DEBUG);

    // parameters
    int dim = 128;
    int num_base = 10000;
    int num_query = 1000;
    int64_t k = 10;
    auto metric_type = GENERATE("l2");
    auto build_parameter_json = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "hnsw": {{
            "max_degree": 16,
            "ef_construction": 200,
            "use_conjugate_graph": true
        }}
    }}
    )";
    auto build_parameter = fmt::format(build_parameter_json, metric_type, dim);

    // create index
    auto createindex = vsag::Factory::CreateIndex("hnsw", build_parameter);
    REQUIRE(createindex.has_value());
    auto index = createindex.value();

    // generate dataset
    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(num_base, dim);
    vsag::Dataset base, queries;
    base.NumElements(num_base)
        .Dim(dim)
        .Ids(base_ids.data())
        .Float32Vectors(base_vectors.data())
        .Owner(false);

    auto [query_ids, query_vectors] = fixtures::generate_ids_and_vectors(num_query, dim);
    queries.NumElements(num_query)
        .Dim(dim)
        .Ids(query_ids.data())
        .Float32Vectors(query_vectors.data())
        .Owner(false);

    // build index
    auto buildindex = index->Build(base);
    REQUIRE(buildindex.has_value());

    // train and search
    float recall[2];
    int correct;
    uint32_t error_fix = 0;
    bool use_conjugate_graph_search = false;
    for (int round = 0; round < 2; round++) {
        correct = 0;

        if (round == 0) {
            logger->Debug("====train stage====");
        } else {
            logger->Debug("====test stage====");
        }

        logger->Debug(fmt::format(R"(Memory Usage: {:.3f} KB)", index->GetMemoryUsage() / 1024.0));

        use_conjugate_graph_search = (round != 0);
        auto search_parameters_json = R"(
        {{
            "hnsw": {{
                "ef_search": 100,
                "use_conjugate_graph_search": {}
            }}
        }}
        )";
        auto search_parameters = fmt::format(search_parameters_json, use_conjugate_graph_search);

        for (int i = 0; i < num_query; i++) {
            vsag::Dataset query;
            query.Dim(dim)
                .Float32Vectors(queries.GetFloat32Vectors() + i * dim)
                .NumElements(1)
                .Owner(false);

            auto result = index->KnnSearch(query, k, search_parameters);
            REQUIRE(result.has_value());
            vsag::Dataset bf_result = fixtures::brute_force(query, base, 1, metric_type);
            int64_t global_optimum = bf_result.GetIds()[0];
            int64_t local_optimum = result->GetIds()[0];

            if (local_optimum != global_optimum and round == 0) {
                error_fix += *index->Feedback(query, 1, search_parameters, global_optimum);
                REQUIRE(*index->Feedback(query, 1, search_parameters) == 0);
            }

            if (local_optimum == global_optimum) {
                correct++;
            }
        }
        recall[round] = correct / (1.0 * num_query);
        logger->Debug(fmt::format(R"(Recall: {:.4f})", recall[round]));
    }

    logger->Debug("====summary====");
    logger->Debug(fmt::format(R"(Error fix: {})", error_fix));

    REQUIRE(std::fabs(recall[0] + error_fix / (1.0 * num_query) - 1.0) < 1e-7);
    REQUIRE(std::fabs(recall[1] - 1.0) < 1e-7);
}

TEST_CASE("static hnsw + feedback without global optimum id", "[ft][index][hnsw]") {
    auto logger = vsag::Options::Instance().logger();
    logger->SetLevel(vsag::Logger::Level::DEBUG);

    // parameters
    int dim = 128;
    int num_base = 10000;
    int num_query = 1000;
    auto metric_type = GENERATE("l2");
    auto build_parameter_json = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "hnsw": {{
            "max_degree": 16,
            "ef_construction": 200,
            "use_conjugate_graph": true,
            "use_static": true
        }}
    }}
    )";
    auto build_parameter = fmt::format(build_parameter_json, metric_type, dim);

    // create index
    auto createindex = vsag::Factory::CreateIndex("hnsw", build_parameter);
    REQUIRE(createindex.has_value());
    auto index = createindex.value();

    // generate dataset
    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(num_base, dim);
    vsag::Dataset base, queries;
    base.NumElements(num_base)
        .Dim(dim)
        .Ids(base_ids.data())
        .Float32Vectors(base_vectors.data())
        .Owner(false);

    auto [query_ids, query_vectors] = fixtures::generate_ids_and_vectors(num_query, dim);
    queries.NumElements(num_query)
        .Dim(dim)
        .Ids(query_ids.data())
        .Float32Vectors(query_vectors.data())
        .Owner(false);

    // build index
    auto buildindex = index->Build(base);
    REQUIRE(buildindex.has_value());

    // train and search
    float recall[2];
    int correct;
    uint32_t error_fix = 0;
    bool use_conjugate_graph_search = false;
    for (int round = 0; round < 2; round++) {
        correct = 0;

        if (round == 0) {
            logger->Debug("====train stage====");
        } else {
            logger->Debug("====test stage====");
        }

        logger->Debug(fmt::format(R"(Memory Usage: {:.3f} KB)", index->GetMemoryUsage() / 1024.0));

        use_conjugate_graph_search = (round != 0);
        auto search_parameters_json = R"(
        {{
            "hnsw": {{
                "ef_search": 100,
                "use_conjugate_graph_search": {}
            }}
        }}
        )";
        auto search_parameters = fmt::format(search_parameters_json, use_conjugate_graph_search);

        for (int i = 0; i < num_query; i++) {
            vsag::Dataset query;
            query.Dim(dim)
                .Float32Vectors(queries.GetFloat32Vectors() + i * dim)
                .NumElements(1)
                .Owner(false);

            auto result = index->KnnSearch(query, 1, search_parameters);
            REQUIRE(result.has_value());
            vsag::Dataset bf_result = fixtures::brute_force(query, base, 1, metric_type);
            int64_t global_optimum = bf_result.GetIds()[0];
            int64_t local_optimum = result->GetIds()[0];

            if (local_optimum != global_optimum and round == 0) {
                error_fix += *index->Feedback(query, 1, search_parameters);
                REQUIRE(*index->Feedback(query, 1, search_parameters, global_optimum) == 0);
            }

            if (local_optimum == global_optimum) {
                correct++;
            }
        }

        recall[round] = correct / (1.0 * num_query);
        logger->Debug(fmt::format(R"(Recall: {:.4f})", recall[round]));
    }

    logger->Debug("====summary====");
    logger->Debug(fmt::format(R"(Error fix: {})", error_fix));

    REQUIRE(std::fabs(recall[0] + error_fix / (1.0 * num_query) - 1.0) < 1e-7);
    REQUIRE(std::fabs(recall[1] - 1.0) < 1e-7);
}

TEST_CASE("using indexes that do not support conjunctive graph", "[ft][index]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);
    int64_t num_vectors = 1000;
    int64_t dim = 64;
    auto index_name = GENERATE("diskann");
    auto metric_type = GENERATE("l2");

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_vectors, dim);
    auto index = fixtures::generate_index(index_name, metric_type, num_vectors, dim, ids, vectors);

    auto search_parameters = R"(
    {
        "diskann": {
            "ef_search": 100,
            "beam_search": 4,
            "io_limit": 100,
            "use_reorder": false
        }
    }
    )";
    vsag::Dataset query;
    query.NumElements(1).Dim(dim).Float32Vectors(vectors.data()).Owner(false);
    int64_t k = 10;
    std::vector<int64_t> base_tag_ids;

    REQUIRE_THROWS(index->Feedback(query, k, search_parameters, -1));
    REQUIRE_THROWS(index->Feedback(query, k, search_parameters));

    REQUIRE_THROWS(index->Pretrain(base_tag_ids, k, search_parameters));
}
