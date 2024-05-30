#include "hnsw.h"

#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <nlohmann/json.hpp>
#include <vector>

#include "../logger.h"
#include "fixtures.h"
#include "vsag/bitset.h"
#include "vsag/errors.h"

TEST_CASE("build & add", "[ut][hnsw]") {
    vsag::logger::set_level(vsag::logger::level::debug);

    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    std::vector<int64_t> ids(1);
    int64_t incorrect_dim = 63;
    std::vector<float> vectors(incorrect_dim);

    vsag::Dataset dataset;
    dataset.Dim(incorrect_dim)
        .NumElements(1)
        .Ids(ids.data())
        .Float32Vectors(vectors.data())
        .Owner(false);

    SECTION("build with incorrect dim") {
        auto result = index->Build(dataset);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("add with incorrect dim") {
        auto result = index->Add(dataset);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("knn_search", "[ut][hnsw]") {
    vsag::logger::set_level(vsag::logger::level::debug);

    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    const int64_t num_elements = 10;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(1).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    vsag::Dataset query;
    query.NumElements(1).Dim(dim).Float32Vectors(vectors.data()).Owner(false);
    int64_t k = 10;
    nlohmann::json params{
        {"hnsw", {{"ef_search", 100}}},
    };

    SECTION("invalid parameters k is 0") {
        auto result = index->KnnSearch(query, 0, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters k less than 0") {
        auto result = index->KnnSearch(query, -1, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters hnsw not found") {
        nlohmann::json invalid_params{};
        auto result = index->KnnSearch(query, k, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters ef_search not found") {
        nlohmann::json invalid_params{
            {"hnsw", {}},
        };
        auto result = index->KnnSearch(query, k, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid bitset length is less than the size of index") {
        auto invalid_bitset = std::make_shared<vsag::Bitset>(1);
        auto result = index->KnnSearch(query, k, params.dump(), invalid_bitset);
        REQUIRE(result.has_value());
    }

    SECTION("query length is not 1") {
        vsag::Dataset query;
        query.NumElements(2).Dim(dim).Float32Vectors(vectors.data()).Owner(false);
        auto result = index->KnnSearch(query, k, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("dimension not equal") {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim - 1).Float32Vectors(vectors.data()).Owner(false);
        auto result = index->KnnSearch(query, k, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("range_search", "[ut][hnsw]") {
    vsag::logger::set_level(vsag::logger::level::debug);

    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    const int64_t num_elements = 10;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(1).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    vsag::Dataset query;
    query.NumElements(1).Dim(dim).Float32Vectors(vectors.data()).Owner(false);
    float radius = 9.9f;
    nlohmann::json params{
        {"hnsw", {{"ef_search", 100}}},
    };

    SECTION("invalid parameter radius equals to 0") {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(vectors.data()).Owner(false);
        auto result = index->RangeSearch(query, 0, params.dump());
        REQUIRE(result.has_value());
    }

    SECTION("invalid parameter radius less than 0") {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(vectors.data()).Owner(false);
        auto result = index->RangeSearch(query, -1, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters hnsw not found") {
        nlohmann::json invalid_params{};
        auto result = index->RangeSearch(query, radius, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("invalid parameters ef_search not found") {
        nlohmann::json invalid_params{
            {"hnsw", {}},
        };
        auto result = index->RangeSearch(query, radius, invalid_params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("query length is not 1") {
        vsag::Dataset query;
        query.NumElements(2).Dim(dim).Float32Vectors(vectors.data()).Owner(false);
        auto result = index->RangeSearch(query, radius, params.dump());
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("serialize empty index", "[ut][hnsw]") {
    vsag::logger::set_level(vsag::logger::level::debug);

    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    SECTION("serialize to binaryset") {
        auto result = index->Serialize();
        REQUIRE(result.has_value());
        REQUIRE(result.value().Contains(vsag::BLANK_INDEX));
    }

    SECTION("serialize to fstream") {
        fixtures::temp_dir dir("hnsw_test_serialize_empty_index");
        std::fstream out_stream(dir.path + "empty_index.bin", std::ios::out | std::ios::binary);
        auto result = index->Serialize(out_stream);
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().type == vsag::ErrorType::INDEX_EMPTY);
    }
}

TEST_CASE("deserialize on not empty index", "[ut][hnsw]") {
    vsag::logger::set_level(vsag::logger::level::debug);

    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction, false, false, true);

    const int64_t num_elements = 10;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(1).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    SECTION("serialize to binaryset") {
        auto binary_set = index->Serialize();
        REQUIRE(binary_set.has_value());

        auto voidresult = index->Deserialize(binary_set.value());
        REQUIRE_FALSE(voidresult.has_value());
        REQUIRE(voidresult.error().type == vsag::ErrorType::INDEX_NOT_EMPTY);

        auto another_index = std::make_shared<vsag::HNSW>(std::make_shared<hnswlib::L2Space>(dim),
                                                          max_degree,
                                                          ef_construction,
                                                          false,
                                                          false,
                                                          true);
        auto deserialize_result = another_index->Deserialize(binary_set.value());
        REQUIRE(deserialize_result.has_value());
    }

    SECTION("serialize to fstream") {
        fixtures::temp_dir dir("hnsw_test_deserialize_on_not_empty_index");
        std::fstream out_stream(dir.path + "index.bin", std::ios::out | std::ios::binary);
        auto serialize_result = index->Serialize(out_stream);
        REQUIRE(serialize_result.has_value());
        out_stream.close();

        std::fstream in_stream(dir.path + "index.bin", std::ios::in | std::ios::binary);
        auto voidresult = index->Deserialize(in_stream);
        REQUIRE_FALSE(voidresult.has_value());
        REQUIRE(voidresult.error().type == vsag::ErrorType::INDEX_NOT_EMPTY);
        in_stream.close();
    }
}

TEST_CASE("static hnsw", "[ut][hnsw]") {
    vsag::logger::set_level(vsag::logger::level::debug);

    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction, true);

    const int64_t num_elements = 10;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

    vsag::Dataset dataset;
    dataset.Dim(dim).NumElements(9).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    vsag::Dataset one_vector;
    one_vector.Dim(dim)
        .NumElements(1)
        .Ids(ids.data() + 9)
        .Float32Vectors(vectors.data() + 9 * dim)
        .Owner(false);
    result = index->Add(one_vector);
    REQUIRE_FALSE(result.has_value());
    REQUIRE(result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    nlohmann::json params{
        {"hnsw", {{"ef_search", 100}}},
    };

    auto knn_result = index->KnnSearch(one_vector, 1, params.dump());
    REQUIRE(knn_result.has_value());

    auto range_result = index->RangeSearch(one_vector, 1, params.dump());
    REQUIRE_FALSE(range_result.has_value());
    REQUIRE(range_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);

    REQUIRE_THROWS(std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(127), max_degree, ef_construction, true));

    auto remove_result = index->Remove(ids[0]);
    REQUIRE_FALSE(remove_result.has_value());
    REQUIRE(remove_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
}

TEST_CASE("hnsw add vector with duplicated id", "[ut][hnsw]") {
    vsag::logger::set_level(vsag::logger::level::debug);

    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    std::vector<int64_t> ids{1};
    std::vector<float> vectors(dim);

    vsag::Dataset first_time;
    first_time.Dim(dim).NumElements(1).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);
    auto result = index->Add(first_time);
    REQUIRE(result.has_value());
    // expect failed id list emtpy
    REQUIRE(result.value().empty());

    vsag::Dataset second_time;
    second_time.Dim(dim).NumElements(1).Ids(ids.data()).Float32Vectors(vectors.data()).Owner(false);
    auto result2 = index->Add(second_time);
    REQUIRE(result2.has_value());
    // expected failed id list == {1}
    REQUIRE(result2.value().size() == 1);
    REQUIRE(result2.value()[0] == ids[0]);
}

TEST_CASE("build with reversed edges", "[ut][hnsw]") {
    vsag::logger::set_level(vsag::logger::level::debug);
    int64_t dim = 128;
    int64_t max_degree = 12;
    int64_t ef_construction = 100;
    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction, false, true);

    const int64_t num_elements = 1000;
    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_elements, dim);

    vsag::Dataset dataset;
    dataset.Dim(dim)
        .NumElements(num_elements)
        .Ids(ids.data())
        .Float32Vectors(vectors.data())
        .Owner(false);

    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    REQUIRE(index->CheckGraphIntegrity());

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
        auto new_index = std::make_shared<vsag::HNSW>(
            std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction, false, true);
        REQUIRE(new_index->Deserialize(in_file).has_value());
        REQUIRE(new_index->CheckGraphIntegrity());
    }

    // Serialize(multi-file)
    {
        fixtures::temp_dir dir("test_index_serialize_via_stream");

        if (auto bs = index->Serialize(); bs.has_value()) {
            auto keys = bs->GetKeys();
            for (auto key : keys) {
                vsag::Binary b = bs->Get(key);
                std::ofstream file(dir.path + "hnsw.index." + key, std::ios::binary);
                file.write((const char*)b.data.get(), b.size);
                file.close();
            }
            std::ofstream metafile(dir.path + "hnsw.index._meta", std::ios::out);
            for (auto key : keys) {
                metafile << key << std::endl;
            }
            metafile.close();
        } else if (bs.error().type == vsag::ErrorType::NO_ENOUGH_MEMORY) {
            std::cerr << "no enough memory to serialize index" << std::endl;
        }

        std::ifstream metafile(dir.path + "hnsw.index._meta", std::ios::in);
        std::vector<std::string> keys;
        std::string line;
        while (std::getline(metafile, line)) {
            keys.push_back(line);
        }
        metafile.close();

        vsag::BinarySet bs;
        for (auto key : keys) {
            std::ifstream file(dir.path + "hnsw.index." + key, std::ios::in);
            file.seekg(0, std::ios::end);
            vsag::Binary b;
            b.size = file.tellg();
            b.data.reset(new int8_t[b.size]);
            file.seekg(0, std::ios::beg);
            file.read((char*)b.data.get(), b.size);
            bs.Set(key, b);
        }

        auto new_index = std::make_shared<vsag::HNSW>(
            std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction, false, true);
        REQUIRE(new_index->Deserialize(bs).has_value());
        REQUIRE(new_index->CheckGraphIntegrity());
    }
}

TEST_CASE("feedback with invalid argument", "[ut][hnsw]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);
    // parameters
    int dim = 128;
    int max_degree = 16;
    int ef_construction = 200;
    int64_t num_vectors = 1000;
    int64_t k = 10;

    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction, false, false, true);

    nlohmann::json search_parameters{
        {"hnsw", {{"ef_search", 200}}},
    };

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_vectors, dim);
    vsag::Dataset query;
    query.NumElements(1).Dim(dim).Float32Vectors(vectors.data()).Owner(false);

    SECTION("index feedback with k = 0") {
        REQUIRE(index->Feedback(query, 0, search_parameters.dump(), -1).error().type ==
                vsag::ErrorType::INVALID_ARGUMENT);
        REQUIRE(index->Feedback(query, 0, search_parameters.dump()).error().type ==
                vsag::ErrorType::INVALID_ARGUMENT);
    }

    SECTION("index feedback with invalid global optimum tag id") {
        auto feedback_result = index->Feedback(query, k, search_parameters.dump(), -1000);
        REQUIRE(feedback_result.error().type == vsag::ErrorType::INVALID_ARGUMENT);
    }
}

TEST_CASE("redundant feedback and empty enhancement", "[ut][hnsw]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

    // parameters
    int dim = 128;
    int max_degree = 16;
    int ef_construction = 200;
    int64_t num_base = 10;
    int64_t num_query = 1;
    int64_t k = 10;

    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction, false, false, true);

    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(num_base, dim);
    vsag::Dataset base;
    base.NumElements(num_base)
        .Dim(dim)
        .Ids(base_ids.data())
        .Float32Vectors(base_vectors.data())
        .Owner(false);
    // build index
    auto buildindex = index->Build(base);
    REQUIRE(buildindex.has_value());

    nlohmann::json search_parameters{
        {"hnsw", {{"ef_search", 200}, {"use_conjugate_graph", true}}},
    };

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_query, dim);
    vsag::Dataset query;
    query.NumElements(1).Dim(dim).Float32Vectors(vectors.data()).Owner(false);

    auto search_result = index->KnnSearch(query, k, search_parameters.dump());
    REQUIRE(search_result.has_value());

    SECTION("index redundant feedback") {
        auto feedback_result =
            index->Feedback(query, k, search_parameters.dump(), search_result->GetIds()[0]);
        REQUIRE(*feedback_result == k - 1);

        auto redundant_feedback_result =
            index->Feedback(query, k, search_parameters.dump(), search_result->GetIds()[0]);
        REQUIRE(*redundant_feedback_result == 0);
    }

    SECTION("index search with empty enhancement") {
        auto enhanced_search_result = index->KnnSearch(query, k, search_parameters.dump());
        REQUIRE(enhanced_search_result.has_value());
        for (int i = 0; i < search_result->GetNumElements(); i++) {
            REQUIRE(search_result->GetIds()[i] == enhanced_search_result->GetIds()[i]);
        }
    }
}

TEST_CASE("feedback without use conjugate graph", "[ut][hnsw]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

    // parameters
    int dim = 128;
    int max_degree = 16;
    int ef_construction = 200;
    int64_t num_base = 10;
    int64_t num_query = 1;
    int64_t k = 10;

    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction);

    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(num_base, dim);
    vsag::Dataset base;
    base.NumElements(num_base)
        .Dim(dim)
        .Ids(base_ids.data())
        .Float32Vectors(base_vectors.data())
        .Owner(false);
    // build index
    auto buildindex = index->Build(base);
    REQUIRE(buildindex.has_value());

    nlohmann::json search_parameters{
        {"hnsw", {{"ef_search", 200}}},
    };

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_query, dim);
    vsag::Dataset query;
    query.NumElements(1).Dim(dim).Float32Vectors(vectors.data()).Owner(false);

    auto feedback_result = index->Feedback(query, k, search_parameters.dump());
    REQUIRE(feedback_result.error().type == vsag::ErrorType::UNSUPPORTED_INDEX_OPERATION);
}

TEST_CASE("feedback on empty index", "[ut][hnsw]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

    // parameters
    int dim = 128;
    int max_degree = 16;
    int ef_construction = 200;
    int64_t num_base = 0;
    int64_t num_query = 1;
    int64_t k = 100;

    auto index = std::make_shared<vsag::HNSW>(
        std::make_shared<hnswlib::L2Space>(dim), max_degree, ef_construction, false, false, true);

    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(num_base, dim);
    vsag::Dataset base;
    base.NumElements(num_base)
        .Dim(dim)
        .Ids(base_ids.data())
        .Float32Vectors(base_vectors.data())
        .Owner(false);
    // build index
    auto buildindex = index->Build(base);
    REQUIRE(buildindex.has_value());

    nlohmann::json search_parameters{
        {"hnsw", {{"ef_search", 200}}},
    };

    auto [ids, vectors] = fixtures::generate_ids_and_vectors(num_query, dim);
    vsag::Dataset query;
    query.NumElements(1).Dim(dim).Float32Vectors(vectors.data()).Owner(false);

    auto feedback_result = index->Feedback(query, k, search_parameters.dump());
    REQUIRE(*feedback_result == 0);
}

TEST_CASE("get distance by label", "[ut][hnsw]") {
    vsag::Options::Instance().logger()->SetLevel(vsag::Logger::Level::DEBUG);

    // parameters
    int dim = 128;
    int64_t num_base = 1;

    // data
    auto [base_ids, base_vectors] = fixtures::generate_ids_and_vectors(num_base, dim);

    // hnsw index
    hnswlib::L2Space space(dim);

    SECTION("hnsw test") {
        auto* alg_hnsw = new hnswlib::HierarchicalNSW(&space, 100);
        alg_hnsw->addPoint(base_vectors.data(), 0);
        REQUIRE(alg_hnsw->getDistanceByLabel(0, base_vectors.data()) < 1e-7);
        REQUIRE_THROWS(alg_hnsw->getDistanceByLabel(-1, base_vectors.data()));
        delete alg_hnsw;
    }

    SECTION("hnsw test") {
        auto* alg_hnsw_static = new hnswlib::StaticHierarchicalNSW(&space, 100);
        alg_hnsw_static->addPoint(base_vectors.data(), 0);
        REQUIRE(alg_hnsw_static->getDistanceByLabel(0, base_vectors.data()) < 1e-7);
        REQUIRE_THROWS(alg_hnsw_static->getDistanceByLabel(-1, base_vectors.data()));
        delete alg_hnsw_static;
    }
}