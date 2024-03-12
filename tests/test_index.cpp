
#include <catch2/catch_test_macros.hpp>

#include "catch2/catch_message.hpp"
#include "vsag/index.h"

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
