
#include "hnsw_zparameters.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("create hnsw with invalid metric type", "[ut][hnsw]") {
    auto json_string = R"(
        {
            "dtype": "float32",
            "metric_type": "unknown-metric-type",
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

    REQUIRE_THROWS_AS(vsag::CreateHnswParameters::FromJson(json_string), std::invalid_argument);
}
