
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

#include "./simd.h"
#include "catch2/catch_approx.hpp"
#include "cpuinfo.h"
#include "fixtures.h"

TEST_CASE("sse l2 simd16", "[ut][simd][sse]") {
#if defined(ENABLE_SSE)
    if (cpuinfo_has_x86_sse()) {
        size_t dim = 16;
        auto vectors = fixtures::generate_vectors(2, dim);

        fixtures::dist_t distance =
            vsag::L2SqrSIMD16ExtSSE(vectors.data(), vectors.data() + dim, &dim);
        fixtures::dist_t expected_distance =
            vsag::L2Sqr(vectors.data(), vectors.data() + dim, &dim);
        REQUIRE(distance == expected_distance);
    }
#endif
}

TEST_CASE("sse ip simd4", "[ut][simd][sse]") {
#if defined(ENABLE_SSE)
    if (cpuinfo_has_x86_sse()) {
        size_t dim = 20;
        auto vectors = fixtures::generate_vectors(2, dim);

        fixtures::dist_t distance =
            vsag::InnerProductSIMD4ExtSSE(vectors.data(), vectors.data() + dim, &dim);
        fixtures::dist_t expected_distance =
            vsag::InnerProduct(vectors.data(), vectors.data() + dim, &dim);
        REQUIRE(distance == expected_distance);
    }
#endif
}

TEST_CASE("sse ip simd16", "[ut][simd][sse]") {
#if defined(ENABLE_SSE)
    if (cpuinfo_has_x86_sse()) {
        size_t dim = 32;
        auto vectors = fixtures::generate_vectors(2, dim);

        fixtures::dist_t distance =
            vsag::InnerProductSIMD16ExtSSE(vectors.data(), vectors.data() + dim, &dim);
        fixtures::dist_t expected_distance =
            vsag::InnerProduct(vectors.data(), vectors.data() + dim, &dim);
        REQUIRE(distance == expected_distance);
    }
#endif
}

TEST_CASE("sse pq calculation", "[ut][simd][sse]") {
#if defined(ENABLE_SSE)
    if (cpuinfo_has_x86_sse()) {
        size_t dim = 256;
        float single_dim_value = 0.571;
        float results_expected[256]{0.0f};
        float results[256]{0.0f};
        auto vectors = fixtures::generate_vectors(1, dim);

        vsag::PQDistanceSSEFloat256(vectors.data(), single_dim_value, results);
        vsag::PQDistanceFloat256(vectors.data(), single_dim_value, results_expected);

        for (int i = 0; i < dim; ++i) {
            REQUIRE(fabs(results_expected[i] - results[i]) < 0.001);
        }
    }
#endif
}