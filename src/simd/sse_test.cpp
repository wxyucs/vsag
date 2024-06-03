
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

#include "./simd.h"
#include "catch2/catch_approx.hpp"
#include "cpuinfo.h"
#include "fixtures.h"

namespace vsag {

extern float
L2SqrSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

extern float
InnerProductSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

extern float
InnerProductSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

}  // namespace vsag

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
        size_t dim = 8;
        auto vectors = fixtures::generate_vectors(2, dim);

        fixtures::dist_t distance =
            vsag::InnerProductSIMD4ExtSSE(vectors.data(), vectors.data() + dim, &dim);
        fixtures::dist_t expected_distance =
            vsag::InnerProduct(vectors.data(), vectors.data() + dim, &dim);
        // REQUIRE(distance == expected_distance);
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
