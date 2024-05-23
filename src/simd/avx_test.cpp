
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

#include "./simd.h"
#include "catch2/catch_approx.hpp"
#include "cpuinfo.h"
#include "fixtures.h"

namespace vsag {

extern float
L2SqrSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

extern float
InnerProductSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

}  // namespace vsag

TEST_CASE("avx l2 simd16", "[ut][simd][avx]") {
#if defined(ENABLE_AVX)
    if (cpuinfo_has_x86_sse()) {
        size_t dim = 16;
        auto vectors = fixtures::generate_vectors(2, dim);

        fixtures::dist_t distance =
            vsag::L2SqrSIMD16ExtAVX(vectors.data(), vectors.data() + dim, &dim);
        fixtures::dist_t expected_distance =
            vsag::L2Sqr(vectors.data(), vectors.data() + dim, &dim);
        REQUIRE(distance == expected_distance);
    }
#endif
}

TEST_CASE("avx ip simd16", "[ut][simd][avx]") {
#if defined(ENABLE_AVX)
    if (cpuinfo_has_x86_sse()) {
        size_t dim = 16;
        auto vectors = fixtures::generate_vectors(2, dim);

        fixtures::dist_t distance =
            vsag::InnerProductSIMD16ExtAVX(vectors.data(), vectors.data() + dim, &dim);
        fixtures::dist_t expected_distance =
            vsag::InnerProduct(vectors.data(), vectors.data() + dim, &dim);
        REQUIRE(distance == expected_distance);
    }
#endif
}
