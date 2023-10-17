
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <iostream>
#include <string>

#include "vsag/vsag.h"

TEST_CASE("l2_and_filtering", "[utils]") {
    int64_t dim = 4;
    int64_t nb = 10;
    float* base = new float[nb * dim];
    for (int64_t i = 0; i < nb; ++i) {
        for (int64_t d = 0; d < dim; ++d) {
            base[i * dim + d] = i;
        }
    }

    float* query = new float[dim]{5, 5, 5, 5};
    std::vector<unsigned char> res = vsag::l2_and_filtering(dim, nb, base, query, 20.0f);
    std::vector<unsigned char> expected{0xf8, 0x00}; // 1111_1000, 0000_0000
    REQUIRE(res == expected);
}
