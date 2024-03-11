
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <iostream>
#include <string>

#include "vsag/utils.h"
#include "vsag/vsag.h"

using namespace vsag;

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
    auto res = l2_and_filtering(dim, nb, base, query, 20.0f);
    delete[] base;
    delete[] query;
    CHECK(res->Capacity() == 16);
    CHECK(res->CountOnes() == 5);
    CHECK_FALSE(res->Get(0));
    CHECK_FALSE(res->Get(1));
    CHECK_FALSE(res->Get(2));
    CHECK(res->Get(3));
    CHECK(res->Get(4));
    CHECK(res->Get(5));
    CHECK(res->Get(6));
    CHECK(res->Get(7));
    CHECK_FALSE(res->Get(8));
    CHECK_FALSE(res->Get(9));
}

TEST_CASE("version", "[version]") {
    std::cout << "version: " << vsag::version() << std::endl;
}
