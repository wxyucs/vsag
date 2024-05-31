
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <iostream>
#include <string>

#include "vsag/utils.h"
#include "vsag/vsag.h"

using namespace vsag;

TEST_CASE("l2_and_filtering", "[ft][utils]") {
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
    CHECK(res->Count() == 5);
    CHECK_FALSE(res->Test(0));
    CHECK_FALSE(res->Test(1));
    CHECK_FALSE(res->Test(2));
    CHECK(res->Test(3));
    CHECK(res->Test(4));
    CHECK(res->Test(5));
    CHECK(res->Test(6));
    CHECK(res->Test(7));
    CHECK_FALSE(res->Test(8));
    CHECK_FALSE(res->Test(9));
}

TEST_CASE("version", "[ft][version]") {
    std::cout << "version: " << vsag::version() << std::endl;
}
