
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "vsag/bitmap.h"

using namespace vsag;
using Catch::Matchers::ContainsSubstring;

TEST_CASE("general usage", "[bitmap]") {
    BitmapPtr bp = std::make_shared<Bitmap>();
    CHECK(bp->Capcity() == 0);

    bp->Set(100, true);
    CHECK(bp->Get(100) == true);
    CHECK(bp->CountOnes() == 1);
    CHECK(bp->CountZeros() == 103);
    CHECK(bp->Capcity() == 104);

    bp->Set(79, true);
    CHECK(bp->Get(79) == true);
    CHECK(bp->CountOnes() == 2);
    CHECK(bp->CountZeros() == 102);
    CHECK(bp->Capcity() == 104);

    bp->Set(100, false);
    CHECK(bp->Get(100) == false);
    CHECK(bp->CountOnes() == 1);
    CHECK(bp->CountZeros() == 103);
    CHECK(bp->Capcity() == 104);
}

TEST_CASE("get and set", "[bitmap]") {
    BitmapPtr bp = std::make_shared<Bitmap>();

    CHECK_THROWS_WITH(
        bp->Get(-1),
        ContainsSubstring("failed to get bitmap: offset") && ContainsSubstring("is less than 0"));
    CHECK_THROWS_WITH(bp->Get(1),
                      ContainsSubstring("failed to get from bitmap: offset") &&
                          ContainsSubstring("is greater than capcity"));

    CHECK_THROWS_WITH(
        bp->Set(-1, true),
        ContainsSubstring("failed to set bitmap: offset") && ContainsSubstring("is less than 0"));

    CHECK_NOTHROW(bp->Set(0, true));
    CHECK(bp->Capcity() == 8);
    CHECK_NOTHROW(bp->Get(0));

    CHECK_NOTHROW(bp->Set(8, true));
    CHECK(bp->Capcity() == 16);
    CHECK_NOTHROW(bp->Get(8));
}

TEST_CASE("count ones and zeros", "[bitmap]") {
    BitmapPtr bp = std::make_shared<Bitmap>();

    bp->Set(1, true);
    CHECK(bp->CountOnes() == 1);
    CHECK(bp->CountZeros() == 7);
    bp->Set(11, true);
    CHECK(bp->CountOnes() == 2);
    CHECK(bp->CountZeros() == 14);
}

TEST_CASE("capcity and extend", "[bitmap]") {
    int64_t mem_limit = 1024 * 1024;
    BitmapPtr bp = std::make_shared<Bitmap>(mem_limit);
    CHECK(bp->Capcity() == 0);

    bp->Set(1, true);
    CHECK(bp->Capcity() == 8);

    bp->Set(11, true);
    CHECK(bp->Capcity() == 16);

    bp->Set(1, false);
    CHECK(bp->Capcity() == 16);

    bp->Set(111, false);
    CHECK(bp->Capcity() == 112);

    CHECK_THROWS_WITH(bp->Extend(mem_limit * 8 + 1),
                      ContainsSubstring("failed to extend bitmap: number_of_bytes") &&
                          ContainsSubstring("is greater than memory limit"));

    CHECK_THROWS_WITH(bp->Extend(64),
                      ContainsSubstring("failed to extend bitmap: number_of_bits") &&
                          ContainsSubstring("is less than current capcity"));

    CHECK_NOTHROW(bp->Extend(mem_limit * 8));
    CHECK(bp->Capcity() == mem_limit * 8);
}
