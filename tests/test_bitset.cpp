
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "vsag/bitset.h"

using namespace vsag;
using Catch::Matchers::ContainsSubstring;

TEST_CASE("general usage", "[bitset]") {
    BitsetPtr bp = std::make_shared<Bitset>();
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

TEST_CASE("get and set", "[bitset]") {
    BitsetPtr bp = std::make_shared<Bitset>();

    CHECK_THROWS_WITH(
        bp->Get(-1),
        ContainsSubstring("failed to get bitset: offset") && ContainsSubstring("is less than 0"));
    CHECK_THROWS_WITH(bp->Get(1),
                      ContainsSubstring("failed to get from bitset: offset") &&
                          ContainsSubstring("is greater than capcity"));

    CHECK_THROWS_WITH(
        bp->Set(-1, true),
        ContainsSubstring("failed to set bitset: offset") && ContainsSubstring("is less than 0"));

    CHECK_NOTHROW(bp->Set(0, true));
    CHECK(bp->Capcity() == 8);
    CHECK_NOTHROW(bp->Get(0));

    CHECK_NOTHROW(bp->Set(8, true));
    CHECK(bp->Capcity() == 16);
    CHECK_NOTHROW(bp->Get(8));
}

TEST_CASE("count ones and zeros", "[bitset]") {
    BitsetPtr bp = std::make_shared<Bitset>();

    bp->Set(1, true);
    CHECK(bp->CountOnes() == 1);
    CHECK(bp->CountZeros() == 7);
    bp->Set(11, true);
    CHECK(bp->CountOnes() == 2);
    CHECK(bp->CountZeros() == 14);
}

TEST_CASE("capcity and extend", "[bitset]") {
    int64_t mem_limit = 1024 * 1024;
    BitsetPtr bp = std::make_shared<Bitset>(mem_limit);
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
                      ContainsSubstring("failed to extend bitset: number_of_bytes") &&
                          ContainsSubstring("is greater than memory limit"));

    CHECK_THROWS_WITH(bp->Extend(64),
                      ContainsSubstring("failed to extend bitset: number_of_bits") &&
                          ContainsSubstring("is less than current capcity"));

    CHECK_NOTHROW(bp->Extend(mem_limit * 8));
    CHECK(bp->Capcity() == mem_limit * 8);
}
