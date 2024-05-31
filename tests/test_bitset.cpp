
#include <catch2/catch_test_macros.hpp>

#include "vsag/bitset.h"

TEST_CASE("test bitset", "[ft][bitset]") {
    auto bitset = vsag::Bitset::Make();

    // empty
    REQUIRE(bitset->Count() == 0);

    // set to true
    bitset->Set(100, true);
    REQUIRE(bitset->Test(100));
    REQUIRE(bitset->Count() == 1);

    // set to false
    bitset->Set(100, false);
    REQUIRE_FALSE(bitset->Test(100));
    REQUIRE(bitset->Count() == 0);

    // not set
    REQUIRE_FALSE(bitset->Test(1234567890));

    // dump
    bitset->Set(100, false);
    REQUIRE(bitset->Dump() == "{}");
    bitset->Set(100, true);
    auto dumped = bitset->Dump();
    REQUIRE(dumped == "{100}");
}
