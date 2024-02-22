#include <catch2/catch_test_macros.hpp>

#include "vsag/vsag.h"

TEST_CASE("option test", "[option]") {
    size_t sector_size = 100;
    vsag::Option::Instance().SetSectorSize(sector_size);
    REQUIRE(vsag::Option::Instance().GetSectorSize() == sector_size);
}
