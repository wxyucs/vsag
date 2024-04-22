#include "vsag/option.h"

#include <catch2/catch_test_macros.hpp>

#include "default_allocator.h"

TEST_CASE("option test", "[option][test]") {
    size_t sector_size = 100;
    vsag::Option::Instance().SetSectorSize(sector_size);
    REQUIRE(vsag::Option::Instance().GetSectorSize() == sector_size);

    vsag::Option::Instance().SetAllocator(std::make_unique<vsag::DefaultAllocator>());
    REQUIRE_FALSE(
        vsag::Option::Instance().SetAllocator(std::make_unique<vsag::DefaultAllocator>()));
}
