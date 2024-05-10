#include "vsag/options.h"

#include <catch2/catch_test_macros.hpp>

#include "default_allocator.h"

TEST_CASE("option test", "[option][test]") {
    size_t sector_size = 100;
    vsag::Options::Instance().set_sector_size(sector_size);
    REQUIRE(vsag::Option::Instance().sector_size() == sector_size);

    vsag::Option::Instance().set_allocator(std::make_unique<vsag::DefaultAllocator>());
    REQUIRE_FALSE(
        vsag::Option::Instance().set_allocator(std::make_unique<vsag::DefaultAllocator>()));
}
