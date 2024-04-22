
#include "default_allocator.h"

#include <catch2/catch_test_macros.hpp>

TEST_CASE("default allocator", "[ut]") {
    vsag::DefaultAllocator allocator;
    int number = 69278;
    auto p = (int*)allocator.Allocate(sizeof(int) * 1);

    REQUIRE(p);

    *p = number;
    auto p2 = (int*)allocator.Reallocate(p, sizeof(int) * 2);
    REQUIRE(*p2 == number);

    allocator.Deallocate(p2);
}
