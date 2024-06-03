#include "vsag/options.h"

#include <catch2/catch_test_macros.hpp>

#include "default_allocator.h"

TEST_CASE("option test", "[ut][option]") {
    size_t sector_size = 100;
    vsag::Options::Instance().set_sector_size(sector_size);
    REQUIRE(vsag::Option::Instance().sector_size() == sector_size);

    size_t block_size_limit = 134217728;
    vsag::Options::Instance().set_block_size_limit(block_size_limit);
    REQUIRE(vsag::Option::Instance().block_size_limit() == block_size_limit);
}

TEST_CASE("set allocator twice", "[ut][option][allocator]") {
    vsag::DefaultAllocator allocator;
    vsag::Option::Instance().set_allocator(&allocator);
    vsag::Option::Instance().set_allocator(&allocator);
}

TEST_CASE("set allocator style1", "[ut][option][allocator]") {
    vsag::DefaultAllocator allocator;
    vsag::Option::Instance().set_allocator(&allocator);
}

TEST_CASE("set allocator style2", "[ut][option][allocator]") {
    void* memory = ::operator new(sizeof(vsag::DefaultAllocator));
    vsag::DefaultAllocator* allocator = new (memory) vsag::DefaultAllocator();

    vsag::Option::Instance().set_allocator(allocator);

    allocator->~DefaultAllocator();
    ::operator delete(memory);
}
