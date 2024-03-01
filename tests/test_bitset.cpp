
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "vsag/bitset.h"

using namespace vsag;
using Catch::Matchers::ContainsSubstring;

TEST_CASE("general usage", "[bitset]") {
    BitsetPtr bp = std::make_shared<Bitset>();
    CHECK(bp->Capacity() == 0);

    bp->Set(100, true);
    CHECK(bp->Get(100) == true);
    CHECK(bp->CountOnes() == 1);
    CHECK(bp->CountZeros() == 103);
    CHECK(bp->Capacity() == 104);

    bp->Set(79, true);
    CHECK(bp->Get(79) == true);
    CHECK(bp->CountOnes() == 2);
    CHECK(bp->CountZeros() == 102);
    CHECK(bp->Capacity() == 104);

    bp->Set(100, false);
    CHECK(bp->Get(100) == false);
    CHECK(bp->CountOnes() == 1);
    CHECK(bp->CountZeros() == 103);
    CHECK(bp->Capacity() == 104);
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
    CHECK(bp->Capacity() == 8);
    CHECK_NOTHROW(bp->Get(0));

    CHECK_NOTHROW(bp->Set(8, true));
    CHECK(bp->Capacity() == 16);
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
    CHECK(bp->Capacity() == 0);

    bp->Set(1, true);
    CHECK(bp->Capacity() == 8);

    bp->Set(11, true);
    CHECK(bp->Capacity() == 16);

    bp->Set(1, false);
    CHECK(bp->Capacity() == 16);

    bp->Set(111, false);
    CHECK(bp->Capacity() == 112);

    CHECK_THROWS_WITH(bp->Extend(mem_limit * 8 + 1),
                      ContainsSubstring("failed to extend bitset: number_of_bytes") &&
                          ContainsSubstring("is greater than memory limit"));

    CHECK_THROWS_WITH(bp->Extend(64),
                      ContainsSubstring("failed to extend bitset: number_of_bits") &&
                          ContainsSubstring("is less than current capcity"));

    CHECK_NOTHROW(bp->Extend(mem_limit * 8));
    CHECK(bp->Capacity() == mem_limit * 8);
}

TEST_CASE("construct from memory", "[bitset]") {
    auto memory = std::shared_ptr<uint8_t[]>(new uint8_t[2]);
    memory[0] = 0b01010101;
    memory[1] = 0b11111010;
    BitsetPtr bp = std::make_shared<Bitset>(memory.get(), 2);
    CHECK(bp->Capacity() == 16);

    CHECK(bp->Get(0) == true);
    CHECK(bp->Get(1) == false);
    CHECK(bp->Get(2) == true);
    CHECK(bp->Get(3) == false);
    CHECK(bp->Get(4) == true);
    CHECK(bp->Get(5) == false);
    CHECK(bp->Get(6) == true);
    CHECK(bp->Get(7) == false);

    CHECK(bp->Get(8) == false);
    CHECK(bp->Get(9) == true);
    CHECK(bp->Get(10) == false);
    CHECK(bp->Get(11) == true);
    CHECK(bp->Get(12) == true);
    CHECK(bp->Get(13) == true);
    CHECK(bp->Get(14) == true);
    CHECK(bp->Get(15) == true);

    CHECK(bp->CountOnes() == 10);
    CHECK(bp->CountZeros() == 6);

    std::vector<uint8_t> buffer;
    buffer.resize(1'000'000);
    static auto gen =
        std::bind(std::uniform_real_distribution<float>(0, 1), std::default_random_engine());
    auto ptr = (float*)buffer.data();
    for (uint64_t i = 0; i < buffer.size() / 4; ++i) {
        ptr[i] = gen();
    }

    uint64_t count1 = 0;
    BENCHMARK("popcount") {
        uint64_t count = 0;
        for (uint8_t num : buffer) {
            count += std::__popcount(num);
        }
        count1 = count;
    };

    uint64_t count2 = 0;
    BENCHMARK("while") {
        uint64_t count = 0;
        for (uint8_t num : buffer) {
            while (num) {
                count += num & 1;
                num >>= 1;
            }
        }
        count2 = count;
    };

    REQUIRE(count1 == count2);
}
