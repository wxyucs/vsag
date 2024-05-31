
#include "./bitset_impl.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <sstream>

namespace vsag {

BitsetPtr
Bitset::Random(int64_t length) {
    auto bitset = std::make_shared<BitsetImpl>();
    static auto gen =
        std::bind(std::uniform_int_distribution<>(0, 1), std::default_random_engine());
    for (int64_t i = 0; i < length; ++i) {
        bitset->Set(i, gen());
    }
    return bitset;
}

BitsetPtr
Bitset::Make() {
    return std::make_shared<BitsetImpl>();
}

void
BitsetImpl::Set(int64_t pos, bool value) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (value) {
        r_.add(pos);
    } else {
        r_.remove(pos);
    }
}

bool
BitsetImpl::Test(int64_t pos) {
    std::lock_guard<std::mutex> lock(mutex_);
    return r_.contains(pos);
}

uint64_t
BitsetImpl::Count() {
    return r_.cardinality();
}

std::string
BitsetImpl::Dump() {
    return r_.toString();
}

}  // namespace vsag
