#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <roaring.hh>
#include <vector>

#include "vsag/bitset.h"

namespace vsag {

class BitsetImpl : public Bitset {
public:
    BitsetImpl() = default;
    virtual ~BitsetImpl() = default;

    BitsetImpl(const BitsetImpl&) = delete;
    BitsetImpl&
    operator=(const BitsetImpl&) = delete;
    BitsetImpl(BitsetImpl&&) = delete;

public:
    virtual void
    Set(int64_t pos, bool value) override;

    virtual bool
    Test(int64_t pos) override;

    virtual uint64_t
    Count() override;

    virtual std::string
    Dump() override;

private:
    std::mutex mutex_;
    roaring::Roaring r_;
};

}  //namespace vsag
