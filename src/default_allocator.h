
#pragma once

#include "vsag/allocator.h"

namespace vsag {

class DefaultAllocator : public Allocator {
public:
    DefaultAllocator() = default;
    virtual ~DefaultAllocator() = default;

    DefaultAllocator(const DefaultAllocator&) = delete;
    DefaultAllocator(DefaultAllocator&&) = delete;

public:
    std::string
    Name() override;

    void*
    Allocate(size_t size) override;

    void
    Deallocate(void* p) override;

    void*
    Reallocate(void* p, size_t size) override;
};

}  // namespace vsag
