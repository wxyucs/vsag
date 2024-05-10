
#pragma once

#include "vsag/allocator.h"

namespace vsag {

class DefaultAllocator : public Allocator {
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
