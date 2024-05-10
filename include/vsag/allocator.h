#pragma once

#include <string>

namespace vsag {

class Allocator {
public:
    // Return the name of the allocator.
    virtual std::string
    Name() = 0;

    // Allocate a block of at least size.
    virtual void*
    Allocate(size_t size) = 0;

    // Deallocate previously allocated block.
    virtual void
    Deallocate(void* p) = 0;

    // Reallocate the previously allocated block with long size.
    virtual void*
    Reallocate(void* p, size_t size) = 0;

public:
    virtual ~Allocator() = default;
};

}  // namespace vsag
