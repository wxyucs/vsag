#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "vsag/allocator.h"

namespace vsag {

class Option {
private:
    // Private constructor and destructor to ensure singleton
    Option();
    ~Option();

    // Deleted copy constructor and assignment operator to prevent copies
    Option(const Option&) = delete;
    Option&
    operator=(const Option&) = delete;

    // In a single query, the space size used to store disk vectors.
    std::atomic<size_t> sector_size_;

    // The allocator will only be set once.
    std::unique_ptr<Allocator> global_allocator_;

public:
    static Option&
    Instance();

    size_t
    GetSectorSize() const;

    void
    SetSectorSize(size_t size);

    bool
    SetAllocator(std::unique_ptr<Allocator> allocator);

    Allocator*
    GetAllocator();
};

}  // namespace vsag
