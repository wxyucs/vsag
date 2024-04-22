//
// Created by jinjiabao.jjb on 2/16/24.
//

#include "vsag/option.h"

#include <spdlog/spdlog.h>

#include <utility>

#include "default_allocator.h"

namespace vsag {

// Default value initialization
/*
 * sector_size_: 512
 */
Option::Option() : sector_size_(512) {
}
Option::~Option() {
}

// Returns the singleton instance
Option&
Option::Instance() {
    static Option s_instance;
    return s_instance;
}

// Gets the sector size with memory order acquire for thread safety
size_t
Option::GetSectorSize() const {
    return sector_size_.load(std::memory_order_acquire);
}

// Sets the sector size with memory order release for thread safety
void
Option::SetSectorSize(size_t size) {
    sector_size_.store(size, std::memory_order_release);
}

bool
Option::SetAllocator(std::unique_ptr<Allocator> allocator) {
    if (global_allocator_) {
        spdlog::warn("global allocator will only be set once.");
        return false;
    }
    global_allocator_ = std::move(allocator);
    return true;
}

Allocator*
Option::GetAllocator() {
    if (not global_allocator_) {
        SetAllocator(std::make_unique<DefaultAllocator>());
    }
    return global_allocator_.get();
}

}  // namespace vsag
