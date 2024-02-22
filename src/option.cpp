//
// Created by jinjiabao.jjb on 2/16/24.
//

#include "vsag/option.h"

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

}  // namespace vsag
