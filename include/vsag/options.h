#pragma once

#include <atomic>
#include <memory>
#include <string>

#include "vsag/allocator.h"
#include "vsag/logger.h"

namespace vsag {

class Options {
public:
    static Options&
    Instance();

public:
    // Gets the sector size with memory order acquire for thread safety
    inline size_t
    sector_size() const {
        return sector_size_.load(std::memory_order_acquire);
    }

    inline void
    set_sector_size(size_t size) {
        sector_size_.store(size, std::memory_order_release);
    }

    // Gets the limit of block size with memory order acquire for thread safety
    inline size_t
    block_size_limit() const {
        return block_size_limit_.load(std::memory_order_acquire);
    }

    inline void
    set_block_size_limit(size_t size) {
        block_size_limit_.store(size, std::memory_order_release);
    }

    Allocator*
    allocator();

    inline bool
    set_allocator(std::unique_ptr<Allocator> allocator) {
        if (global_allocator_) {
            // logger_->warn("global allocator will only be set once.");
            return false;
        }
        global_allocator_ = std::move(allocator);
        return true;
    }

    LoggerPtr
    logger();

    inline bool
    set_logger(const LoggerPtr& logger) {
        logger_ = logger;
        return true;
    }

private:
    Options() = default;
    ~Options() = default;

    // Deleted copy constructor and assignment operator to prevent copies
    Options(const Options&) = delete;
    Options(const Options&&) = delete;
    Options&
    operator=(const Options&) = delete;

private:
    // In a single query, the space size used to store disk vectors.
    std::atomic<size_t> sector_size_ = 512;

    // The allocator will only be set once.
    std::unique_ptr<Allocator> global_allocator_;

    // The size of the maximum memory allocated each time (default is 128MB)
    std::atomic<size_t> block_size_limit_ = 128 * 1024 * 1024;

    LoggerPtr logger_ = nullptr;
};
using Option = Options;  // for compatibility

}  // namespace vsag
