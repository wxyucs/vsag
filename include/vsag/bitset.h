#pragma once

#include <algorithm>
#include <bit>
#include <bitset>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace vsag {

class Bitset;
using BitsetPtr = std::shared_ptr<Bitset>;

class Bitset {
    static constexpr int64_t DEFAULT_MEM_LIMIT = 1024 * 1024 * 1024;  // 1MB
public:
    static BitsetPtr
    Random(uint64_t number_of_bits) {
        auto bitset = std::make_shared<Bitset>();
        bitset->Extend(number_of_bits);
        static auto gen =
            std::bind(std::uniform_int_distribution<>(0, 1), std::default_random_engine());
        for (uint64_t i = 0; i < number_of_bits; ++i) {
            bitset->Set(i, gen());
        }
        return bitset;
    }

public:
    Bitset(uint64_t mem_limit = DEFAULT_MEM_LIMIT) : mem_limit_(mem_limit){};
    /**
      * Construct a bitset object from a bitset binary
      *
      * @param src pointer to the memory of the bitset binary
      * @param length the number of bytes
      * @param mem_limit the maximum size of memory that can be used by this bitset
      */
    Bitset(const uint8_t* src, const uint64_t length, uint64_t mem_limit = DEFAULT_MEM_LIMIT)
        : mem_limit_(mem_limit) {
        this->Extend(length * 8);
        memcpy(data_.data(), src, length);
        for (uint8_t num : data_) {
            num_ones_ += std::__popcount(num);
        }
    }
    ~Bitset() = default;

    Bitset(const Bitset&) = delete;
    Bitset(Bitset&&) = delete;

    /**
      * Get a bit value from the bitset
      *
      * @param offset the index of bit to be get, available range: [0, mem_limit * 8)
      */
    bool
    Get(int64_t offset) {
        if (offset < 0) {
            throw std::runtime_error("failed to get bitset: offset(" + std::to_string(offset) +
                                     ") is less than 0");
        }

        int64_t byte_index = offset / 8;
        int64_t bit_index = offset % 8;
        int64_t data_size = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            data_size = data_.size();
        }
        if (byte_index >= data_size) {
            throw std::runtime_error("failed to get from bitset: offset(" + std::to_string(offset) +
                                     ") is greater than capcity(" + std::to_string(data_size * 8) +
                                     ")");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        return (data_.data()[byte_index] & (1 << bit_index)) > 0;
    }

    /**
      * Set a bit value to the bitset
      * 
      * @param offset the index of bit to be set, available range: [0, mem_limit * 8)
      * @param value to be set into the specify bit
      */
    void
    Set(int64_t offset, bool value) {
        if (offset < 0) {
            throw std::runtime_error("failed to set bitset: offset(" + std::to_string(offset) +
                                     ") is less than 0");
        }

        int64_t byte_index = offset / 8;
        int64_t bit_index = offset % 8;
        int64_t data_size = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            data_size = data_.size();
        }
        if (byte_index >= data_size) {
            this->Extend(offset + 1);
        }

        std::lock_guard<std::mutex> lock(mutex_);
        int64_t origin_value = (data_.data()[byte_index] & (1 << bit_index)) > 0;
        if (value) {
            data_.data()[byte_index] |= (1 << bit_index);
            // num_ones plus one only origin_value is 0
            num_ones_ += (1 - origin_value);
        } else {
            data_.data()[byte_index] &= ~(1 << bit_index);
            // num_onse sub one only origin_value is 1
            num_ones_ -= origin_value;
        }
    }

    /**
      * Count the number of bits have been set to 1
      * 
      * @return the number of 1s in the bitset
      */
    uint64_t
    CountOnes() {
        std::lock_guard<std::mutex> lock(mutex_);
        return num_ones_;
    }

    /**
      * Count the number of bits have been set to 0
      * 
      * @return the number of 0s in the bitset
      */
    uint64_t
    CountZeros() {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.size() * 8 - num_ones_;
    }

public:
    /**
      * Get the capacity of this bitset
      * 
      * @return the number of bits can be used
      */
    uint64_t
    Capacity() {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.size() * 8;
    }

    /**
      * Extend the bitset to specify number of bits
      */
    void
    Extend(uint64_t number_of_bits) {
        std::lock_guard<std::mutex> lock(mutex_);
        uint64_t number_of_bytes = number_of_bits / 8 + (number_of_bits % 8 > 0 ? 1 : 0);
        if (number_of_bytes > mem_limit_) {
            throw std::runtime_error(
                "failed to extend bitset: number_of_bytes(" + std::to_string(number_of_bytes) +
                ") is greater than memory limit(" + std::to_string(mem_limit_) + ")");
        }
        if (number_of_bytes < data_.size()) {
            throw std::runtime_error(
                "failed to extend bitset: number_of_bits(" + std::to_string(number_of_bits) +
                ") is less than current capcity(" + std::to_string(data_.size() * 8) + ")");
        }
        data_.resize(number_of_bytes);
    };

    /**
      * For debugging
      */
    std::string
    Dump(int64_t offset) {
        std::lock_guard<std::mutex> lock(mutex_);
        int64_t byte_index = offset / 8;
        uint8_t ch = data_.data()[byte_index];
        std::stringstream ss;
        for (int64_t i = 0; i < 8; ++i) {
            if (ch & (1 << i)) {
                ss << "1";
            } else {
                ss << "0";
            }
        }
        return ss.str();
    }

private:
    const int64_t mem_limit_;
    std::mutex mutex_;
    std::vector<uint8_t> data_;
    int64_t num_ones_ = 0;
};

}  //namespace vsag
