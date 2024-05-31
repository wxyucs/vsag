#pragma once

#include <memory>
#include <mutex>
#include <string>

namespace vsag {

class Bitset;
using BitsetPtr = std::shared_ptr<Bitset>;

class Bitset {
public:
    /**
      * Generate a random bitset with specified length, indices start from 0.
      * 
      * @param length means the number of bit
      * @return a random bitset
      */
    static BitsetPtr
    Random(int64_t length);

    /**
      * Create an empty bitset object.
      * 
      * @return the bitset
      */
    static BitsetPtr
    Make();

protected:
    Bitset() = default;
    virtual ~Bitset() = default;

    Bitset(const Bitset&) = delete;
    Bitset(Bitset&&) = delete;

public:
    /**
      * Set one bit to specified value.
      * 
      * @param pos the position of the bit to set
      * @param value the value to set the bit to
      */
    virtual void
    Set(int64_t pos, bool value = true) = 0;

    /**
      * Return the value of the bit at position.
      * 
      * @param pos the position of bit to return
      * @return true if the bit is set, false otherwise
      */
    virtual bool
    Test(int64_t pos) = 0;

    /**
      * Returns the number of bits that set to true
      * 
      * @return the number of bit that set to true
      */
    virtual uint64_t
    Count() = 0;

public:
    /**
      * For debugging
      */
    virtual std::string
    Dump() = 0;
};

}  //namespace vsag
