#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <stdexcept>

#include "bitset.h"
#include "vsag/binaryset.h"
#include "vsag/bitset.h"
#include "vsag/dataset.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/readerset.h"

namespace vsag {

class Index {
public:
    /**
      * Building index with all vectors
      * 
      * @param base should contains dim, num_elements, ids and vectors
      * @return IDs that failed to insert into the index
      */
    virtual tl::expected<std::vector<int64_t>, Error>
    Build(const Dataset& base) = 0;

    /**
      * Adding vectors into a built index, only HNSW supported now, called on other index will cause exception
      * 
      * @param base should contains dim, num_elements, ids and vectors
      * @return IDs that failed to insert into the index
      */
    virtual tl::expected<std::vector<int64_t>, Error>
    Add(const Dataset& base) {
        throw std::runtime_error("Index not support addding vectors");
    }

    /**
      * Performing single KNN search on index
      * 
      * @param query should contains dim, num_elements and vectors
      * @param k the result size of every query
      * @param invalid represents whether an element is filteing out by pre-filter
      * @return result contains 
      *                - num_elements: 1
      *                - ids, distances: length is (num_elements * k)
      */
    virtual tl::expected<Dataset, Error>
    KnnSearch(const Dataset& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const = 0;

    /**
      * Performing single range search on index
      *
      * @param query should contains dim, num_elements and vectors
      * @param radius of search, determines which results will be returned
      * @param invalid represents whether an element is filteing out by pre-filter
      * @return result contains
      *                - num_elements: 1
      *                - dim: the size of results
      *                - ids, distances: length is dim
      */
    virtual tl::expected<Dataset, Error>
    RangeSearch(const Dataset& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid = nullptr) const {
        throw std::runtime_error("Index not support range search");
    }

public:
    /**
      * Serialize index to a set of byte array
      *
      * @return binaryset contains all parts of the index
      */
    virtual tl::expected<BinarySet, Error>
    Serialize() const = 0;

    /**
      * Deserialize index from a set of byte array. Causing exception if this index is not empty
      *
      * @param binaryset contains all parts of the index
      */
    virtual tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) = 0;

    /**
      * Deserialize index from a set of reader array. Causing exception if this index is not empty
      *
      * @param reader contains all parts of the index
      */
    virtual tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) = 0;

public:
    /**
      * Return the number of elements in the index
      *
      * @return number of elements in the index.
      */
    virtual int64_t
    GetNumElements() const = 0;

    /**
      * Return the memory occupied by the index
      *
      * @return number of bytes occupied by the index.
      */
    virtual int64_t
    GetMemoryUsage() const = 0;

    /**
      * Get the statstics from index
      *
      * @return a json string contains runtime statstics of the index.
      */
    virtual std::string
    GetStats() const {
        throw std::runtime_error("Index not support range search");
    }
};

}  // namespace vsag
