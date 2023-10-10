#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <stdexcept>

#include "vsag/binaryset.h"
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
      * @return number of elements in the index
      */
    virtual tl::expected<int64_t, index_error>
    Build(const Dataset& base) = 0;

    /**
      * Adding vectors into a built index, only HNSW supported now, called on other index will cause exception
      * 
      * @param base should contains dim, num_elements, ids and vectors
      * @return number of elements have been added into the index
      */
    virtual tl::expected<int64_t, index_error>
    Add(const Dataset& base) {
        throw std::runtime_error("Index not support addding vectors");
    }

    /**
      * Performing batch KNN search on index
      * 
      * @param query should contains dim, num_elements, ids and vectors
      * @param k the result size of every query
      * @return result contains 
      *                - num_elements: equals to num_elements in query
      *                - ids, distances: length is (num_elements * k)
      */
    virtual tl::expected<Dataset, index_error>
    KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) const = 0;

public:
    /**
      * Serialize index to a set of byte array
      *
      * @return binaryset contains all parts of the index
      */
    virtual tl::expected<BinarySet, index_error>
    Serialize() const {
        throw std::runtime_error("Index not support serialize");
    };

    /**
      * Deserialize index from a set of byte array. Causing exception if this index is not empty
      *
      * @param binaryset contains all parts of the index
      */
    virtual tl::expected<void, index_error>
    Deserialize(const BinarySet& binary_set) {
        throw std::runtime_error("Index not support deserialize");
    }

    /**
      * Deserialize index from a set of reader array. Causing exception if this index is not empty
      *
      * @param reader contains all parts of the index
      */
    virtual tl::expected<void, index_error>
    Deserialize(const ReaderSet& reader_set) {
        throw std::runtime_error("Index not support deserialize from reader");
    }

public:
    virtual int64_t
    GetNumElements() const {
        throw std::runtime_error("not implemented");
    }
};

}  // namespace vsag
