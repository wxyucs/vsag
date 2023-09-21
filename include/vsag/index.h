#pragma once

#include <cstddef>
#include <cstdint>
#include <queue>
#include <stdexcept>

#include "vsag/binaryset.h"
#include "vsag/dataset.h"
#include "vsag/readerset.h"

namespace vsag {
class Index {
public:
    /**
      * Building index with all vectors
      * 
      * @param base should contains dim, num_elements, ids and vectors
      */
    virtual void
    Build(const Dataset& base) = 0;

    /**
      * Adding vectors into a built index, only HNSW supported now, called on other index will cause exception
      * 
      * @param base should contains dim, num_elements, ids and vectors
      */
    virtual void
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
    virtual Dataset
    KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) = 0;

public:
    /**
      * Serialize index to a set of byte array
      *
      * @return binaryset contains all parts of the index
      */
    virtual BinarySet
    Serialize() {
        throw std::runtime_error("Index not support serialize");
    };

    /**
      * Deserialize index from a set of byte array. Causing exception if this index is not empty
      *
      * @param binaryset contains all parts of the index
      */
    virtual void
    Deserialize(const BinarySet& binary_set) {
        throw std::runtime_error("Index not support deserialize");
    }

    virtual void
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
