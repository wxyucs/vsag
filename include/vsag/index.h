#pragma once

#include <queue>
#include <stdexcept>

// #include "vsag/binaryset.h"
#include "vsag/dataset.h"

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
      * @return result contains ids and distances
      */
    virtual Dataset
    KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) = 0;

    //public:
    //    virtual BinarySet
    //    Serialize() {
    //        throw std::runtime_error("Index not support serialize");
    //    };
    //
    //    virtual void
    //    Deserialize(const BinarySet& binary_set) {
    //        throw std::runtime_error("Index not support deserialize");
    //    }
};

}  // namespace vsag
