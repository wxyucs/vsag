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

class Index;
using IndexPtr = std::shared_ptr<Index>;

class Index {
public:
    // [basic methods]

    /**
      * Building index with all vectors
      * 
      * @param base should contains dim, num_elements, ids and vectors
      * @return IDs that failed to insert into the index
      */
    virtual tl::expected<std::vector<int64_t>, Error>
    Build(const Dataset& base) = 0;

    struct Checkpoint {
        BinarySet data;
        bool finish = false;
    };

    /**
      * Provide dynamism for indexes that do not support insertions
      *
      * @param base should contains dim, num_elements, ids and vectors
      * @param binary_set contains intermediate data from the last checkpoint
      * @return intermediate data of the current checkpoint
      */
    virtual tl::expected<Checkpoint, Error>
    ContinueBuild(const Dataset& base, const BinarySet& binary_set) {
        throw std::runtime_error("Index not support partial build");
    }

    /**
      * Adding vectors into a built index, only HNSW supported now, called on other index will cause exception
      * 
      * @param base should contains dim, num_elements, ids and vectors
      * @return IDs that failed to insert into the index
      */
    virtual tl::expected<std::vector<int64_t>, Error>
    Add(const Dataset& base) {
        throw std::runtime_error("Index not support adding vectors");
    }

    /**
      * Remove the vector corresponding to the given ID from the index
      *
      * @param id of the vector that need to be removed from the index
      * @return result indicates whether the remove operation is successful.
      */
    virtual tl::expected<bool, Error>
    Remove(int64_t id) {
        throw std::runtime_error("Index not support delete vector");
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

    /**
     * Pretraining the conjugate graph involves searching with generated queries and providing feedback.
     *
     * @param base_tag_ids is the label of choosen base vectors that need to be enhanced
     * @param k is the number of edges inserted into conjugate graph
     * @return result is the number of successful insertions into conjugate graph
     */
    virtual tl::expected<uint32_t, Error>
    Pretrain(const std::vector<int64_t>& base_tag_ids, uint32_t k, const std::string& parameters) {
        throw std::runtime_error("Index doesn't support pretrain");
    };

    /**
     * Performing feedback on conjugate graph
     *
     * @param query should contains dim, num_elements and vectors
     * @param k is the number of edges inserted into conjugate graph
     * @param global_optimum_tag_id is the label of exact nearest neighbor
     * @return result is the number of successful insertions into conjugate graph
     */
    virtual tl::expected<uint32_t, Error>
    Feedback(const Dataset& query,
             int64_t k,
             const std::string& parameters,
             int64_t global_optimum_tag_id = std::numeric_limits<int64_t>::max()) {
        throw std::runtime_error("Index doesn't support feedback");
    };

public:
    // [serialize/deserialize with binaryset]

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
    // [serialize/deserialize with file stream]

    /**
      * Serialize index to a file stream
      *
      * @param out_stream is a already opened file stream for outputing the serialized index
      */
    virtual tl::expected<void, Error>
    Serialize(std::ostream& out_stream) {
        throw std::runtime_error("Index not support serialize to a file stream");
    }

    /**
      * Deserialize index from a file stream
      * 
      * @param in_stream is a already opened file stream contains serialized index
      * @param length is the length of serialized index(may differ from the actual file size
      *   if there is additional content in the file)
      */
    virtual tl::expected<void, Error>
    Deserialize(std::istream& in_stream) {
        throw std::runtime_error("Index not support deserialize from a file stream");
    }

public:
    // [statstics methods]

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
      * Return the estimated memory required during building
      *
      * @param num_elements denotes the amount of data used to build the index.
      * @return estimated memory required during building.
      */
    virtual int64_t
    GetEstimateBuildMemory(const int64_t num_elements) const {
        throw std::runtime_error("Index not support estimate the memory while building");
    }

    /**
      * Get the statstics from index
      *
      * @return a json string contains runtime statstics of the index.
      */
    virtual std::string
    GetStats() const {
        throw std::runtime_error("Index not support range search");
    }

public:
    virtual ~Index() = default;
};

/**
  * check if the build parameter is valid
  *
  * @return true if the parameter is valid, otherwise error with detail message.
  */
tl::expected<bool, Error>
check_diskann_hnsw_build_parameters(const std::string& json_string);

/**
  * check if the build parameter is valid
  *
  * @return true if the parameter is valid, otherwise error with detail message.
  */
tl::expected<bool, Error>
check_diskann_hnsw_search_parameters(const std::string& json_string);

/**
  * [experimental]
  * generate build index parameters from data size and dim
  *
  * @return the build parameter string
  */
tl::expected<std::string, Error>
generate_build_parameters(std::string metric_type,
                          int64_t num_elements,
                          int64_t dim,
                          bool use_conjugate_graph = false);

}  // namespace vsag
