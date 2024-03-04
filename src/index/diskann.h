//
// Created by jinjiabao on 9/6/23.
//

#pragma once

#include <abstract_index.h>
#include <disk_utils.h>
#include <index.h>
#include <libaio.h>
#include <linux_aligned_file_reader.h>
#include <omp.h>
#include <pq_flash_index.h>

#include <functional>
#include <map>
#include <queue>
#include <string>

#include "../common.h"
#include "../utils.h"
#include "vsag/index.h"
#include "vsag/option.h"

namespace vsag {

enum IndexStatus { EMPTY = 0, MEMORY = 1, HYBRID = 2 };

class DiskANN : public Index {
public:
    using rs = std::pair<float, size_t>;

    // offset: uint64, len: uint64, dest: void*
    using read_request = std::tuple<uint64_t, uint64_t, void*>;

    DiskANN(diskann::Metric metric,
            std::string data_type,
            int L,
            int R,
            float p_val,
            size_t disk_pq_dims,
            int64_t dim,
            bool preload,
            bool use_reference);

    ~DiskANN() = default;

    tl::expected<std::vector<int64_t>, Error>
    Build(const Dataset& base) override {
        SAFE_CALL(return this->build(base));
    }

    tl::expected<Dataset, Error>
    KnnSearch(const Dataset& query,
              int64_t k,
              const std::string& parameters,
              BitsetPtr invalid = nullptr) const override {
        SAFE_CALL(return this->knn_search(query, k, parameters, invalid));
    }

    tl::expected<Dataset, Error>
    RangeSearch(const Dataset& query,
                float radius,
                const std::string& parameters,
                BitsetPtr invalid = nullptr) const override {
        SAFE_CALL(return this->range_search(query, radius, parameters, invalid));
    }

public:
    tl::expected<BinarySet, Error>
    Serialize() const override {
        SAFE_CALL(return this->serialize());
    }

    tl::expected<void, Error>
    Deserialize(const BinarySet& binary_set) override {
        SAFE_CALL(return this->deserialize(binary_set));
    }

    tl::expected<void, Error>
    Deserialize(const ReaderSet& reader_set) override {
        SAFE_CALL(return this->deserialize(reader_set));
    }

public:
    int64_t
    GetNumElements() const override {
        if (status_ == EMPTY)
            return 0;
        return index_->get_data_num();
    }

    int64_t
    GetMemoryUsage() const override {
        if (status_ == MEMORY) {
            return index_->get_memory_usage() + disk_layout_stream_.str().size() +
                   pq_pivots_stream_.str().size() + disk_layout_stream_.str().size();
        } else if (status_ == HYBRID) {
            return index_->get_memory_usage();
        }
        return 0;
    }

    std::string
    GetStats() const override;

private:
    tl::expected<std::vector<int64_t>, Error>
    build(const Dataset& base);

    tl::expected<Dataset, Error>
    knn_search(const Dataset& query,
               int64_t k,
               const std::string& parameters,
               BitsetPtr invalid = nullptr) const;

    tl::expected<Dataset, Error>
    range_search(const Dataset& query,
                 float radius,
                 const std::string& parameters,
                 BitsetPtr invalid = nullptr) const;

    tl::expected<BinarySet, Error>
    serialize() const;

    tl::expected<void, Error>
    deserialize(const BinarySet& binary_set);

    tl::expected<void, Error>
    deserialize(const ReaderSet& reader_set);

private:
    std::shared_ptr<AlignedFileReader> reader_;
    std::shared_ptr<diskann::PQFlashIndex<float, int64_t>> index_;
    std::shared_ptr<diskann::Index<float, int64_t, int64_t>> build_index_;
    std::stringstream pq_pivots_stream_;
    std::stringstream disk_pq_compressed_vectors_;
    std::stringstream disk_layout_stream_;
    std::stringstream tag_stream_;
    std::stringstream graph_stream_;

    std::function<void(const std::vector<read_request>&)> batch_read_;
    std::function<bool(uint32_t)> filter = nullptr;
    diskann::Metric metric_;
    std::shared_ptr<Reader> disk_layout_reader_;
    std::string data_type_;

    int L_ = 200;
    int R_ = 64;
    float p_val_ = 0.5;
    size_t disk_pq_dims_ = 8;
    size_t sector_len_ = 4096;

    int64_t dim_;
    bool use_reference_ = true;
    bool preload_;
    IndexStatus status_;

private:  // Request Statistics
    mutable std::mutex stats_mutex_;

    mutable std::map<std::string, WindowResultQueue> result_queues_;
};

}  // namespace vsag
