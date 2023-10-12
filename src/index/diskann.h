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

#include <cstring>
#include <functional>
#include <queue>

#include "vsag/index.h"

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
            size_t disk_pq_dims);

    ~DiskANN() = default;

    tl::expected<int64_t, index_error>
    Build(const Dataset& base) override;

    tl::expected<Dataset, index_error>
    KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) const override;

    tl::expected<BinarySet, index_error>
    Serialize() const override;

    tl::expected<void, index_error>
    Deserialize(const BinarySet& binary_set) override;

    tl::expected<void, index_error>
    Deserialize(const ReaderSet& reader_set) override;

    int64_t
    GetNumElements() const override {
        if (status == EMPTY)
            return 0;
        return index->get_data_num();
    }

    int64_t
    GetMemoryUsage() const override {
        return memory_usage_;
    }

private:
    std::shared_ptr<AlignedFileReader> reader;
    std::shared_ptr<diskann::PQFlashIndex<float>> index;
    std::shared_ptr<diskann::Index<float, int64_t, int64_t>> build_index;
    std::stringstream pq_pivots_stream_;
    std::stringstream disk_pq_compressed_vectors_;
    std::stringstream disk_layout_stream_;
    std::function<void(const std::vector<read_request>&)> batch_read;
    diskann::Metric metric_;
    std::shared_ptr<Reader> disk_layout_reader;
    std::string data_type_;
    int L_ = 200;
    int R_ = 64;
    float p_val_ = 0.5;
    size_t disk_pq_dims_ = 8;
    int64_t memory_usage_ = 0;
    IndexStatus status;
};

}  // namespace vsag
