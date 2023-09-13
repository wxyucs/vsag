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
#include <queue>

#include "vsag/index.h"

namespace vsag {

class DiskANN : public Index {
public:
    using rs = std::pair<float, size_t>;
    DiskANN(diskann::Metric metric,
            std::string data_type,
            int L,
            int R,
            float p_val,
            size_t disk_pq_dims);

    void
    Build(const Dataset& base) override;

    Dataset
    KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) override;

    void
    SetEfRuntime(int64_t ef_runtime);

    void
    Deserialize(const BinarySet& binary_set) override;

    BinarySet 
    Serialize() override;


private:
    std::shared_ptr<AlignedFileReader> reader;
    std::shared_ptr<diskann::PQFlashIndex<float>> index;
    std::stringstream pq_pivots_stream_;
    std::stringstream disk_pq_compressed_vectors_;
    diskann::Metric metric_;
    std::string data_type_;
    int L_ = 200;
    int R_ = 64;
    float p_val_ = 0.5;
    size_t disk_pq_dims_ = 8;
    std::string disk_layout_file = "/tmp/index.out";
};

}  // namespace vsag
