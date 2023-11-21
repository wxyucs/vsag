//
// Created by root on 10/9/23.
//
#include <assert.h>

#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>

#include "simpleflat.h"
#include "version.h"
#include "vsag/vsag.h"

namespace vsag {

std::shared_ptr<Index>
Factory::CreateIndex(const std::string& name, const std::string& parameters) {
    nlohmann::json params = nlohmann::json::parse(parameters);
    if (not params.contains("metric_type") and not params.contains("dim")) {
        return nullptr;
    }
    return std::make_shared<SimpleFlat>(params["metric_type"], params["dim"]);
}

class LocalFileReader : public Reader {
public:
    LocalFileReader(const std::string& filename, int64_t base_offset = 0, int64_t size = 0)
        : filename_(filename),
          file_(std::ifstream(filename, std::ios::binary)),
          base_offset_(base_offset),
          size_(size) {
    }

    ~LocalFileReader() {
        file_.close();
    }

    virtual void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        std::lock_guard<std::mutex> lock(mutex_);
        file_.seekg(base_offset_ + offset, std::ios::beg);
        file_.read((char*)dest, len);
    }

    virtual uint64_t
    Size() const override {
        return size_;
    }

private:
    const std::string filename_;
    std::ifstream file_;
    int64_t base_offset_;
    uint64_t size_;
    std::mutex mutex_;
};

std::shared_ptr<Reader>
Factory::CreateLocalFileReader(const std::string& filename, int64_t base_offset, int64_t size) {
    return std::make_shared<LocalFileReader>(filename, base_offset, size);
}

float
l2sqr(const void* vec1, const void* vec2, int64_t dim) {
    float* v1 = (float*)vec1;
    float* v2 = (float*)vec2;

    float res = 0;
    for (int64_t i = 0; i < dim; i++) {
        float t = *v1 - *v2;
        v1++;
        v2++;
        res += t * t;
    }

    return res;
}

BitsetPtr
l2_and_filtering(int64_t dim, int64_t nb, const float* base, const float* query, float threshold) {
    BitsetPtr bp = std::make_shared<Bitset>();
    bp->Extend(nb);

    int64_t count = 0;
    for (int64_t i = 0; i < nb; ++i) {
        const float dist = l2sqr(base + i * dim, query, dim);
        if (dist <= threshold) {
            bp->Set(i, true);
        }
    }

    return bp;
}

std::string
version() {
    return VSAG_VERSION;
}

float
range_search_recall(const float* base,
                    int64_t base_num,
                    const float* query,
                    int64_t data_dim,
                    const int64_t* result_ids,
                    int64_t result_size,
                    float threshold) {
    BitsetPtr groundtruth = l2_and_filtering(data_dim, base_num, base, query, threshold);
    for (int64_t i = 0; i < result_size; ++i) {
        assert(groundtruth->Get(result_ids[i]));
    }
    if (groundtruth->CountOnes() == 0) {
        return 1;
    }
    return (float)(result_size) / groundtruth->CountOnes();
}

float
knn_search_recall(const float* base,
                  int64_t base_num,
                  const float* query,
                  int64_t data_dim,
                  const int64_t* result_ids,
                  int64_t result_size) {
    float recall = 0;
    int64_t nearest_id = 0;
    float nearest_dis = std::numeric_limits<float>::max();
    for (int64_t i = 0; i < base_num; ++i) {
        float dis = l2sqr(base + i * data_dim, query, data_dim);
        if (nearest_dis > dis) {
            nearest_id = i;
            nearest_dis = dis;
        }
    }
    for (int64_t i = 0; i < result_size; ++i) {
        if (result_ids[i] == nearest_id) {
            return 1;
        }
    }
    return 0;
}

}  // namespace vsag
