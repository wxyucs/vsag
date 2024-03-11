
#include <fmt/format-inl.h>
#include <fmt/format.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <exception>
#include <fstream>
#include <ios>
#include <locale>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

#include "../common.h"
#include "../index/diskann.h"
#include "../index/hnsw.h"
#include "./create_index_parameters.h"
#include "vsag/vsag.h"

namespace vsag {

tl::expected<std::shared_ptr<Index>, Error>
Factory::CreateIndex(const std::string& origin_name, const std::string& parameters) {
    try {
        std::string name = origin_name;
        transform(name.begin(), name.end(), name.begin(), ::tolower);
        if (name == INDEX_HNSW) {
            // read parameters from json, throw exception if not exists
            auto params = CreateHnswParameters::FromJson(parameters);
            return std::make_shared<HNSW>(params.space, params.max_degree, params.ef_construction);
        } else if (name == INDEX_DISKANN) {
            // read parameters from json, throw exception if not exists
            auto params = CreateDiskannParameters::FromJson(parameters);
            return std::make_shared<DiskANN>(params.metric,
                                             params.dtype,
                                             params.max_degree,
                                             params.ef_construction,
                                             params.pq_sample_rate,
                                             params.pq_dims,
                                             params.dim,
                                             params.use_preload,
                                             params.use_reference,
                                             params.use_opq);
        } else {
            LOG_ERROR_AND_RETURNS(
                ErrorType::UNSUPPORTED_INDEX, "failed to create index(unsupported): ", name);
        }
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::INVALID_ARGUMENT, "failed to create index(invalid argument): ", e.what());
    } catch (const std::exception& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::UNSUPPORTED_INDEX, "failed to create index(unknown error): ", e.what());
    }
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

}  // namespace vsag
