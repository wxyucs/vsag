

#include "vsag/factory.h"

#include <fmt/format-inl.h>

#include <cstdint>
#include <fstream>
#include <ios>
#include <locale>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "index/diskann.h"
#include "index/hnsw.h"
#include "vsag/errors.h"

namespace vsag {
bool
case_insensitive_char_compare(char a, char b) {
    return std::tolower(a) == std::tolower(b);
}

ErrorType
convert_error(const char* message) {
    if (std::strcmp(MESSAGE_PARAMETER, message) == 0) {
        return ErrorType::INVALID_ARGUMENT;
    } else {
        return ErrorType::UNKNOWN_ERROR;
    }
}

tl::expected<std::shared_ptr<Index>, Error>
Factory::CreateIndex(const std::string& name, const std::string& parameters) {
    nlohmann::json params = nlohmann::json::parse(parameters);

    if (params[PARAMETER_DTYPE] != DATATYPE_FLOAT32) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT, "only support ", DATATYPE_FLOAT32);
    }

    if (!params.contains(PARAMETER_DIM)) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT, "missing parameter: ", PARAMETER_DIM);
    }
    if (std::equal(name.begin(), name.end(), INDEX_HNSW, case_insensitive_char_compare)) {
        if (not params.contains(INDEX_HNSW)) {
            LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                                  fmt::format("{} not found in parameters", INDEX_HNSW));
        }
        std::shared_ptr<hnswlib::SpaceInterface> space = nullptr;
        if (params[PARAMETER_METRIC_TYPE] == METRIC_L2) {
            space = std::make_shared<hnswlib::L2Space>(params[PARAMETER_DIM]);
        } else if (params[PARAMETER_METRIC_TYPE] == METRIC_IP) {
            space = std::make_shared<hnswlib::InnerProductSpace>(params[PARAMETER_DIM]);
        } else {
            std::string metric = params[PARAMETER_METRIC_TYPE];
            LOG_ERROR_AND_RETURNS(
                ErrorType::INVALID_ARGUMENT,
                fmt::format("{} not support this metric: {}", INDEX_HNSW, metric));
        }
        std::shared_ptr<HNSW> index;
        try {
            index = std::make_shared<HNSW>(space,
                                           params[INDEX_HNSW][HNSW_PARAMETER_M],
                                           params[INDEX_HNSW][HNSW_PARAMETER_CONSTRUCTION]);
        } catch (std::runtime_error e) {
            LOG_ERROR_AND_RETURNS(convert_error(e.what()),
                                  fmt::format("create {} index faild: {}", INDEX_HNSW, e.what()));
        }
        return index;
    } else if (std::equal(name.begin(), name.end(), INDEX_DISKANN, case_insensitive_char_compare)) {
        if (not params.contains(INDEX_DISKANN)) {
            LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                                  fmt::format("{} not found in parameters", INDEX_DISKANN));
        }
        diskann::Metric metric;
        if (params[PARAMETER_METRIC_TYPE] == METRIC_L2) {
            metric = diskann::Metric::L2;
        } else if (params[PARAMETER_METRIC_TYPE] == METRIC_IP) {
            metric = diskann::Metric::INNER_PRODUCT;
        } else {
            std::string metric = params[PARAMETER_METRIC_TYPE];
            LOG_ERROR_AND_RETURNS(
                ErrorType::INVALID_ARGUMENT,
                fmt::format("{} not support this metric: {}", INDEX_DISKANN, metric));
        }

        bool preload = false;
        if (params[INDEX_DISKANN].contains(DISKANN_PARAMETER_PRELOAD)) {
            preload = params[INDEX_DISKANN][DISKANN_PARAMETER_PRELOAD];
        }
        bool use_reference = true;
        if (params[INDEX_DISKANN].contains(DISKANN_PARAMETER_USE_REFERENCE)) {
            use_reference = params[INDEX_DISKANN][DISKANN_PARAMETER_USE_REFERENCE];
        }
        std::shared_ptr<DiskANN> index;
        try {
            index = std::make_shared<DiskANN>(metric,
                                              params[PARAMETER_DTYPE],
                                              params[INDEX_DISKANN][DISKANN_PARAMETER_L],
                                              params[INDEX_DISKANN][DISKANN_PARAMETER_R],
                                              params[INDEX_DISKANN][DISKANN_PARAMETER_P_VAL],
                                              params[INDEX_DISKANN][DISKANN_PARAMETER_DISK_PQ_DIMS],
                                              params[PARAMETER_DIM],
                                              preload,
                                              use_reference);
        } catch (std::runtime_error& e) {
            LOG_ERROR_AND_RETURNS(
                convert_error(e.what()),
                fmt::format("create {} index faild: {}", INDEX_DISKANN, e.what()));
        }
        return index;
    } else {
        LOG_ERROR_AND_RETURNS(ErrorType::UNSUPPORTED_INDEX, "no support this index: ", name);
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
