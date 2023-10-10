

#include "vsag/factory.h"

#include <cstdint>
#include <fstream>
#include <ios>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "index/diskann.h"
#include "index/hnsw.h"
#include "index/vamana.h"
namespace vsag {

std::shared_ptr<Index>
Factory::CreateIndex(const std::string& name, const std::string& parameters) {
    nlohmann::json params = nlohmann::json::parse(parameters);

    if (params["dtype"] != "float32") {
        return nullptr;
    }
    if (name == "hnsw") {
        if (not params.contains("hnsw")) {
            throw std::runtime_error("hnsw not found in parameters");
        }
        std::shared_ptr<hnswlib::SpaceInterface> space = nullptr;
        if (params["metric_type"] == "l2") {
            space = std::make_shared<hnswlib::L2Space>(params["dim"]);
        } else {
            space = std::make_shared<hnswlib::InnerProductSpace>(params["dim"]);
        }
        auto index = std::make_shared<HNSW>(space,
                                            params["hnsw"]["max_elements"],
                                            params["hnsw"]["M"],
                                            params["hnsw"]["ef_construction"]);
        if (params["hnsw"].contains("ef_runtime")) {
            index->SetEfRuntime(params["hnsw"]["ef_runtime"]);
        }
        return index;
    } else if (name == "vamana") {
        if (not params.contains("vamana")) {
            throw std::runtime_error("vamana not found in parameters");
        }
        std::string dtype = "float32";
        return std::make_shared<Vamana>(diskann::Metric::FAST_L2, 1, 1, dtype);
    } else if (name == "diskann") {
        if (not params.contains("diskann")) {
            throw std::runtime_error("diskann not found in parameters");
        }
        std::string dtype = "float32";
        std::cout << params.dump() << std::endl;
        auto index = std::make_shared<DiskANN>(diskann::Metric::L2,
                                               dtype,
                                               params["diskann"]["L"],
                                               params["diskann"]["R"],
                                               params["diskann"]["p_val"],
                                               params["diskann"]["disk_pq_dims"]);
        return index;
    } else {
        // not support
        return nullptr;
    }
}

class LocalFileReader : public Reader {
public:
    LocalFileReader(const std::string& filename, int64_t base_offset = 0)
        : filename_(filename),
          file_(std::ifstream(filename, std::ios::binary)),
          base_offset_(base_offset) {
        file_.seekg(0, std::ios::end);
        size_ = file_.tellg();
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
Factory::CreateLocalFileReader(const std::string& filename, int64_t base_offset) {
    return std::make_shared<LocalFileReader>(filename, base_offset);
}

}  // namespace vsag
