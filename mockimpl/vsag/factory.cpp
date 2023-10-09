//
// Created by root on 10/9/23.
//
#include "vsag/factory.h"

#include <fstream>
#include <mutex>
#include <nlohmann/json.hpp>

#include "simpleflat.h"
namespace vsag {

std::shared_ptr<Index>
Factory::CreateIndex(const std::string& name, const std::string& parameters) {
    nlohmann::json params = nlohmann::json::parse(parameters);
    if (not params.contains("metric_type")) {
        return nullptr;
    }
    return std::make_shared<SimpleFlat>(params["metric_type"]);
}

class LocalFileReader : public Reader {
public:
    LocalFileReader(const std::string& filename)
        : filename_(filename), file_(std::ifstream(filename, std::ios::binary)) {
        file_.seekg(0, std::ios::end);
        size_ = file_.tellg();
    }

    ~LocalFileReader() {
        file_.close();
    }

    virtual void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        std::lock_guard<std::mutex> lock(mutex_);
        file_.seekg(offset, std::ios::beg);
        file_.read((char*)dest, len);
    }

    virtual uint64_t
    Size() const override {
        return size_;
    }

private:
    const std::string filename_;
    std::ifstream file_;
    uint64_t size_;
    std::mutex mutex_;
};

std::shared_ptr<Reader>
Factory::CreateLocalFileReader(const std::string& filename) {
    return std::make_shared<LocalFileReader>(filename);
}

}  // namespace vsag
