#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace vsag {

class Reader {
public:
    Reader() = default;
    ~Reader() = default;

public:
    // offset: uint64, len: uint64, dest: void*
    using read_request = std::tuple<uint64_t, uint64_t, void*>;

    virtual void
    Read(uint64_t offset, uint64_t len, void* dest) = 0;

    virtual void
    BatchRead(const std::vector<read_request>& requests) = 0;

    virtual uint64_t
    Size() const = 0;
};

class ReaderSet {
public:
    ReaderSet() = default;
    ~ReaderSet() = default;

    void
    Set(const std::string& name, std::shared_ptr<Reader> reader) {
        data_[name] = reader;
    }

    std::shared_ptr<Reader>
    Get(const std::string& name) const {
        if (data_.find(name) == data_.end()) {
            return nullptr;
        }
        return data_.at(name);
    }

private:
    std::unordered_map<std::string, std::shared_ptr<Reader>> data_;
};

}  // namespace vsag
