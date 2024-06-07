#pragma once

#include <algorithm>
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
    // Read len bytes from file/memory to the memory pointed to by dest.
    // thread-safe
    virtual void
    Read(uint64_t offset, uint64_t len, void* dest) = 0;

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

    std::vector<std::string>
    GetKeys() const {
        std::vector<std::string> keys;
        keys.resize(data_.size());
        transform(data_.begin(),
                  data_.end(),
                  keys.begin(),
                  [](std::pair<std::string, std::shared_ptr<Reader>> pair) { return pair.first; });
        return keys;
    }

    bool
    Contains(const std::string& key) const {
        return data_.find(key) != data_.end();
    }

private:
    std::unordered_map<std::string, std::shared_ptr<Reader>> data_;
};

}  // namespace vsag
