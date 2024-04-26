#pragma once

#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace vsag {

struct Binary {
    std::shared_ptr<int8_t[]> data;
    size_t size = 0;
};

class BinarySet {
public:
    BinarySet() = default;

    ~BinarySet() = default;

    void
    Set(const std::string& name, Binary binary) {
        data_[name] = binary;
    }

    Binary
    Get(const std::string& name) const {
        if (data_.find(name) == data_.end()) {
            return Binary();
        }
        return data_.at(name);
    }

    std::vector<std::string>
    GetKeys() const {
        std::vector<std::string> keys;
        keys.resize(data_.size());
        transform(data_.begin(), data_.end(), keys.begin(), [](auto pair) { return pair.first; });
        return keys;
    }

    bool
    Contains(const std::string& key) const {
        return data_.find(key) != data_.end();
    }

private:
    std::unordered_map<std::string, Binary> data_;
};

}  // namespace vsag
