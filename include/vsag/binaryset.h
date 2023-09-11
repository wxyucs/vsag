#pragma once

#include <memory>
#include <set>
#include <string>
#include <unordered_map>

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

private:
    std::unordered_map<std::string, Binary> data_;
};

}  // namespace vsag
