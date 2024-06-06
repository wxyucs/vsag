#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

#include "vsag/dataset.h"

namespace vsag {

class DatasetImpl : public Dataset {
    using var = std::variant<int64_t, const float*, const int8_t*, const int64_t*>;

public:
    DatasetImpl() = default;

    ~DatasetImpl() {
        if (not owner_) {
            return;
        }
        delete[] this->GetIds();
        delete[] this->GetDistances();
        delete[] this->GetInt8Vectors();
        delete[] this->GetFloat32Vectors();
    }

    DatasetImpl(const DatasetImpl&) = delete;
    DatasetImpl&
    operator=(const DatasetImpl&) = delete;

    DatasetImpl(DatasetImpl&& other) noexcept {
        this->owner_ = other.owner_;
        other.owner_ = false;
        this->data_ = other.data_;
        other.data_.clear();
    }

    DatasetPtr
    Owner(bool is_owner) override {
        this->owner_ = is_owner;
        return shared_from_this();
    }

public:
    DatasetPtr
    NumElements(const int64_t num_elements) override {
        this->data_[NUM_ELEMENTS] = num_elements;
        return shared_from_this();
    }

    int64_t
    GetNumElements() const override {
        if (auto iter = this->data_.find(NUM_ELEMENTS); iter != this->data_.end()) {
            return std::get<int64_t>(iter->second);
        }

        return 0;
    }

    DatasetPtr
    Dim(const int64_t dim) override {
        this->data_[DIM] = dim;
        return shared_from_this();
    }

    int64_t
    GetDim() const override {
        if (auto iter = this->data_.find(DIM); iter != this->data_.end()) {
            return std::get<int64_t>(iter->second);
        }

        return 0;
    }

    DatasetPtr
    Ids(const int64_t* ids) override {
        this->data_[IDS] = ids;
        return shared_from_this();
    }

    const int64_t*
    GetIds() const override {
        if (auto iter = this->data_.find(IDS); iter != this->data_.end()) {
            return std::get<const int64_t*>(iter->second);
        }

        return nullptr;
    }

    DatasetPtr
    Distances(const float* dists) override {
        this->data_[DISTS] = dists;
        return shared_from_this();
    }

    const float*
    GetDistances() const override {
        if (auto iter = this->data_.find(DISTS); iter != this->data_.end()) {
            return std::get<const float*>(iter->second);
        }

        return nullptr;
    }

    DatasetPtr
    Int8Vectors(const int8_t* vectors) override {
        this->data_[INT8_VECTORS] = vectors;
        return shared_from_this();
    }

    const int8_t*
    GetInt8Vectors() const override {
        if (auto iter = this->data_.find(INT8_VECTORS); iter != this->data_.end()) {
            return std::get<const int8_t*>(iter->second);
        }

        return nullptr;
    }

    DatasetPtr
    Float32Vectors(const float* vectors) override {
        this->data_[FLOAT32_VECTORS] = vectors;
        return shared_from_this();
    }

    const float*
    GetFloat32Vectors() const override {
        if (auto iter = this->data_.find(FLOAT32_VECTORS); iter != this->data_.end()) {
            return std::get<const float*>(iter->second);
        }

        return nullptr;
    }

private:
    bool owner_ = true;
    std::unordered_map<std::string, var> data_;
};

};  // namespace vsag
