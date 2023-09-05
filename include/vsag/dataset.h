#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>

#include "constants.h"

namespace vsag {

class Dataset {
    using var = std::variant<int64_t, const float*, const int8_t*, const int64_t*>;

public:
    Dataset() = default;
    ~Dataset() {
        if (not owner_) {
            return;
        }
        delete[] this->GetIds();
        delete[] this->GetDistances();
        delete[] this->GetInt8Vectors();
        delete[] this->GetFloat32Vectors();
    }

    void
    SetOwner(bool isOwner) {
        this->owner_ = isOwner;
    }

public:
    void
    SetNumElements(const int64_t num_elements) {
        this->data_[NUM_ELEMENTS] = num_elements;
    }

    int64_t
    GetNumElements() const {
        if (this->data_.find(NUM_ELEMENTS) == this->data_.end()) {
            return 0;
        }
        return std::get<int64_t>(this->data_.at(NUM_ELEMENTS));
    }

    void
    SetDim(const int64_t dim) {
        this->data_[DIM] = dim;
    }

    int64_t
    GetDim() const {
        if (this->data_.find(DIM) == this->data_.end()) {
            return 0;
        }
        return std::get<int64_t>(this->data_.at(DIM));
    }

    void
    SetIds(const int64_t* ids) {
        this->data_[IDS] = ids;
    }

    const int64_t*
    GetIds() const {
        if (this->data_.find(IDS) == this->data_.end()) {
            return nullptr;
        }
        return std::get<const int64_t*>(this->data_.at(IDS));
    }

    void
    SetDistances(const float* dists) {
        this->data_[DISTS] = dists;
    }

    const float*
    GetDistances() const {
        if (this->data_.find(DISTS) == this->data_.end()) {
            return nullptr;
        }
        return std::get<const float*>(this->data_.at(DISTS));
    }

    void
    SetInt8Vectors(const int8_t* vectors) {
        this->data_[INT8_VECTORS] = vectors;
    }

    const int8_t*
    GetInt8Vectors() const {
        if (this->data_.find(INT8_VECTORS) == this->data_.end()) {
            return nullptr;
        }
        return std::get<const int8_t*>(this->data_.at(INT8_VECTORS));
    }

    void
    SetFloat32Vectors(const float* vectors) {
        this->data_[FLOAT32_VECTORS] = vectors;
    }

    const float*
    GetFloat32Vectors() const {
        if (this->data_.find(FLOAT32_VECTORS) == this->data_.end()) {
            return nullptr;
        }
        return std::get<const float*>(this->data_.at(FLOAT32_VECTORS));
    }

private:
    bool owner_ = true;
    std::unordered_map<std::string, var> data_;
};

};  // namespace vsag
