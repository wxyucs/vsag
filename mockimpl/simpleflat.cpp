#include "simpleflat.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

#include "nlohmann/json.hpp"

namespace vsag {

SimpleFlat::SimpleFlat(const std::string& metric_type) : metric_type_(metric_type) {
}

void
SimpleFlat::Build(const Dataset& base) {
    if (not this->data_.empty()) {
        throw std::runtime_error("cannot build index twice, use ::Add() to adding vectors");
    }

    this->num_elements_ = base.GetNumElements();
    this->dim_ = base.GetDim();

    this->ids_.resize(this->num_elements_);
    std::memcpy(this->ids_.data(), base.GetIds(), base.GetNumElements() * sizeof(int64_t));

    this->data_.resize(this->num_elements_ * this->dim_);
    std::memcpy(this->data_.data(),
                base.GetFloat32Vectors(),
                base.GetNumElements() * this->dim_ * sizeof(float));
}

void
SimpleFlat::Add(const Dataset& base) {
    if (not this->data_.empty()) {
        if (this->dim_ != base.GetDim()) {
            throw std::runtime_error("cannot adding vector(dim=" + std::to_string(base.GetDim()) +
                                     ") into index(dim=" + std::to_string(this->dim_) + ")");
        }
    }

    int64_t num_elements_existed = this->num_elements_;
    this->num_elements_ += base.GetNumElements();

    this->ids_.resize(this->num_elements_);
    std::memcpy(this->ids_.data() + num_elements_existed,
                base.GetIds(),
                base.GetNumElements() * sizeof(int64_t));

    this->data_.resize(this->num_elements_ * this->dim_);
    std::memcpy(this->data_.data() + num_elements_existed * this->dim_,
                base.GetFloat32Vectors(),
                base.GetNumElements() * this->dim_ * sizeof(float));
}

Dataset
SimpleFlat::KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) {
    int64_t nq = query.GetNumElements();
    int64_t dim = query.GetDim();
    if (this->dim_ != dim) {
        throw std::runtime_error("dimension not equal: index(" + std::to_string(this->dim_) +
                                 ") query(" + std::to_string(dim) + ")");
    }

    int64_t* ids = new int64_t[nq * k];
    float* dists = new float[nq * k];
    for (int64_t i = 0; i < query.GetNumElements(); ++i) {
        auto result = knn_search(query.GetFloat32Vectors() + i * dim, k);
        for (int64_t kk = 0; kk < k; ++kk) {
            ids[i * k + kk] = result[kk].second;
            dists[i * k + kk] = result[kk].first;
        }
    }

    Dataset results;
    results.SetIds(ids);
    results.SetDistances(dists);
    return std::move(results);
}

BinarySet
SimpleFlat::Serialize() {
    throw std::runtime_error("not support yet");
}

void
SimpleFlat::Deserialize(const BinarySet& binary_set) {
    throw std::runtime_error("not support yet");
}

std::vector<SimpleFlat::rs>
SimpleFlat::knn_search(const float* query, int64_t k) const {
    std::priority_queue<SimpleFlat::rs> q;
    for (int64_t i = 0; i < this->num_elements_; ++i) {
        const float* base = data_.data() + i * this->dim_;
        float distance = 0.0f;
        if (this->metric_type_ == "l2") {
            distance = l2(base, query, this->dim_);
        } else if (this->metric_type_ == "ip") {
            distance = ip(base, query, this->dim_);
        } else if (this->metric_type_ == "cosine") {
            distance = cosine(base, query, this->dim_);
        } else {
            distance = 0;
        }

        q.push(std::make_pair(distance, this->ids_[i]));
        if (q.size() > k) {
            q.pop();
        }
    }

    std::vector<SimpleFlat::rs> results;
    while (not q.empty()) {
        results.push_back(q.top());
        q.pop();
    }
    return results;
}

float
SimpleFlat::l2(const float* v1, const float* v2, int64_t dim) {
    float dist = 0;
    for (int64_t i = 0; i < dim; ++i) {
        dist += std::pow(v1[i] - v2[i], 2);
    }
    dist = std::sqrt(dist);
    return dist;
}

float
SimpleFlat::ip(const float* v1, const float* v2, int64_t dim) {
    return 0;
}

float
SimpleFlat::cosine(const float* v1, const float* v2, int64_t dim) {
    return 0;
}

}  // namespace vsag
