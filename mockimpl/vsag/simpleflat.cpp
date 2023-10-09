#include "simpleflat.h"

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <nlohmann/json.hpp>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

#include "vsag/readerset.h"

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
    BinarySet bs;
    size_t ids_size = num_elements_ * sizeof(int64_t);
    size_t vector_size = num_elements_ * dim_ * sizeof(float);

    std::shared_ptr<int8_t[]> ids(new int8_t[sizeof(int64_t) * 2 + ids_size]);
    std::shared_ptr<int8_t[]> vectors(new int8_t[vector_size]);

    int8_t* tmp_ptr = ids.get();
    std::memcpy(tmp_ptr, &num_elements_, sizeof(int64_t));

    tmp_ptr += sizeof(int64_t);
    std::memcpy(tmp_ptr, &dim_, sizeof(int64_t));

    tmp_ptr += sizeof(int64_t);
    std::memcpy(tmp_ptr, ids_.data(), ids_size);

    std::memcpy(vectors.get(), data_.data(), vector_size);

    Binary ids_binary{
        .data = ids,
        .size = sizeof(int64_t) + sizeof(int64_t) + ids_size,
    };
    bs.Set(SIMPLEFLAT_IDS, ids_binary);

    Binary vectors_binary{
        .data = vectors,
        .size = vector_size,
    };
    bs.Set(SIMPLEFLAT_VECTORS, vectors_binary);
    return bs;
}

void
SimpleFlat::Deserialize(const BinarySet& binary_set) {
    Binary ids_binary = binary_set.Get(SIMPLEFLAT_IDS);
    Binary data_binary = binary_set.Get(SIMPLEFLAT_VECTORS);

    int8_t* tmp_ptr = ids_binary.data.get();
    std::memcpy(&num_elements_, tmp_ptr, sizeof(int64_t));
    tmp_ptr += sizeof(int64_t);

    std::memcpy(&dim_, tmp_ptr, sizeof(int64_t));
    tmp_ptr += sizeof(int64_t);

    size_t ids_size = num_elements_ * sizeof(int64_t);
    size_t vector_size = num_elements_ * dim_ * sizeof(float);

    if (sizeof(int64_t) + sizeof(int64_t) + ids_size != ids_binary.size ||
        vector_size != data_binary.size) {
        throw std::runtime_error("bs parse error");
    }

    ids_.resize(this->num_elements_);
    std::memcpy(ids_.data(), tmp_ptr, ids_size);

    data_.resize(this->num_elements_ * this->dim_);
    std::memcpy(data_.data(), data_binary.data.get(), vector_size);
}

void
SimpleFlat::Deserialize(const ReaderSet& reader_set) {
    BinarySet bs;

    std::shared_ptr<Reader> vectors_reader = reader_set.Get(SIMPLEFLAT_VECTORS);
    std::shared_ptr<Reader> ids_reader = reader_set.Get(SIMPLEFLAT_IDS);

    std::shared_ptr<int8_t[]> vectors(new int8_t[vectors_reader->Size()]);
    std::shared_ptr<int8_t[]> ids(new int8_t[ids_reader->Size()]);

    vectors_reader->Read(0, vectors_reader->Size(), vectors.get());
    ids_reader->Read(0, ids_reader->Size(), ids.get());

    Binary vectors_binary{
        .data = vectors,
        .size = vectors_reader->Size(),
    };
    bs.Set(SIMPLEFLAT_VECTORS, vectors_binary);

    Binary ids_binary{
        .data = ids,
        .size = ids_reader->Size(),
    };
    bs.Set(SIMPLEFLAT_IDS, ids_binary);

    Deserialize(bs);
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
