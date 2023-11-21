#include "simpleflat.h"

#include <cmath>
#include <cstring>
#include <nlohmann/json.hpp>

#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/readerset.h"

namespace vsag {

SimpleFlat::SimpleFlat(const std::string& metric_type, int64_t dim)
    : metric_type_(metric_type), dim_(dim) {
}

tl::expected<int64_t, index_error>
SimpleFlat::Build(const Dataset& base) {
    if (not this->data_.empty()) {
        return tl::unexpected(index_error::build_twice);
    }

    if (this->dim_ != base.GetDim()) {
        return tl::unexpected(index_error::dimension_not_equal);
    }

    this->num_elements_ = base.GetNumElements();

    this->ids_.resize(this->num_elements_);
    std::memcpy(this->ids_.data(), base.GetIds(), base.GetNumElements() * sizeof(int64_t));

    this->data_.resize(this->num_elements_ * this->dim_);
    std::memcpy(this->data_.data(),
                base.GetFloat32Vectors(),
                base.GetNumElements() * this->dim_ * sizeof(float));

    return this->GetNumElements();
}

tl::expected<int64_t, index_error>
SimpleFlat::Add(const Dataset& base) {
    if (not this->data_.empty()) {
        if (this->dim_ != base.GetDim()) {
            return tl::unexpected(index_error::dimension_not_equal);
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

    return base.GetNumElements();
}

tl::expected<Dataset, index_error>
SimpleFlat::KnnSearch(const Dataset& query,
                      int64_t k,
                      const std::string& parameters,
                      BitsetPtr invalid) const {
    int64_t dim = query.GetDim();
    k = std::min(k, GetNumElements());
    int64_t num_elements = query.GetNumElements();
    if (num_elements != 1) {
        return tl::unexpected(index_error::internal_error);
    }
    if (this->dim_ != dim) {
        return tl::unexpected(index_error::dimension_not_equal);
    }

    std::vector<rs> results;
    results = knn_search(query.GetFloat32Vectors(), k, invalid);

    Dataset dataset;
    int64_t* ids = new int64_t[results.size()];
    float* dists = new float[results.size()];
    dataset.Dim(results.size()).NumElements(1).Ids(ids).Distances(dists);
    for (int64_t i = results.size() - 1; i >= 0; --i) {
        ids[i] = results[results.size() - 1 - i].second;
        dists[i] = results[results.size() - 1 - i].first;
    }

    return std::move(dataset);
}

tl::expected<Dataset, index_error>
SimpleFlat::RangeSearch(const Dataset& query,
                        float radius,
                        const std::string& parameters,
                        BitsetPtr invalid) const {
    int64_t nq = query.GetNumElements();
    int64_t dim = query.GetDim();
    if (this->dim_ != dim) {
        return tl::unexpected(index_error::dimension_not_equal);
    }

    if (nq != 1) {
        return tl::unexpected(index_error::internal_error);
    }
    auto result = range_search(query.GetFloat32Vectors(), radius, invalid);

    int64_t* ids = new int64_t[result.size()];
    float* dists = new float[result.size()];
    for (int64_t kk = 0; kk < result.size(); ++kk) {
        ids[kk] = result[result.size() - 1 - kk].second;
        dists[kk] = result[result.size() - 1 - kk].first;
    }

    Dataset results;
    results.NumElements(1).Dim(result.size()).Ids(ids).Distances(dists);
    return std::move(results);
}

tl::expected<BinarySet, index_error>
SimpleFlat::Serialize() const {
    try {
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
    } catch (const std::bad_alloc& e) {
        return tl::unexpected(index_error::no_enough_memory);
    }
}

tl::expected<void, index_error>
SimpleFlat::Deserialize(const BinarySet& binary_set) {
    Binary ids_binary = binary_set.Get(SIMPLEFLAT_IDS);
    Binary data_binary = binary_set.Get(SIMPLEFLAT_VECTORS);

    int8_t* tmp_ptr = ids_binary.data.get();
    std::memcpy(&num_elements_, tmp_ptr, sizeof(int64_t));
    tmp_ptr += sizeof(int64_t);

    int64_t tmp_dim;
    std::memcpy(&tmp_dim, tmp_ptr, sizeof(int64_t));

    if (tmp_dim != dim_) {
        return tl::unexpected(index_error::dimension_not_equal);
    }

    tmp_ptr += sizeof(int64_t);

    size_t ids_size = num_elements_ * sizeof(int64_t);
    size_t vector_size = num_elements_ * dim_ * sizeof(float);

    if (sizeof(int64_t) + sizeof(int64_t) + ids_size != ids_binary.size ||
        vector_size != data_binary.size) {
        return tl::unexpected(index_error::invalid_binary);
    }

    ids_.resize(this->num_elements_);
    std::memcpy(ids_.data(), tmp_ptr, ids_size);

    data_.resize(this->num_elements_ * this->dim_);
    std::memcpy(data_.data(), data_binary.data.get(), vector_size);

    return {};
}

tl::expected<void, index_error>
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

    return {};
}
std::vector<SimpleFlat::rs>
SimpleFlat::knn_search(const float* query, int64_t k, BitsetPtr invalid) const {
    std::priority_queue<SimpleFlat::rs> q;
    for (int64_t i = 0; i < this->num_elements_; ++i) {
        if (invalid && invalid->Get(i)) {
            continue;
        }
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

std::vector<SimpleFlat::rs>
SimpleFlat::range_search(const float* query, float radius, BitsetPtr invalid) const {
    std::priority_queue<SimpleFlat::rs> q;
    for (int64_t i = 0; i < this->num_elements_; ++i) {
        if (invalid && invalid->Get(i)) {
            continue;
        }
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

        if (distance < radius) {
            q.push(std::make_pair(distance, this->ids_[i]));
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
    return dist;
}

float
SimpleFlat::ip(const float* v1, const float* v2, int64_t dim) {
    float dist = 0;
    float mold_v1 = 0;
    float mold_v2 = 0;
    for (int64_t i = 0; i < dim; ++i) {
        mold_v1 += std::pow(v1[i], 2);
        mold_v2 += std::pow(v2[i], 2);
        dist += v1[i] * v2[i];
    }
    dist = (1 - dist / std::sqrt(mold_v1 * mold_v2)) / 2 + 0.5;
    return dist;
}

float
SimpleFlat::cosine(const float* v1, const float* v2, int64_t dim) {
    return 0;
}

std::string
SimpleFlat::GetStats() const {
    nlohmann::json j;
    j["num_elements"] = num_elements_;
    j["dim"] = dim_;
    return j.dump();
}

}  // namespace vsag
