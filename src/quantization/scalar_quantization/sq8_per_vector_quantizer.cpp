
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "sq8_per_vector_quantizer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "simd/sq8_per_vector_simd.h"

namespace vsag {
SQ8PerVectorQuantizer::SQ8PerVectorQuantizer(int dim, Allocator* allocator)
    : Quantizer<SQ8PerVectorQuantizer>(dim, allocator) {
    CHECK_ARGUMENT(dim > 0, "sq8_per_vector requires a positive dimension");
    this->code_size_ = this->dim_ + 2 * sizeof(float);
    this->query_code_size_ = this->dim_ * sizeof(float);
    this->metric_ = MetricType::METRIC_TYPE_L2SQR;
}

SQ8PerVectorQuantizer::SQ8PerVectorQuantizer(const QuantizerParamPtr& param,
                                             const IndexCommonParam& common_param)
    : SQ8PerVectorQuantizer(static_cast<int>(common_param.dim_), common_param.allocator_.get()) {
    CHECK_ARGUMENT(common_param.dim_ <= std::numeric_limits<int>::max(),
                   "sq8_per_vector dimension exceeds supported range");
    CHECK_ARGUMENT(common_param.metric_ == MetricType::METRIC_TYPE_L2SQR,
                   "sq8_per_vector supports only L2");
}

bool
SQ8PerVectorQuantizer::TrainImpl(const float* data, uint64_t count) {
    this->is_trained_ = true;
    return true;
}

bool
SQ8PerVectorQuantizer::EncodeOneImpl(const float* data, uint8_t* codes) const {
    float minimum = data[0];
    float maximum = data[0];
    for (uint64_t i = 0; i < this->dim_; ++i) {
        uint32_t bits;
        memcpy(&bits, data + i, sizeof(bits));
        CHECK_ARGUMENT((bits & 0x7f800000U) != 0x7f800000U, "sq8_per_vector requires finite input");
        minimum = std::min(minimum, data[i]);
        maximum = std::max(maximum, data[i]);
    }
    // Compute the range in double to avoid overflow for finite float endpoints.
    const double range = static_cast<double>(maximum) - minimum;
    const auto step = static_cast<float>(range / 255.0);
    memcpy(codes, &minimum, sizeof(float));
    memcpy(codes + sizeof(float), &step, sizeof(float));
    for (uint64_t i = 0; i < this->dim_; ++i) {
        const double scaled = step == 0.0F ? 0.0 : (static_cast<double>(data[i]) - minimum) / step;
        codes[2 * sizeof(float) + i] =
            static_cast<uint8_t>(std::clamp(std::round(scaled), 0.0, 255.0));
    }
    return true;
}

bool
SQ8PerVectorQuantizer::DecodeOneImpl(const uint8_t* codes, float* data) {
    float minimum;
    float step;
    memcpy(&minimum, codes, sizeof(float));
    memcpy(&step, codes + sizeof(float), sizeof(float));
    for (uint64_t i = 0; i < this->dim_; ++i) {
        data[i] = std::fma(static_cast<float>(codes[2 * sizeof(float) + i]), step, minimum);
    }
    return true;
}

float
SQ8PerVectorQuantizer::ComputeImpl(const uint8_t* codes1, const uint8_t* codes2) {
    return SQ8PerVectorComputeCodesL2Sqr(codes1, codes2, this->dim_);
}

void
SQ8PerVectorQuantizer::ProcessQueryImpl(const float* query,
                                        Computer<SQ8PerVectorQuantizer>& computer) const {
    if (computer.buf_ == nullptr) {
        computer.buf_ = static_cast<uint8_t*>(this->allocator_->Allocate(this->query_code_size_));
    }
    memcpy(computer.buf_, query, this->query_code_size_);
}

void
SQ8PerVectorQuantizer::ComputeDistImpl(Computer<SQ8PerVectorQuantizer>& computer,
                                       const uint8_t* codes,
                                       float* dists) const {
    *dists =
        SQ8PerVectorComputeL2Sqr(reinterpret_cast<const float*>(computer.buf_), codes, this->dim_);
}

void
SQ8PerVectorQuantizer::DeserializeImpl(StreamReader& reader) {
    const bool valid = this->dim_ > 0 && this->dim_ <= std::numeric_limits<int>::max() &&
                       this->code_size_ == this->dim_ + 2 * sizeof(float) &&
                       this->metric_ == MetricType::METRIC_TYPE_L2SQR;
    CHECK_ARGUMENT(valid, "invalid sq8_per_vector metadata");
    this->query_code_size_ = this->dim_ * sizeof(float);
}
}  // namespace vsag
