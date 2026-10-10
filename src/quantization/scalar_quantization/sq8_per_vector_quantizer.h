
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

#pragma once

#include "index_common_param.h"
#include "quantization/quantizer.h"
#include "sq8_per_vector_quantizer_parameter.h"

namespace vsag {
// L2 ADC with FP32 queries. Layout: float minimum, float step, dim uint8 codes.
// Metadata is copied with memcpy: neither record addresses nor strides need float alignment.
class SQ8PerVectorQuantizer : public Quantizer<SQ8PerVectorQuantizer> {
public:
    explicit SQ8PerVectorQuantizer(int dim, Allocator* allocator);
    explicit SQ8PerVectorQuantizer(const QuantizerParamPtr& param,
                                   const IndexCommonParam& common_param);
    bool
    TrainImpl(const float* data, uint64_t count);
    bool
    EncodeOneImpl(const float* data, uint8_t* codes) const;
    bool
    DecodeOneImpl(const uint8_t* codes, float* data);
    float
    ComputeImpl(const uint8_t* codes1, const uint8_t* codes2);
    void
    ProcessQueryImpl(const float* query, Computer<SQ8PerVectorQuantizer>& computer) const;
    void
    ComputeDistImpl(Computer<SQ8PerVectorQuantizer>& computer,
                    const uint8_t* codes,
                    float* dists) const;
    void
    SerializeImpl(StreamWriter& writer) {
    }
    void
    DeserializeImpl(StreamReader& reader);
    [[nodiscard]] static std::string
    NameImpl() {
        return QUANTIZATION_TYPE_VALUE_SQ8_PER_VECTOR;
    }
};
}  // namespace vsag
