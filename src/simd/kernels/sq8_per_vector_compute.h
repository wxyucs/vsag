
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
#include <cmath>
#include <cstdint>
#include <cstring>
namespace vsag::simd {
// Both operands are reconstructed independently, including their offsets.
// memcpy permits odd dimensions and unaligned records; tails never overread codes.
template <typename T, bool adc>
inline float
SQ8PerVectorL2(const float* query, const uint8_t* a, const uint8_t* b, uint64_t dim) {
    float min_a = 0;
    float step_a = 0;
    float min_b;
    float step_b;
    memcpy(&min_b, b, sizeof(float));
    memcpy(&step_b, b + sizeof(float), sizeof(float));
    b += 2 * sizeof(float);
    if constexpr (!adc) {
        memcpy(&min_a, a, sizeof(float));
        memcpy(&step_a, a + sizeof(float), sizeof(float));
        a += 2 * sizeof(float);
    }
    auto sum = T::zero();
    const auto offset_b = T::set1(min_b);
    const auto scale_b = T::set1(step_b);
    const auto offset_a = T::set1(min_a);
    const auto scale_a = T::set1(step_a);
    uint64_t i = 0;
    for (; i + T::Width <= dim; i += T::Width) {
        auto vb = T::fmadd(T::load_u8_as_float(b + i), scale_b, offset_b);
        typename T::FloatVec va;
        if constexpr (adc) {
            va = T::load(query + i);
        } else {
            va = T::fmadd(T::load_u8_as_float(a + i), scale_a, offset_a);
        }
        auto delta = T::sub(va, vb);
        sum = T::fmadd(delta, delta, sum);
    }
    float result = T::reduce_add(sum);
    for (; i < dim; ++i) {
        float va = adc ? query[i] : std::fma(static_cast<float>(a[i]), step_a, min_a);
        float vb = std::fma(static_cast<float>(b[i]), step_b, min_b);
        float delta = va - vb;
        result += delta * delta;
    }
    return result;
}
}  // namespace vsag::simd
