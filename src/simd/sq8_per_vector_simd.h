
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
#include <cstdint>
namespace vsag {
#define DECLARE_SQ8_PER_VECTOR(ns)                                                    \
    namespace ns {                                                                    \
    float                                                                             \
    SQ8PerVectorComputeL2Sqr(const float* query, const uint8_t* codes, uint64_t dim); \
    float                                                                             \
    SQ8PerVectorComputeCodesL2Sqr(const uint8_t* a, const uint8_t* b, uint64_t dim);  \
    }
DECLARE_SQ8_PER_VECTOR(generic)
DECLARE_SQ8_PER_VECTOR(sse)
DECLARE_SQ8_PER_VECTOR(avx)
DECLARE_SQ8_PER_VECTOR(avx2)
DECLARE_SQ8_PER_VECTOR(avx512)
DECLARE_SQ8_PER_VECTOR(neon)
DECLARE_SQ8_PER_VECTOR(sve)
#undef DECLARE_SQ8_PER_VECTOR
using SQ8PerVectorComputeType = float (*)(const float*, const uint8_t*, uint64_t);
using SQ8PerVectorComputeCodesType = float (*)(const uint8_t*, const uint8_t*, uint64_t);
extern SQ8PerVectorComputeType SQ8PerVectorComputeL2Sqr;
extern SQ8PerVectorComputeCodesType SQ8PerVectorComputeCodesL2Sqr;
}  // namespace vsag
