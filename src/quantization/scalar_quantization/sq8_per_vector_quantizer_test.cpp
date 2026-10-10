
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

#include <catch2/catch_approx.hpp>
#include <cmath>
#include <limits>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "quantization/quantizer_test.h"
#include "simd/simd_status.h"
#include "simd/sq8_per_vector_simd.h"
#include "unittest.h"
#include "vsag/vsag.h"

using namespace vsag;

TEST_CASE("Per-vector SQ8 decoded reference and SIMD", "[ut][SQ8PerVector]") {
    auto allocator = SafeAllocator::FactoryDefaultAllocator();
    REQUIRE_THROWS(SQ8PerVectorQuantizer(0, allocator.get()));
    REQUIRE_THROWS(SQ8PerVectorQuantizer(-1, allocator.get()));
    IndexCommonParam unsupported;
    unsupported.dim_ = 17;
    unsupported.allocator_ = allocator;
    unsupported.metric_ = MetricType::METRIC_TYPE_IP;
    REQUIRE_THROWS(
        SQ8PerVectorQuantizer(QuantizerParameter::CreateDefault("sq8_per_vector"), unsupported));
    for (int dim : {1, 3, 7, 16, 31, 64, 65, 960}) {
        SQ8PerVectorQuantizer quant(dim, allocator.get());
        REQUIRE(quant.GetCodeSize() == dim + 8);
        REQUIRE(quant.GetQueryCodeSize() == dim * sizeof(float));
        REQUIRE(quant.Train(nullptr, 0));
        std::vector<float> a(dim), b(dim), query(dim), da(dim), db(dim);
        // Deliberately unaligned starts and odd strides exercise metadata copies.
        std::vector<uint8_t> storage(2 * quant.GetCodeSize() + 1);
        auto* ca = storage.data() + 1;
        auto* cb = ca + quant.GetCodeSize();
        for (int pattern = 0; pattern < 4; ++pattern) {
            for (int i = 0; i < dim; ++i) {
                a[i] = pattern == 0   ? 0.0F
                       : pattern == 1 ? -3.5F
                                      : std::sin(i * 1.7F) * 2.0F - 5.0F;
                b[i] = pattern < 2 ? 8.0F : std::cos(i * 0.9F) * 10.0F + 20.0F;
                query[i] = std::sin(i * 0.7F) * 5.0F;
            }
            REQUIRE(quant.EncodeOne(a.data(), ca));
            REQUIRE(quant.EncodeOne(b.data(), cb));
            REQUIRE(quant.DecodeOne(ca, da.data()));
            REQUIRE(quant.DecodeOne(cb, db.data()));
            double code_ref = 0, query_ref = 0;
            for (int i = 0; i < dim; ++i) {
                REQUIRE(std::abs(da[i] - a[i]) <= (pattern < 2 ? 0.0F : 4.0F / 510 + 1e-5F));
                REQUIRE(std::abs(db[i] - b[i]) <= (pattern < 2 ? 0.0F : 20.0F / 510 + 1e-5F));
                code_ref += std::pow(static_cast<double>(da[i]) - db[i], 2);
                query_ref += std::pow(static_cast<double>(query[i]) - db[i], 2);
            }
            std::vector<SQ8PerVectorComputeType> adc{generic::SQ8PerVectorComputeL2Sqr,
                                                     SQ8PerVectorComputeL2Sqr};
            std::vector<SQ8PerVectorComputeCodesType> sdc{generic::SQ8PerVectorComputeCodesL2Sqr,
                                                          SQ8PerVectorComputeCodesL2Sqr};
#define CHECK_ISA(isa, check)                              \
    if (SimdStatus::check()) {                             \
        adc.push_back(isa::SQ8PerVectorComputeL2Sqr);      \
        sdc.push_back(isa::SQ8PerVectorComputeCodesL2Sqr); \
    }
            CHECK_ISA(sse, SupportSSE)
            CHECK_ISA(avx, SupportAVX)
            CHECK_ISA(avx2, SupportAVX2)
            CHECK_ISA(avx512, SupportAVX512)
            CHECK_ISA(neon, SupportNEON)
            CHECK_ISA(sve, SupportSVE)
#undef CHECK_ISA
            for (auto fn : adc) {
                REQUIRE(fn(query.data(), cb, dim) == Catch::Approx(query_ref).epsilon(2e-5));
            }
            for (auto fn : sdc) {
                REQUIRE(fn(ca, cb, dim) == Catch::Approx(code_ref).epsilon(2e-5));
                REQUIRE(fn(cb, ca, dim) == Catch::Approx(code_ref).epsilon(2e-5));
                REQUIRE(fn(ca, ca, dim) == 0.0F);
            }
            auto computer = quant.FactoryComputer();
            computer->SetQuery(query.data());
            float distance;
            computer->ComputeDist(cb, &distance);
            REQUIRE(distance == Catch::Approx(query_ref).epsilon(2e-5));
            REQUIRE(quant.Compute(ca, cb) == Catch::Approx(code_ref).epsilon(2e-5));
            SQ8PerVectorQuantizer restored(1, allocator.get());
            test_serializion(quant, restored);
            REQUIRE(restored.GetDim() == dim);
            REQUIRE(restored.GetQueryCodeSize() == quant.GetQueryCodeSize());
            REQUIRE(restored.Name() == "sq8_per_vector");
            REQUIRE(restored.Compute(ca, cb) == Catch::Approx(code_ref).epsilon(2e-5));
            auto restored_computer = restored.FactoryComputer();
            restored_computer->SetQuery(query.data());
            restored_computer->ComputeDist(cb, &distance);
            REQUIRE(distance == Catch::Approx(query_ref).epsilon(2e-5));
        }
        a[0] = std::numeric_limits<float>::infinity();
        REQUIRE_THROWS(quant.EncodeOne(a.data(), ca));
    }
}

TEST_CASE("Per-vector SQ8 HGraph and metric validation", "[ut][SQ8PerVector]") {
    const std::string config = R"({"dtype":"float32","metric_type":"l2","dim":17,
        "index_param":{"base_quantization_type":"sq8_per_vector","use_reorder":false,
        "max_degree":16,"ef_construction":100,"build_thread_count":1}})";
    auto result = Factory::CreateIndex("hgraph", config);
    REQUIRE(result.has_value());
    auto index = result.value();
    auto vectors = fixtures::generate_vectors(100, 17);
    std::vector<int64_t> ids(100);
    std::iota(ids.begin(), ids.end(), 0);
    auto base = Dataset::Make()
                    ->NumElements(100)
                    ->Dim(17)
                    ->Float32Vectors(vectors.data())
                    ->Ids(ids.data())
                    ->Owner(false);
    REQUIRE(index->Build(base).has_value());
    auto query =
        Dataset::Make()->NumElements(1)->Dim(17)->Float32Vectors(vectors.data())->Owner(false);
    const std::string search = R"({"hgraph":{"ef_search":100}})";
    auto before = index->KnnSearch(query, 10, search);
    REQUIRE(before.has_value());
    REQUIRE(before.value()->GetIds()[0] == 0);
    auto serialized = index->Serialize();
    REQUIRE(serialized.has_value());
    auto restored = Factory::CreateIndex("hgraph", config).value();
    REQUIRE(restored->Deserialize(serialized.value()).has_value());
    auto after = restored->KnnSearch(query, 10, search);
    REQUIRE(after.has_value());
    for (int i = 0; i < 10; ++i) {
        REQUIRE(before.value()->GetIds()[i] == after.value()->GetIds()[i]);
        REQUIRE(before.value()->GetDistances()[i] == after.value()->GetDistances()[i]);
    }
    for (auto metric : {"ip", "cosine"}) {
        auto unsupported = config;
        unsupported.replace(unsupported.find("l2"), 2, metric);
        REQUIRE_FALSE(Factory::CreateIndex("hgraph", unsupported).has_value());
    }
    auto param = QuantizerParameter::CreateDefault("sq8_per_vector");
    REQUIRE(QuantizerParameter::IsValidQuantizationType(param->GetTypeName()));
    REQUIRE(QuantizerParameter::GetQuantizerParameterByJson(param->ToJson())->GetTypeName() ==
            "sq8_per_vector");
}
