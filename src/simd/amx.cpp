
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

#include "simd/int8_simd.h"
#if defined(ENABLE_AMX)
#include <immintrin.h>
#endif

#include <cmath>

#include "simd.h"

namespace vsag::amx {

#if defined(ENABLE_AMX)
// Tile configuration structure for AMX
struct __tile_config {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved_0[14];
    uint16_t colsb[16];  // Column size in bytes for each tile
    uint8_t rows[16];    // Number of rows for each tile
};

// Global tile configuration for INT8 operations (16x64 tiles)
static __tile_config g_tile_cfg_int8 = {0};
static bool g_amx_initialized = false;

// Initialize AMX tile configuration for INT8
static inline void
InitAMXTileConfigInt8() {
    if (g_amx_initialized) {
        return;
    }

    g_tile_cfg_int8.palette_id = 1;
    g_tile_cfg_int8.start_row = 0;

    // Configure tiles for INT8 matrix multiplication
    // Tile 0: A matrix (16 rows x 64 bytes = 16x64 int8)
    g_tile_cfg_int8.rows[0] = 16;
    g_tile_cfg_int8.colsb[0] = 64;

    // Tile 1: B matrix (16 rows x 64 bytes = 16x64 int8)
    g_tile_cfg_int8.rows[1] = 16;
    g_tile_cfg_int8.colsb[1] = 64;

    // Tile 2-5: Accumulator tiles (16 rows x 16 dwords = 16x16 int32)
    for (int i = 2; i < 6; ++i) {
        g_tile_cfg_int8.rows[i] = 16;
        g_tile_cfg_int8.colsb[i] = 64;  // 16 int32s = 64 bytes
    }

    _tile_loadconfig(&g_tile_cfg_int8);
    g_amx_initialized = true;
}

// Release AMX tiles
static inline void
ReleaseAMX() {
    if (g_amx_initialized) {
        _tile_release();
        g_amx_initialized = false;
    }
}
#endif  // ENABLE_AMX

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    // AMX is not optimal for single vector L2 distance, use AVX512
    return avx512::L2Sqr(pVect1v, pVect2v, qty_ptr);
}

float
InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    // AMX is not optimal for single vector inner product, use AVX512
    return avx512::InnerProduct(pVect1v, pVect2v, qty_ptr);
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - amx::InnerProduct(pVect1, pVect2, qty_ptr);
}

float
INT8L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    // AMX is not optimal for single vector L2, use AVX512
    return avx512::INT8L2Sqr(pVect1v, pVect2v, qty_ptr);
}

float
INT8InnerProduct(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    // AMX is not optimal for single vector IP, use AVX512
    return avx512::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
}

float
INT8InnerProductDistance(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    return -amx::INT8InnerProduct(pVect1v, pVect2v, qty_ptr);
}

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result) {
    // Use AVX512 implementation
    avx512::PQDistanceFloat256(single_dim_centers, single_dim_val, result);
}

void
Prefetch(const void* data) {
    avx512::Prefetch(data);
}

float
FP32ComputeIP(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
    // AMX is designed for matrix multiplication, not single vector dot product
    // Fall back to AVX512 for this operation
    return avx512::FP32ComputeIP(query, codes, dim);
}

float
FP32ComputeL2Sqr(const float* RESTRICT query, const float* RESTRICT codes, uint64_t dim) {
    // AMX is not optimal for L2 distance computation
    // Fall back to AVX512
    return avx512::FP32ComputeL2Sqr(query, codes, dim);
}

void
FP32ComputeIPBatch4(const float* RESTRICT query,
                    uint64_t dim,
                    const float* RESTRICT codes1,
                    const float* RESTRICT codes2,
                    const float* RESTRICT codes3,
                    const float* RESTRICT codes4,
                    float& result1,
                    float& result2,
                    float& result3,
                    float& result4) {
    // For batch 4, AVX512 is more efficient than AMX setup overhead
    avx512::FP32ComputeIPBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
}

void
FP32ComputeL2SqrBatch4(const float* RESTRICT query,
                       uint64_t dim,
                       const float* RESTRICT codes1,
                       const float* RESTRICT codes2,
                       const float* RESTRICT codes3,
                       const float* RESTRICT codes4,
                       float& result1,
                       float& result2,
                       float& result3,
                       float& result4) {
    avx512::FP32ComputeL2SqrBatch4(
        query, dim, codes1, codes2, codes3, codes4, result1, result2, result3, result4);
}

void
FP32Sub(const float* x, const float* y, float* z, uint64_t dim) {
    avx512::FP32Sub(x, y, z, dim);
}

void
FP32Add(const float* x, const float* y, float* z, uint64_t dim) {
    avx512::FP32Add(x, y, z, dim);
}

void
FP32Mul(const float* x, const float* y, float* z, uint64_t dim) {
    avx512::FP32Mul(x, y, z, dim);
}

void
FP32Div(const float* x, const float* y, float* z, uint64_t dim) {
    avx512::FP32Div(x, y, z, dim);
}

float
FP32ReduceAdd(const float* x, uint64_t dim) {
    return avx512::FP32ReduceAdd(x, dim);
}

float
INT8ComputeIP(const int8_t* __restrict query, const int8_t* __restrict codes, uint64_t dim) {
#if defined(ENABLE_AMX)
    // For larger dimensions, AMX can be beneficial for INT8 dot products
    // AMX works best with matrix-style operations
    if (dim < 256) {
        return avx512::INT8ComputeIP(query, codes, dim);
    }

    InitAMXTileConfigInt8();

    // Process in tiles of 16x64
    constexpr int TILE_ROWS = 16;
    constexpr int TILE_COLS = 64;

    int32_t sum = 0;
    uint64_t i = 0;

    // Process full tiles
    for (; i + TILE_COLS <= dim; i += TILE_COLS) {
        // Load query tile
        _tile_loadd(0, query + i, TILE_COLS);

        // Load codes tile
        _tile_loadd(1, codes + i, TILE_COLS);

        // Zero accumulator
        _tile_zero(2);

        // Compute dot product: tile2 += tile0 * tile1 (INT8 multiplication with accumulation)
        _tile_dpbssd(2, 0, 1);

        // Store result and accumulate
        alignas(64) int32_t temp[TILE_ROWS * 16];
        _tile_stored(2, temp, 64);

        // Sum up the results
        for (int j = 0; j < TILE_ROWS * 16; ++j) {
            sum += temp[j];
        }
    }

    // Process remaining elements with AVX512
    if (dim > i) {
        sum += static_cast<int32_t>(avx512::INT8ComputeIP(query + i, codes + i, dim - i));
    }

    return static_cast<float>(sum);
#else
    return avx512::INT8ComputeIP(query, codes, dim);
#endif
}

float
INT8ComputeL2Sqr(const int8_t* RESTRICT query, const int8_t* RESTRICT codes, uint64_t dim) {
    // AMX doesn't directly support L2 distance computation
    // Use AVX512 implementation
    return avx512::INT8ComputeL2Sqr(query, codes, dim);
}

float
BF16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    // AMX-BF16 could be used for larger matrix operations
    // For single vector dot product, AVX512 is more efficient
    return avx512::BF16ComputeIP(query, codes, dim);
}

float
BF16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    return avx512::BF16ComputeL2Sqr(query, codes, dim);
}

float
FP16ComputeIP(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    return avx512::FP16ComputeIP(query, codes, dim);
}

float
FP16ComputeL2Sqr(const uint8_t* RESTRICT query, const uint8_t* RESTRICT codes, uint64_t dim) {
    return avx512::FP16ComputeL2Sqr(query, codes, dim);
}

float
SQ8ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
    return avx512::SQ8ComputeIP(query, codes, lower_bound, diff, dim);
}

float
SQ8ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
    return avx512::SQ8ComputeL2Sqr(query, codes, lower_bound, diff, dim);
}

float
SQ8ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
    return avx512::SQ8ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
}

float
SQ8ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
    return avx512::SQ8ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
}

float
SQ4ComputeIP(const float* RESTRICT query,
             const uint8_t* RESTRICT codes,
             const float* RESTRICT lower_bound,
             const float* RESTRICT diff,
             uint64_t dim) {
    return avx512::SQ4ComputeIP(query, codes, lower_bound, diff, dim);
}

float
SQ4ComputeL2Sqr(const float* RESTRICT query,
                const uint8_t* RESTRICT codes,
                const float* RESTRICT lower_bound,
                const float* RESTRICT diff,
                uint64_t dim) {
    return avx512::SQ4ComputeL2Sqr(query, codes, lower_bound, diff, dim);
}

float
SQ4ComputeCodesIP(const uint8_t* RESTRICT codes1,
                  const uint8_t* RESTRICT codes2,
                  const float* RESTRICT lower_bound,
                  const float* RESTRICT diff,
                  uint64_t dim) {
    return avx512::SQ4ComputeCodesIP(codes1, codes2, lower_bound, diff, dim);
}

float
SQ4ComputeCodesL2Sqr(const uint8_t* RESTRICT codes1,
                     const uint8_t* RESTRICT codes2,
                     const float* RESTRICT lower_bound,
                     const float* RESTRICT diff,
                     uint64_t dim) {
    return avx512::SQ4ComputeCodesL2Sqr(codes1, codes2, lower_bound, diff, dim);
}

float
SQ4UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
    return avx512::SQ4UniformComputeCodesIP(codes1, codes2, dim);
}

float
SQ8UniformComputeCodesIP(const uint8_t* RESTRICT codes1,
                         const uint8_t* RESTRICT codes2,
                         uint64_t dim) {
    return avx512::SQ8UniformComputeCodesIP(codes1, codes2, dim);
}

float
RaBitQFloatBinaryIP(const float* vector, const uint8_t* bits, uint64_t dim, float inv_sqrt_d) {
    return avx512::RaBitQFloatBinaryIP(vector, bits, dim, inv_sqrt_d);
}

uint32_t
RaBitQSQ4UBinaryIP(const uint8_t* codes, const uint8_t* bits, uint64_t dim) {
    return avx512::RaBitQSQ4UBinaryIP(codes, bits, dim);
}

void
DivScalar(const float* from, float* to, uint64_t dim, float scalar) {
    avx512::DivScalar(from, to, dim, scalar);
}

float
Normalize(const float* from, float* to, uint64_t dim) {
    float norm = std::sqrt(FP32ComputeIP(from, from, dim));
    amx::DivScalar(from, to, dim, norm);
    return norm;
}

void
PQFastScanLookUp32(const uint8_t* RESTRICT lookup_table,
                   const uint8_t* RESTRICT codes,
                   uint64_t pq_dim,
                   int32_t* RESTRICT result) {
    avx512::PQFastScanLookUp32(lookup_table, codes, pq_dim, result);
}

void
BitAnd(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
    avx512::BitAnd(x, y, num_byte, result);
}

void
BitOr(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
    avx512::BitOr(x, y, num_byte, result);
}

void
BitXor(const uint8_t* x, const uint8_t* y, const uint64_t num_byte, uint8_t* result) {
    avx512::BitXor(x, y, num_byte, result);
}

void
BitNot(const uint8_t* x, const uint64_t num_byte, uint8_t* result) {
    avx512::BitNot(x, num_byte, result);
}

void
KacsWalk(float* data, uint64_t len) {
    avx512::KacsWalk(data, len);
}

void
FlipSign(const uint8_t* flip, float* data, uint64_t dim) {
    avx512::FlipSign(flip, data, dim);
}

void
VecRescale(float* data, uint64_t dim, float val) {
    avx512::VecRescale(data, dim, val);
}

void
RotateOp(float* data, int idx, int dim_, int step) {
    avx512::RotateOp(data, idx, dim_, step);
}

void
FHTRotate(float* data, uint64_t dim_) {
    avx512::FHTRotate(data, dim_);
}

}  // namespace vsag::amx
