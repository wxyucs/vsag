#pragma once
#include "hnswlib.h"

namespace hnswlib {

static float
InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float *) pVect1)[i] * ((float *) pVect2)[i];
    }
    return res;
}

static float
InnerProductDistance(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
}

static float
INT8_InnerProduct_impl(const void *pVect1, const void *pVect2, size_t qty) {
    int8_t *vec1 = (int8_t *)pVect1;
    int8_t *vec2 = (int8_t *)pVect2;
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        res += vec1[i] * vec2[i];
    }
    return res;
}
static
float INT8_InnerProduct(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float score = INT8_InnerProduct_impl(pVect1, pVect2, qty);
    score = score / 5242.88 + 50;
    return 1 - score / 100;
}

#if defined(USE_AVX)

// Favor using AVX if available.
static float
InnerProductSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float *pEnd1 = pVect1 + 16 * qty16;
    const float *pEnd2 = pVect1 + 4 * qty4;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    __m128 v1, v2;
    __m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    return sum;
}

static float
InnerProductDistanceSIMD4ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE)

static float
InnerProductSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;
    size_t qty4 = qty / 4;

    const float *pEnd1 = pVect1 + 16 * qty16;
    const float *pEnd2 = pVect1 + 4 * qty4;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    while (pVect1 < pEnd2) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }

    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

static float
InnerProductDistanceSIMD4ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD4ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif


#if defined(USE_AVX512)

static float
InnerProductSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN64 TmpRes[16];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;


    const float *pEnd1 = pVect1 + 16 * qty16;

    __m512 sum512 = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m512 v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        __m512 v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
    }

    _mm512_store_ps(TmpRes, sum512);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] + TmpRes[15];

    return sum;
}

static float
InnerProductDistanceSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtAVX512(pVect1v, pVect2v, qty_ptr);
}

static float INT8_InnerProduct512_AVX512_impl(const void *pVect1v, const void *pVect2v, size_t qty) {
    __mmask32 mask = 0xFFFFFFFF;
    __mmask64 mask64 = 0xFFFFFFFF;

    int32_t cTmp[16];

    int8_t *pVect1 = (int8_t *)pVect1v;
    int8_t *pVect2 = (int8_t *)pVect2v;
    const int8_t *pEnd1 = pVect1 + qty;

    __m512i sum512 = _mm512_set1_epi32(0);

    while (pVect1 < pEnd1) {
        __m256i v1 = _mm256_maskz_loadu_epi8(mask, pVect1);
        __m512i v1_512 = _mm512_cvtepi8_epi16(v1);
        pVect1 += 32;
        __m256i v2 = _mm256_maskz_loadu_epi8(mask, pVect2);
        __m512i v2_512 = _mm512_cvtepi8_epi16(v2);
        pVect2 += 32;

        sum512 = _mm512_add_epi32(sum512, _mm512_madd_epi16(v1_512, v2_512));
    }

    _mm512_mask_storeu_epi32(cTmp, mask64, sum512);
    float res = 0;
    for (int i = 0; i < 16; i ++) {
        res += cTmp[i];
    }
    return res;
}

static float INT8_InnerProduct256_AVX512_impl(const void *pVect1v, const void *pVect2v, size_t qty) {
    __mmask32 mask = 0xFFFF;
    __mmask64 mask64 = 0xFFFFFFFF;

    int32_t cTmp[16];

    int8_t *pVect1 = (int8_t *)pVect1v;
    int8_t *pVect2 = (int8_t *)pVect2v;
    const int8_t *pEnd1 = pVect1 + qty;

    __m512i sum512 = _mm512_set1_epi32(0);

    while (pVect1 < pEnd1) {
        __m128i v1 = _mm_maskz_loadu_epi8(mask, pVect1);
        __m512i v1_512 = _mm512_cvtepi8_epi32(v1);
        pVect1 += 16;
        __m128i v2 = _mm_maskz_loadu_epi8(mask, pVect2);
        __m512i v2_512 = _mm512_cvtepi8_epi32(v2);
        pVect2 += 16;

        sum512 = _mm512_add_epi32(sum512, _mm512_mullo_epi32(v1_512, v2_512));
    }

    _mm512_mask_storeu_epi32(cTmp, mask64, sum512);
    float res = 0;
    for (int i = 0; i < 16; i ++) {
        res += cTmp[i];
    }
    return res;
}



static float INT8_InnerProduct256Residuals_AVX512_impl(const void *pVect1v, const void *pVect2v,
                                          size_t qty) {
    size_t qty2 = qty >> 4 << 4;
    float res = INT8_InnerProduct256_AVX512_impl(pVect1v, pVect2v, qty2);
    int8_t *pVect1 = (int8_t *)pVect1v + qty2;
    int8_t *pVect2 = (int8_t *)pVect2v + qty2;

    size_t qty_left = qty - qty2;
    if (qty_left != 0) {
        res += INT8_InnerProduct_impl(pVect1, pVect2, qty_left);
    }
    return res;
}


static float INT8_InnerProduct512Residuals_AVX512_impl(const void *pVect1v, const void *pVect2v,
                                          size_t qty) {
    size_t qty2 = qty >> 5 << 5;
    float res = INT8_InnerProduct512_AVX512_impl(pVect1v, pVect2v, qty2);
    int8_t *pVect1 = (int8_t *)pVect1v + qty2;
    int8_t *pVect2 = (int8_t *)pVect2v + qty2;

    size_t qty_left = qty - qty2;
    if (qty_left != 0) {
        res += INT8_InnerProduct256Residuals_AVX512_impl(pVect1, pVect2, qty_left);
    }
    return res;
}

static float INT8_InnerProduct256Residuals_AVX512(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float score = INT8_InnerProduct256Residuals_AVX512_impl(pVect1, pVect2, qty);
    score = score / 5242.88 + 50;
    return 1 - score / 100;
}

static float INT8_InnerProduct512Residuals_AVX512(const void *pVect1, const void *pVect2, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    float score = INT8_InnerProduct512Residuals_AVX512_impl(pVect1, pVect2, qty);
    score = score / 5242.88 + 50;
    return 1 - score / 100;
}


#endif

#if defined(USE_AVX)

static float
InnerProductSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;


    const float *pEnd1 = pVect1 + 16 * qty16;

    __m256 sum256 = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);

        __m256 v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        __m256 v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
    }

    _mm256_store_ps(TmpRes, sum256);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

    return sum;
}

static float
InnerProductDistanceSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtAVX(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE)

static float
InnerProductSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    float PORTABLE_ALIGN32 TmpRes[8];
    float *pVect1 = (float *) pVect1v;
    float *pVect2 = (float *) pVect2v;
    size_t qty = *((size_t *) qty_ptr);

    size_t qty16 = qty / 16;

    const float *pEnd1 = pVect1 + 16 * qty16;

    __m128 v1, v2;
    __m128 sum_prod = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
    }
    _mm_store_ps(TmpRes, sum_prod);
    float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

    return sum;
}

static float
InnerProductDistanceSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    return 1.0f - InnerProductSIMD16ExtSSE(pVect1v, pVect2v, qty_ptr);
}

#endif

#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
static DISTFUNC InnerProductSIMD16Ext = InnerProductSIMD16ExtSSE;
static DISTFUNC InnerProductSIMD4Ext = InnerProductSIMD4ExtSSE;
static DISTFUNC InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtSSE;
static DISTFUNC InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtSSE;

static float
InnerProductDistanceSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty16 = qty >> 4 << 4;
    float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
    float *pVect1 = (float *) pVect1v + qty16;
    float *pVect2 = (float *) pVect2v + qty16;

    size_t qty_left = qty - qty16;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
    return 1.0f - (res + res_tail);
}

static float
InnerProductDistanceSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    size_t qty = *((size_t *) qty_ptr);
    size_t qty4 = qty >> 2 << 2;

    float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
    size_t qty_left = qty - qty4;

    float *pVect1 = (float *) pVect1v + qty4;
    float *pVect2 = (float *) pVect2v + qty4;
    float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

    return 1.0f - (res + res_tail);
}
#endif

class InnerProductSpace : public SpaceInterface {
    DISTFUNC fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    InnerProductSpace(size_t dim) {
        fstdistfunc_ = InnerProductDistance;
#if defined(USE_AVX512)
        if (dim % 16 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = InnerProductDistanceSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = InnerProductDistanceSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = InnerProductDistanceSIMD4ExtResiduals;
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() {
        return data_size_;
    }

    DISTFUNC get_dist_func() {
        return fstdistfunc_;
    }

    void *get_dist_func_param() {
        return &dim_;
    }

~InnerProductSpace() {}
};


class InnerProductSpaceInt8 : public SpaceInterface {
DISTFUNC fstdistfunc_;
size_t data_size_;
size_t dim_;

public:
    InnerProductSpaceInt8(size_t dim) {
        fstdistfunc_ = INT8_InnerProduct;
#if defined(USE_AVX512)

        if (dim >= 32) {
            fstdistfunc_ = INT8_InnerProduct512Residuals_AVX512;
        } else if (dim >= 16) {
            fstdistfunc_ = INT8_InnerProduct256Residuals_AVX512;
        }
#endif
        dim_ = dim;
        data_size_ = dim * sizeof(int8_t);
}

size_t get_data_size() {
        return data_size_;
}

DISTFUNC get_dist_func() {
        return fstdistfunc_;
}

void *get_dist_func_param() {
        return &dim_;
}

~InnerProductSpaceInt8() {}
};




}  // namespace hnswlib
