//
// Created by jinjiabao.jjb on 1/2/24.
//

#include "simd.h"

#include <cpuinfo.h>

#include <iostream>

namespace vsag {

float (*L2SqrSIMD16Ext)(const void*, const void*, const void*);
float (*L2SqrSIMD16ExtResiduals)(const void*, const void*, const void*);
float (*L2SqrSIMD4Ext)(const void*, const void*, const void*);
float (*L2SqrSIMD4ExtResiduals)(const void*, const void*, const void*);

float (*InnerProductSIMD4Ext)(const void*, const void*, const void*);
float (*InnerProductSIMD16Ext)(const void*, const void*, const void*);
float (*InnerProductDistanceSIMD16Ext)(const void*, const void*, const void*);
float (*InnerProductDistanceSIMD16ExtResiduals)(const void*, const void*, const void*);
float (*InnerProductDistanceSIMD4Ext)(const void*, const void*, const void*);
float (*InnerProductDistanceSIMD4ExtResiduals)(const void*, const void*, const void*);

void
setup_simd() {
    L2SqrSIMD16Ext = L2Sqr;
    L2SqrSIMD16ExtResiduals = L2Sqr;
    L2SqrSIMD4Ext = L2Sqr;
    L2SqrSIMD4ExtResiduals = L2Sqr;

    InnerProductSIMD4Ext = InnerProduct;
    InnerProductSIMD16Ext = InnerProduct;
    InnerProductDistanceSIMD16Ext = InnerProductDistance;
    InnerProductDistanceSIMD16ExtResiduals = InnerProductDistance;
    InnerProductDistanceSIMD4Ext = InnerProductDistance;
    InnerProductDistanceSIMD4ExtResiduals = InnerProductDistance;

#if defined(ENABLE_SSE)
    if (cpuinfo_has_x86_sse()) {
        L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;
        L2SqrSIMD16ExtResiduals = L2SqrSIMD16ExtResidualsSSE;
        L2SqrSIMD4Ext = L2SqrSIMD4ExtSSE;
        L2SqrSIMD4ExtResiduals = L2SqrSIMD4ExtResidualsSSE;

        InnerProductSIMD4Ext = InnerProductSIMD4ExtSSE;
        InnerProductSIMD16Ext = InnerProductSIMD16ExtSSE;
        InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtSSE;
        InnerProductDistanceSIMD16ExtResiduals = InnerProductDistanceSIMD16ExtResidualsSSE;
        InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtSSE;
        InnerProductDistanceSIMD4ExtResiduals = InnerProductDistanceSIMD4ExtResidualsSSE;
    }
#endif

#if defined(ENABLE_AVX)
    if (cpuinfo_has_x86_avx()) {
        L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;

        InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
        InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
    }
#endif

#if defined(ENABLE_AVX512)
    if (cpuinfo_has_x86_avx512f()) {
        L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;

        InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
    }
#endif
}

DistanceFunc
GetInnerProductDistanceFunc(size_t dim) {
    if (dim % 16 == 0) {
        return vsag::InnerProductDistanceSIMD16Ext;
    } else if (dim % 4 == 0) {
        return vsag::InnerProductDistanceSIMD4Ext;
    } else if (dim > 16) {
        return vsag::InnerProductDistanceSIMD16ExtResiduals;
    } else if (dim > 4) {
        return vsag::InnerProductDistanceSIMD4ExtResiduals;
    } else {
        return vsag::InnerProductDistance;
    }
}

PQDistanceFunc
GetPQDistanceFunc() {
#ifdef ENABLE_AVX
    return PQDistanceAVXFloat256;
#endif
#ifdef ENABLE_SSE
    return PQDistanceSSEFloat256;
#endif
    return PQDistanceFloat256;
}

DistanceFunc
GetL2DistanceFunc(size_t dim) {
    if (dim % 16 == 0) {
        return vsag::L2SqrSIMD16Ext;
    } else if (dim % 4 == 0) {
        return vsag::L2SqrSIMD4Ext;
    } else if (dim > 16) {
        return vsag::L2SqrSIMD16ExtResiduals;
    } else if (dim > 4) {
        return vsag::L2SqrSIMD4ExtResiduals;
    } else {
        return vsag::L2Sqr;
    }
}

}  // namespace vsag
