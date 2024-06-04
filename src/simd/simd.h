#pragma once

#include <stdlib.h>

namespace vsag {

void
setup_simd();

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

float
InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr);
float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr);

void
PQDistanceFloat256(const void* single_dim_centers, float single_dim_val, void* result);

#if defined(ENABLE_SSE)
float
L2SqrSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
L2SqrSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
L2SqrSIMD4ExtResidualsSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
L2SqrSIMD16ExtResidualsSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

float
InnerProductSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductDistanceSIMD16ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductDistanceSIMD4ExtSSE(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductDistanceSIMD4ExtResidualsSSE(const void* pVect1v,
                                         const void* pVect2v,
                                         const void* qty_ptr);
float
InnerProductDistanceSIMD16ExtResidualsSSE(const void* pVect1v,
                                          const void* pVect2v,
                                          const void* qty_ptr);
void
PQDistanceSSEFloat256(const void* single_dim_centers, float single_dim_val, void* result);
#endif

#if defined(ENABLE_AVX)
float
L2SqrSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductSIMD4ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
void
PQDistanceAVXFloat256(const void* single_dim_centers, float single_dim_val, void* result);
#endif

#if defined(ENABLE_AVX512)
float
L2SqrSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
#endif

typedef float (*DistanceFunc)(const void* pVect1, const void* pVect2, const void* qty_ptr);
DistanceFunc
GetL2DistanceFunc(size_t dim);
DistanceFunc
GetInnerProductDistanceFunc(size_t dim);

typedef void (*PQDistanceFunc)(const void* single_dim_centers, float single_dim_val, void* result);

PQDistanceFunc
GetPQDistanceFunc();

}  // namespace vsag
