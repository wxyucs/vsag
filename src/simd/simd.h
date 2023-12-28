#pragma once

namespace vsag {

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

float
InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr);
float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr);

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
#endif

#if defined(ENABLE_AVX)
float
L2SqrSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductDistanceSIMD16ExtAVX(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
#endif

#if defined(ENABLE_AVX512)
float
L2SqrSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
float
InnerProductDistanceSIMD16ExtAVX512(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
#endif

}  // namespace vsag
