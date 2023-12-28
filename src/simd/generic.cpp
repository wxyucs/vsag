
#include <iostream>

namespace vsag {

float
L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
    float* pVect1 = (float*)pVect1v;
    float* pVect2 = (float*)pVect2v;
    size_t qty = *((size_t*)qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}

float
InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    size_t qty = *((size_t*)qty_ptr);
    float res = 0;
    for (unsigned i = 0; i < qty; i++) {
        res += ((float*)pVect1)[i] * ((float*)pVect2)[i];
    }
    return res;
}

float
InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr) {
    return 1.0f - InnerProduct(pVect1, pVect2, qty_ptr);
}

}  // namespace vsag
