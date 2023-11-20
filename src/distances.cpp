#include <cstdint>
#include <vector>

#include "vsag/utils.h"

namespace vsag {

float
l2sqr(const void* vec1, const void* vec2, int64_t dim) {
    float* v1 = (float*)vec1;
    float* v2 = (float*)vec2;

    float res = 0;
    for (int64_t i = 0; i < dim; i++) {
        float t = *v1 - *v2;
        v1++;
        v2++;
        res += t * t;
    }

    return res;
}

BitsetPtr
l2_and_filtering(int64_t dim, int64_t nb, const float* base, const float* query, float threshold) {
    BitsetPtr bp = std::make_shared<Bitset>();
    bp->Extend(nb);

    int64_t count = 0;
    for (int64_t i = 0; i < nb; ++i) {
        const float dist = l2sqr(base + i * dim, query, dim);
        if (dist <= threshold) {
            bp->Set(i, true);
        }
    }

    return bp;
}

}  // namespace vsag
