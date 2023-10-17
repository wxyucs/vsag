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

std::pair<int64_t, std::vector<unsigned char>>
l2_and_filtering(int64_t dim, int64_t nb, const float* base, const float* query, float threshold) {
    std::vector<unsigned char> res;
    res.resize(nb / 8 + (nb % 8 == 0 ? 0 : 1));

    int64_t count = 0;
    for (int64_t i = 0; i < nb; ++i) {
        const float dist = l2sqr(base + i * dim, query, dim);
        if (dist <= threshold) {
            ++count;
            int byte_index = i / 8;
            int bit_index = i % 8;
            res.data()[byte_index] |= (1 << bit_index);
        }
    }

    return std::make_pair(count, res);
}

}  // namespace vsag
