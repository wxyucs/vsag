#include <assert.h>

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

float
range_search_recall(const float* base,
                    int64_t base_num,
                    const float* query,
                    int64_t data_dim,
                    const int64_t* result_ids,
                    int64_t result_size,
                    float threshold) {
    BitsetPtr groundtruth = l2_and_filtering(data_dim, base_num, base, query, threshold);
    for (int64_t i = 0; i < result_size; ++i) {
        assert(groundtruth->Get(result_ids[i]));
    }
    if (groundtruth->CountOnes() == 0) {
        return 1;
    }
    return (float)(result_size) / groundtruth->CountOnes();
}

float
knn_search_recall(const float* base,
                  int64_t base_num,
                  const float* query,
                  int64_t data_dim,
                  const int64_t* result_ids,
                  int64_t result_size) {
    float recall = 0;
    int64_t nearest_id = 0;
    float nearest_dis = std::numeric_limits<float>::max();
    for (int64_t i = 0; i < base_num; ++i) {
        float dis = l2sqr(base + i * data_dim, query, data_dim);
        if (nearest_dis > dis) {
            nearest_id = i;
            nearest_dis = dis;
        }
    }
    for (int64_t i = 0; i < result_size; ++i) {
        if (result_ids[i] == nearest_id) {
            return 1;
        }
    }
    return 0;
}
}  // namespace vsag
