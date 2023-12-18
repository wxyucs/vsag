#pragma once

#include <cstddef>
#include <random>
#include <string>
#include <vector>

#include "bitset.h"

namespace vsag {

float
kmeans_clustering(
    size_t d, size_t n, size_t k, const float* x, float* centroids, const std::string& dis_type);

BitsetPtr
l2_and_filtering(int64_t dim, int64_t nb, const float* base, const float* query, float threshold);

float
range_search_recall(const float* base,
                    const int64_t* id_map,
                    int64_t base_num,
                    const float* query,
                    int64_t data_dim,
                    const int64_t* result_ids,
                    int64_t result_size,
                    float threshold);

float
knn_search_recall(const float* base,
                  const int64_t* id_map,
                  int64_t base_num,
                  const float* query,
                  int64_t data_dim,
                  const int64_t* result_ids,
                  int64_t result_size);

}  // namespace vsag
