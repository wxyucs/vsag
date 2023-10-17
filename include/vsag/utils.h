#pragma once

#include <cstddef>
#include <random>
#include <string>
#include <vector>

namespace vsag {

float
kmeans_clustering(
    size_t d, size_t n, size_t k, const float* x, float* centroids, const std::string& dis_type);

std::pair<int64_t, std::vector<unsigned char>>
l2_and_filtering(int64_t dim, int64_t nb, const float* base, const float* query, float threshold);

}  // namespace vsag
