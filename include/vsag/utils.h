#pragma once

#include <cstddef>
#include <string>

namespace vsag {

float
kmeans_clustering(
    size_t d, size_t n, size_t k, const float* x, float* centroids, const std::string& dis_type);

}  // namespace vsag
