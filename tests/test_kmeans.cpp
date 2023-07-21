//
// Created by inabao on 2023/7/21.
//
#include <catch2/catch_test_macros.hpp>
#include "vsag/vsag.h"

float fvec_norm_L2sqr(const float* x, size_t d) {
    // the double in the _ref is suspected to be a typo. Some of the manual
    // implementations this replaces used float.
    float res = 0;
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * x[i];
    }

    return res;
}
void fvec_renorm_L2(size_t d, size_t nx, float* __restrict x) {
    for (int64_t i = 0; i < nx; i++) {
        float* __restrict xi = x + i * d;

        float nr = fvec_norm_L2sqr(xi, d);

        if (nr > 0) {
            size_t j;
            const float inv_nr = 1.0 / sqrtf(nr);
            for (j = 0; j < d; j++)
                xi[j] *= inv_nr;
        }
    }
}


TEST_CASE("InnerProduct", "[Kmeans]") {
    int dim = 16;
    int max_elements = 100000;
    int clusters = 58;


    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float * data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }
    fvec_renorm_L2(dim, max_elements, data); // keep the data with the mold equal to 1
    float * centroids = new float[dim * clusters];

    float loss = vsag::kmeans_clustering(dim, max_elements, clusters, data, centroids, "ip");

    hnswlib::InnerProductSpace l2(dim);
    hnswlib::DISTFUNC  distfunc = l2.get_dist_func();
    std::vector<std::vector<int>> data_id;
    for (int i = 0; i < clusters; ++i) {
        data_id.push_back(std::vector<int>());
    }
    for (int i = 0; i < max_elements; ++i) {
        float distance = 10000;
        int id = 0;
        for (int l = 0; l < clusters; ++l) {
            float tmp_distance = distfunc(data + i * dim, centroids + l * dim, &dim);
            if (tmp_distance < distance) {
                id = l;
                distance = tmp_distance;
            }
        }
        data_id[id].push_back(i);
    }
    float avg = 0;
    for (int i = 0; i < clusters; ++i) {
        avg += data_id[i].size();
    }
    avg /= clusters;
    float deviation = 0;

    for (int i = 0; i < clusters; ++i) {
        deviation += (avg - data_id[i].size()) * (avg - data_id[i].size());
    }
    REQUIRE(std::sqrt(deviation) < 422);
    REQUIRE(loss < 8340);
}



TEST_CASE("L2", "[Kmeans]") {
    int dim = 16;
    int max_elements = 100000;
    int clusters = 58;


    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float * data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    float * centroids = new float[dim * clusters];
    float loss = vsag::kmeans_clustering(dim, max_elements, clusters, data, centroids, "l2");

    hnswlib::L2Space l2(dim);
    hnswlib::DISTFUNC  distfunc = l2.get_dist_func();
    std::vector<std::vector<int>> data_id;
    for (int i = 0; i < clusters; ++i) {
        data_id.push_back(std::vector<int>());
    }
    for (int i = 0; i < max_elements; ++i) {
        float distance = 10000;
        int id = 0;
        for (int l = 0; l < clusters; ++l) {
            float tmp_distance = distfunc(data + i * dim, centroids + l * dim, &dim);
            if (tmp_distance < distance) {
                id = l;
                distance = tmp_distance;
            }
        }
        data_id[id].push_back(i);
    }
    float avg = 0;
    for (int i = 0; i < clusters; ++i) {
        avg += data_id[i].size();
    }
    avg /= clusters;
    float deviation = 0;

    for (int i = 0; i < clusters; ++i) {
        deviation += (avg - data_id[i].size()) * (avg - data_id[i].size());
    }
    REQUIRE(std::sqrt(deviation) < 550);
    REQUIRE(loss < 88890);
}