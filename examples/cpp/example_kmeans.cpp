#include <iostream>
#include <random>

#include <hnswlib/hnswlib.h>

#include <vsag/vsag.h>

float
fvec_norm_L2sqr(const float* x, size_t d) {
    // the double in the _ref is suspected to be a typo. Some of the manual
    // implementations this replaces used float.
    float res = 0;
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * x[i];
    }

    return res;
}
void
fvec_renorm_L2(size_t d, size_t nx, float* __restrict x) {
    for (int64_t i = 0; i < nx; i++) {
        float* __restrict xi = x + i * d;

        float nr = fvec_norm_L2sqr(xi, d);

        if (nr > 0) {
            size_t j;
            const float inv_nr = 1.0 / sqrtf(nr);
            for (j = 0; j < d; j++) xi[j] *= inv_nr;
        }
    }
}

int
ip_kmeans() {
    std::cout << "begin ip kmeans" << std::endl;

    int dim = 16;
    int max_elements = 1000;
    int clusters = 5;
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }
    fvec_renorm_L2(dim, max_elements, data);  // keep the data with the mold equal to 1
    float* centroids = new float[dim * clusters];
    std::cout << "loss:"
              << vsag::kmeans_clustering(dim, max_elements, clusters, data, centroids, "ip")
              << std::endl;

    hnswlib::InnerProductSpace l2(dim);
    hnswlib::DISTFUNC distfunc = l2.get_dist_func();
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

    for (int i = 0; i < clusters; ++i) {
        std::cout << "ip cluster " << i << ":" << data_id[i].size() << std::endl;
    }

    std::cout << "end ip kmeans" << std::endl;

    return 0;
}

int
l2_kmeans() {
    std::cout << "begin l2 kmeans" << std::endl;

    int dim = 16;
    int max_elements = 1000;
    int clusters = 5;

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng);
    }

    float* centroids = new float[dim * clusters];
    std::cout << "loss:"
              << vsag::kmeans_clustering(dim, max_elements, clusters, data, centroids, "l2")
              << std::endl;

    hnswlib::L2Space l2(dim);
    hnswlib::DISTFUNC distfunc = l2.get_dist_func();
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

    for (int i = 0; i < clusters; ++i) {
        std::cout << "l2 cluster " << i << ":" << data_id[i].size() << std::endl;
    }

    std::cout << "end l2 kmeans" << std::endl;
    return 0;
}

int
main() {
    ip_kmeans();
    std::cout << "--------------" << std::endl;
    l2_kmeans();
}
