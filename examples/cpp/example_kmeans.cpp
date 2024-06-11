
#include <vsag/vsag.h>

#include <iostream>
#include <random>

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
    return 0;
}

int
main() {
    ip_kmeans();
    std::cout << "--------------" << std::endl;
    l2_kmeans();
}
