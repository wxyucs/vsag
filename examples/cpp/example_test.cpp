#include "vsag/vsag.h"

int
main() {
    int dim = 256;               // Dimension of the elements
    int max_elements = 10000;       // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    hnswlib::InnerProductSpaceInt8 space(dim);
    hnswlib::HierarchicalNSW* hnsw = new hnswlib::HierarchicalNSW(&space, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    int8_t * data = new int8_t[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng) * 256;
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        hnsw->addPoint(data + i * dim, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            hnsw->searchKnn(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i)
            correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";
    return 0;
}
