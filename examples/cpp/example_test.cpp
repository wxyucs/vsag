#include "vsag/vsag.h"

int
main() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 2;       // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff

    // Initing index
    vsag::HNSW hnsw(dim, max_elements, M, ef_construction);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    std::vector<std::vector<float>> datas;
    for (int i = 0; i < max_elements; i++) {
        std::vector<float> data(dim);
        for (int j = 0; j < dim; ++j) {
            data[i] = distrib_real(rng);
        }
        datas.push_back(data);
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        hnsw.addPoint(datas[i], i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            hnsw.searchTopK(datas[i], 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i)
            correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";
    return 0;
}
