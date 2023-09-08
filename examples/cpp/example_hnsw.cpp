#include <iostream>
#include <random>

#include <nlohmann/json.hpp>

#include "vsag/vsag.h"

/*
void
int8_hnsw() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 1000;    // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int ef_runtime = 200;
    // Initing index

    vsag::HNSW hnsw(std::make_shared<hnswlib::InnerProductSpaceInt8>(dim),
                    max_elements,
                    M,
                    ef_construction,
                    ef_runtime);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    int8_t* data = new int8_t[dim * max_elements];
    for (int i = 0; i < dim * max_elements; i++) {
        data[i] = distrib_real(rng) * 256;
    }

    // Add data to index
    for (int i = 0; i < max_elements; i++) {
        hnsw.addPoint(data + i * dim, i);
    }

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
            hnsw.searchTopK(data + i * dim, 1);
        hnswlib::labeltype label = result.top().second;
        if (label == i)
            correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";
}
*/

void
float_hnsw() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 1000;    // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int ef_runtime = 200;

    // Initing index
    nlohmann::json index_parameters{
        {"dtype", "float32"},
        {"metric_type", "l2"},
        {"dim", dim},
        {"max_elements", max_elements},
        {"M", M},
        {"ef_construction", ef_construction},
        // {"ef_runtime", ef_runtime},
    };
    auto hnsw = vsag::Factory::create("hnsw", index_parameters.dump());


    int64_t* ids = new int64_t[max_elements];
    float* data = new float[dim * max_elements];

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int i = 0; i < max_elements; i++) ids[i] = i;
    for (int i = 0; i < dim * max_elements; i++) data[i] = distrib_real(rng);

    // Build index
    vsag::Dataset dataset;
    dataset.SetDim(dim);
    dataset.SetNumElements(max_elements - 1);
    dataset.SetIds(ids);
    dataset.SetFloat32Vectors(data);
    hnsw->Build(dataset);

    // Adding data after index built
    vsag::Dataset incremental;
    incremental.SetDim(dim);
    incremental.SetNumElements(1);
    incremental.SetIds(ids + max_elements - 1);
    incremental.SetFloat32Vectors(data + (max_elements - 1) * dim);
    incremental.SetOwner(false);
    hnsw->Add(incremental);

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.SetNumElements(1);
        query.SetDim(dim);
        query.SetFloat32Vectors(data + i * dim);
        query.SetOwner(false);
        nlohmann::json parameters{
            {"ef_runtime", ef_runtime},
        };
        auto result = hnsw->KnnSearch(query, 1, parameters.dump());
        if (result.GetIds()[0] == i) {
            correct++;
        }
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << std::endl;
}

int
main() {
    // int8_hnsw();
    float_hnsw();
    return 0;
}
