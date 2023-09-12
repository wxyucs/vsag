#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>

#include "simpleflat.h"

int
main() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 1000;    // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int ef_runtime = 200;

    // Initing index
    std::shared_ptr<vsag::Index> simpleflat = std::make_shared<vsag::SimpleFlat>("l2");

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
    simpleflat->Build(dataset);

    // Adding data after index built
    vsag::Dataset incremental;
    incremental.SetDim(dim);
    incremental.SetNumElements(1);
    incremental.SetIds(ids + max_elements - 1);
    incremental.SetFloat32Vectors(data + (max_elements - 1) * dim);
    incremental.SetOwner(false);
    simpleflat->Add(incremental);

    // Query the elements for themselves and measure recall
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.SetNumElements(1);
        query.SetDim(dim);
        query.SetFloat32Vectors(data + i * dim);
        query.SetOwner(false);
        nlohmann::json parameters{};
        auto result = simpleflat->KnnSearch(query, 1, parameters.dump());
        if (result.GetIds()[0] == i) {
            correct++;
        }
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << std::endl;

    // not support yet
    /*
    // Serialize
    {
        vsag::BinarySet bs = simpleflat->Serialize();
        simpleflat = nullptr;
        vsag::Binary b = bs.Get(vsag::SIMPLEFLAT_DATA);
        std::ofstream file("simpleflat.index", std::ios::binary);
        file.write((const char*)b.data.get(), b.size);
        file.close();
    }

    // Deserialize
    {
        std::ifstream file("simpleflat.index", std::ios::binary);
        file.seekg(0, std::ios::end);
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::shared_ptr<int8_t[]> buff(new int8_t[size]);
        file.read(reinterpret_cast<char*>(buff.get()), size);
        vsag::Binary b{
            .data = buff,
            .size = size,
        };
        vsag::BinarySet bs;
        bs.Set(vsag::SIMPLEFLAT_DATA, b);
        simpleflat = vsag::Factory::create("simpleflat", index_parameters.dump());
        simpleflat->Deserialize(bs);
    }

    // Query the elements for themselves and measure recall
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.SetNumElements(1);
        query.SetDim(dim);
        query.SetFloat32Vectors(data + i * dim);
        query.SetOwner(false);
        nlohmann::json parameters{};
        auto result = simpleflat->KnnSearch(query, 1, parameters.dump());
        if (result.GetIds()[0] == i) {
            correct++;
        }
    }
    recall = correct / max_elements;
    std::cout << "Recall: " << recall << std::endl;
    */

    return 0;
}
