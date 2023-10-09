#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>

#include "vsag/factory.h"

int
main() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 10;      // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int ef_runtime = 200;

    // Initing index
    nlohmann::json index_parameters{{"metric_type", "l2"}};
    std::shared_ptr<vsag::Index> simpleflat =
        vsag::Factory::CreateIndex("simpleflat", index_parameters.dump());

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
        auto result = simpleflat->KnnSearch(query, 1, "");
        if (result.GetIds()[0] == i) {
            correct++;
        }
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << std::endl;

    // not support yet

    // Serialize
    {
        vsag::BinarySet bs = simpleflat->Serialize();
        simpleflat = nullptr;
        vsag::Binary b_ids = bs.Get(vsag::SIMPLEFLAT_IDS);
        std::ofstream file_ids("simpleflat_ids.index", std::ios::binary);
        file_ids.write((const char*)b_ids.data.get(), b_ids.size);
        file_ids.close();

        vsag::Binary b_vectors = bs.Get(vsag::SIMPLEFLAT_VECTORS);
        std::ofstream file_vectors("simpleflat_vectors.index", std::ios::binary);
        file_vectors.write((const char*)b_vectors.data.get(), b_vectors.size);
        file_vectors.close();
    }

    // Deserialize
    {
        vsag::BinarySet bs;

        std::ifstream file_ids("simpleflat_ids.index", std::ios::binary);
        file_ids.seekg(0, std::ios::end);
        size_t size_ids = file_ids.tellg();
        file_ids.seekg(0, std::ios::beg);
        std::shared_ptr<int8_t[]> buff_ids(new int8_t[size_ids]);
        file_ids.read(reinterpret_cast<char*>(buff_ids.get()), size_ids);
        vsag::Binary b_ids{
            .data = buff_ids,
            .size = size_ids,
        };
        bs.Set(vsag::SIMPLEFLAT_IDS, b_ids);

        std::ifstream file_vectors("simpleflat_vectors.index", std::ios::binary);
        file_vectors.seekg(0, std::ios::end);
        size_t size_vectors = file_vectors.tellg();
        file_vectors.seekg(0, std::ios::beg);
        std::shared_ptr<int8_t[]> buff_vectors(new int8_t[size_vectors]);
        file_vectors.read(reinterpret_cast<char*>(buff_vectors.get()), size_vectors);
        vsag::Binary b_vectors{
            .data = buff_vectors,
            .size = size_vectors,
        };
        bs.Set(vsag::SIMPLEFLAT_VECTORS, b_vectors);

        simpleflat = vsag::Factory::CreateIndex("simpleflat", index_parameters.dump());
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
        auto result = simpleflat->KnnSearch(query, 1, "");
        if (result.GetIds()[0] == i) {
            correct++;
        }
    }
    recall = correct / max_elements;
    std::cout << "Recall: " << recall << std::endl;

    // Deserialize
    {
        vsag::ReaderSet rs;

        auto ids_reader = vsag::Factory::CreateLocalFileReader("simpleflat_ids.index");
        rs.Set(vsag::SIMPLEFLAT_IDS, ids_reader);

        auto vectors_reader = vsag::Factory::CreateLocalFileReader("simpleflat_vectors.index");
        rs.Set(vsag::SIMPLEFLAT_VECTORS, vectors_reader);

        simpleflat = vsag::Factory::CreateIndex("simpleflat", index_parameters.dump());

        simpleflat->Deserialize(rs);
    }

    // Query the elements for themselves and measure recall
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.SetNumElements(1);
        query.SetDim(dim);
        query.SetFloat32Vectors(data + i * dim);
        query.SetOwner(false);
        auto result = simpleflat->KnnSearch(query, 1, "");
        if (result.GetIds()[0] == i) {
            correct++;
        }
    }
    recall = correct / max_elements;
    std::cout << "Recall: " << recall << std::endl;

    return 0;
}
