#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>

#include "vsag/factory.h"
#include "vsag/readerset.h"
#include "vsag/vsag.h"

void
float_hnsw() {
    int dim = 16;               // Dimension of the elements
    int max_elements = 1000;    // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int ef_runtime = 200;

    // Initing index
    nlohmann::json hnsw_parameters{
        {"max_elements", max_elements},
        {"M", M},
        {"ef_construction", ef_construction},
        {"ef_runtime", ef_runtime},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};
    auto hnsw = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());

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
    std::cout << "After Build(), Index constains: " << hnsw->GetNumElements() << std::endl;

    // Adding data after index built
    vsag::Dataset incremental;
    incremental.SetDim(dim);
    incremental.SetNumElements(1);
    incremental.SetIds(ids + max_elements - 1);
    incremental.SetFloat32Vectors(data + (max_elements - 1) * dim);
    incremental.SetOwner(false);
    hnsw->Add(incremental);
    std::cout << "After Add(), Index constains: " << hnsw->GetNumElements() << std::endl;

    // Query the elements for themselves and measure recall 1@1
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.SetNumElements(1);
        query.SetDim(dim);
        query.SetFloat32Vectors(data + i * dim);
        query.SetOwner(false);
        nlohmann::json parameters{
            {"hnsw", {"ef_runtime", ef_runtime}},
        };
        int64_t k = 10;
        auto result = hnsw->KnnSearch(query, k, parameters.dump());
        if (result.GetIds()[0] == i) {
            correct++;
        }
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << std::endl;

    // Serialize
    {
        vsag::BinarySet bs = hnsw->Serialize();
        hnsw = nullptr;
        vsag::Binary b = bs.Get(vsag::HNSW_DATA);
        std::ofstream file("hnsw.index", std::ios::binary);
        file.write((const char*)b.data.get(), b.size);
        file.close();
    }

    // Deserialize
    {
        std::ifstream file("hnsw.index", std::ios::binary);
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
        bs.Set(vsag::HNSW_DATA, b);
        hnsw = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
        hnsw->Deserialize(bs);
    }

    // Query the elements for themselves and measure recall 1@10
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.SetNumElements(1);
        query.SetDim(dim);
        query.SetFloat32Vectors(data + i * dim);
        query.SetOwner(false);
        nlohmann::json parameters{
            {"hnsw", {"ef_runtime", ef_runtime}},
        };
        int64_t k = 10;
        auto result = hnsw->KnnSearch(query, k, parameters.dump());
        if (result.GetNumElements() == 1) {
            if (result.GetIds()[0] == i or result.GetIds()[1] == i) {
                correct++;
            }
        }
    }
    recall = correct / max_elements;
    std::cout << "Recall: " << recall << std::endl;

    // Deserialize
    {
        auto file_reader = vsag::Factory::CreateLocalFileReader("hnsw.index");
        vsag::ReaderSet rs;
        rs.Set(vsag::HNSW_DATA, file_reader);
        hnsw = vsag::Factory::CreateIndex("hnsw", index_parameters.dump());
        hnsw->Deserialize(rs);
    }

    // Query the elements for themselves and measure recall 1@10
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.SetNumElements(1);
        query.SetDim(dim);
        query.SetFloat32Vectors(data + i * dim);
        query.SetOwner(false);
        nlohmann::json parameters{
            {"hnsw", {"ef_runtime", ef_runtime}},
        };
        int64_t k = 10;
        auto result = hnsw->KnnSearch(query, k, parameters.dump());
        if (result.GetNumElements() == 1) {
            if (result.GetIds()[0] == i or result.GetIds()[1] == i) {
                correct++;
            }
        }
    }
    recall = correct / max_elements;
    std::cout << "Recall: " << recall << std::endl;
}

int
main() {
    float_hnsw();
    return 0;
}
