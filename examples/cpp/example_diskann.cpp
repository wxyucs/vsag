//
// Created by inabao on 2023/8/21.
//
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

#include "vsag/errors.h"
#include "vsag/vsag.h"

const std::string tmp_dir = "/tmp/";

void
float_diskann() {
    int dim = 256;             // Dimension of the elements
    int max_elements = 10000;  // Maximum number of elements, should be known beforehand
    int M = 16;                // Tightly connected with internal dimensionality of the data
    // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int ef_runtime = 200;
    float p_val =
        0.5;  // p_val represents how much original data is selected during the training of pq compressed vectors.
    int chunks_num = 32;  // chunks_num represents the dimensionality of the compressed vector.
    std::string disk_layout_file = "/tmp/index.out";
    // Initing index
    // {
    // 	"dim": 256,
    // 	"dtype": "float32",
    // 	"metric_type": "l2",
    // 	"diskann": {
    // 	    "L": 200,
    // 	    "R": 16,
    // 	    "disk_pq_dims": 32,
    // 	    "p_val": 0.5
    // 	}
    // }
    nlohmann::json diskann_parameters{
        {"R", M},
        {"L", ef_construction},
        {"p_val", p_val},
        {"disk_pq_dims", chunks_num},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"},
        {"metric_type", "l2"},
        {"dim", dim},
        {"diskann", diskann_parameters},
    };
    auto diskann = vsag::Factory::CreateIndex("diskann", index_parameters.dump());

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
    dataset.SetNumElements(max_elements);
    dataset.SetIds(ids);
    dataset.SetFloat32Vectors(data);
    if (const auto num = diskann->Build(dataset); num.has_value()) {
        std::cout << "After Build(), Index constains: " << diskann->GetNumElements() << std::endl;
    } else if (num.error() == vsag::index_error::internal_error) {
        std::cerr << "Failed to build index: internal error" << std::endl;
        exit(-1);
    }

    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.SetNumElements(1);
        query.SetDim(dim);
        query.SetFloat32Vectors(data + i * dim);
        query.SetOwner(false);
        // {
        //  "diskann": {
        //    "ef_search": 200,
        //    "beam_search": 4,
        //    "io_limit": 200
        //  }
        // }
        nlohmann::json parameters{
            {"diskann", {{"ef_search", ef_runtime}, {"beam_search", 4}, {"io_limit", 200}}}};
        int64_t k = 2;
        if (auto result = diskann->KnnSearch(query, k, parameters.dump()); result.has_value()) {
            if (result->GetNumElements() == 1) {
                if (result->GetIds()[0] == i) {
                    correct++;
                }
            }
        } else if (result.error() == vsag::index_error::internal_error) {
            std::cerr << "failed to perform knn search on index" << std::endl;
        }
    }
    float recall = correct / max_elements;
    std::cout << "Stard Recall: " << recall << std::endl;
    // Serialize
    {
        if (auto bs = diskann->Serialize(); bs.has_value()) {
            diskann = nullptr;
            vsag::Binary pq_b = bs->Get(vsag::DISKANN_PQ);
            std::ofstream pq(tmp_dir + "diskann_pq.index", std::ios::binary);
            pq.write((const char*)pq_b.data.get(), pq_b.size);
            pq.close();

            vsag::Binary compressed_vector_b = bs->Get(vsag::DISKANN_COMPRESSED_VECTOR);
            std::ofstream compressed(tmp_dir + "diskann_compressed_vector.index", std::ios::binary);
            compressed.write((const char*)compressed_vector_b.data.get(), compressed_vector_b.size);
            compressed.close();

            vsag::Binary layout_file_b = bs->Get(vsag::DISKANN_LAYOUT_FILE);
            std::ofstream layout(tmp_dir + disk_layout_file, std::ios::binary);
            layout.write((const char*)layout_file_b.data.get(), layout_file_b.size);
            layout.close();
        } else if (bs.error() == vsag::index_error::no_enough_memory) {
            std::cerr << "no enough memory to serialize index" << std::endl;
        }
    }

    //     Deserialize
    {
        vsag::ReaderSet rs;
        auto pq_reader = vsag::Factory::CreateLocalFileReader(tmp_dir + "diskann_pq.index");
        auto compressed_vector_reader =
            vsag::Factory::CreateLocalFileReader(tmp_dir + "diskann_compressed_vector.index");
        auto disk_layout_reader = vsag::Factory::CreateLocalFileReader(tmp_dir + disk_layout_file);
        rs.Set(vsag::DISKANN_PQ, pq_reader);
        rs.Set(vsag::DISKANN_COMPRESSED_VECTOR, compressed_vector_reader);
        rs.Set(vsag::DISKANN_LAYOUT_FILE, disk_layout_reader);

        diskann = nullptr;
        diskann = vsag::Factory::CreateIndex("diskann", index_parameters.dump());

        std::cout << "#####" << std::endl;
        diskann->Deserialize(rs);
    }

    // Query the elements for themselves and measure recall 1@2
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.SetNumElements(1);
        query.SetDim(dim);
        query.SetFloat32Vectors(data + i * dim);
        query.SetOwner(false);
        nlohmann::json parameters{
            {"diskann", {{"ef_search", ef_runtime}, {"beam_search", 4}, {"io_limit", 200}}}};
        int64_t k = 2;
        if (auto result = diskann->KnnSearch(query, k, parameters.dump()); result.has_value()) {
            if (result->GetNumElements() == 1) {
                if (result->GetIds()[0] == i) {
                    correct++;
                }
            }
        } else if (result.error() == vsag::index_error::internal_error) {
            std::cerr << "failed to perform knn search on index" << std::endl;
        }
    }
    recall = correct / max_elements;
    std::cout << "RS Recall: " << recall << std::endl;

    // Deserialize
    {
        vsag::BinarySet bs;

        std::ifstream pq(tmp_dir + "diskann_pq.index", std::ios::binary);
        pq.seekg(0, std::ios::end);
        size_t size = pq.tellg();
        pq.seekg(0, std::ios::beg);
        std::shared_ptr<int8_t[]> buff(new int8_t[size]);
        pq.read(reinterpret_cast<char*>(buff.get()), size);
        vsag::Binary pq_b{
            .data = buff,
            .size = size,
        };
        bs.Set(vsag::DISKANN_PQ, pq_b);

        std::ifstream compressed(tmp_dir + "diskann_compressed_vector.index", std::ios::binary);
        compressed.seekg(0, std::ios::end);
        size = compressed.tellg();
        compressed.seekg(0, std::ios::beg);
        buff.reset(new int8_t[size]);
        compressed.read(reinterpret_cast<char*>(buff.get()), size);
        vsag::Binary compressed_vector_b{
            .data = buff,
            .size = size,
        };
        bs.Set(vsag::DISKANN_COMPRESSED_VECTOR, compressed_vector_b);

        std::ifstream disk_layout(tmp_dir + disk_layout_file, std::ios::binary);
        disk_layout.seekg(0, std::ios::end);
        size = disk_layout.tellg();
        disk_layout.seekg(0, std::ios::beg);
        buff.reset(new int8_t[size]);
        disk_layout.read(reinterpret_cast<char*>(buff.get()), size);
        vsag::Binary disk_layout_b{
            .data = buff,
            .size = size,
        };
        bs.Set(vsag::DISKANN_LAYOUT_FILE, disk_layout_b);

        diskann = nullptr;
        diskann = vsag::Factory::CreateIndex("diskann", index_parameters.dump());

        std::cout << "#####" << std::endl;
        diskann->Deserialize(bs);
    }
    // Query the elements for themselves and measure recall 1@2
    correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.SetNumElements(1);
        query.SetDim(dim);
        query.SetFloat32Vectors(data + i * dim);
        query.SetOwner(false);
        nlohmann::json parameters{
            {"diskann", {{"ef_search", ef_runtime}, {"beam_search", 4}, {"io_limit", 200}}}};
        int64_t k = 2;
        if (auto result = diskann->KnnSearch(query, k, parameters.dump()); result.has_value()) {
            if (result->GetNumElements() == 1) {
                if (result->GetIds()[0] == i) {
                    correct++;
                }
            }
        } else if (result.error() == vsag::index_error::internal_error) {
            std::cerr << "failed to perform knn search on index" << std::endl;
        }
    }
    recall = correct / max_elements;
    std::cout << "BS Recall: " << recall << std::endl;
}

int
main() {
    float_diskann();
    //
    //    uint32_t a, b;
    //    uint64_t c, d;
    //    std::ifstream in("/tmp/index.out");
    //    in.read((char *)&a, sizeof(uint32_t));
    //    in.read((char *)&b, sizeof(uint32_t));
    //    in.read((char *)&c, sizeof(uint64_t));
    //    in.read((char *)&d, sizeof(uint64_t));

    return 0;
}
