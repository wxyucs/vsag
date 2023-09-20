//
// Created by inabao on 2023/8/21.
//
#include "vsag/vsag.h"
#include <sstream>
#include <nlohmann/json.hpp>
#include <iostream>
#include <fstream>
// int venama_memory() {


//     diskann::Metric metric = diskann::Metric::L2;

//     size_t data_num = 10000, data_dim = 256;
//     int L = 200;
//     int R = 64;
//     std::string data_type = "float";
//     std::string label_type = "uint32";

//     auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
//                                   .build();

//     diskann::Index<float, uint32_t, uint32_t> index(metric, data_dim, data_num, false, false, false, false, 4, false);

//     std::mt19937 rng;
//     rng.seed(47);
//     std::uniform_real_distribution<> distrib_real;
//     float * data = new float[data_dim * data_num];
//     std::vector<uint32_t> tags;
//     for (uint32_t i = 0; i < data_dim * data_num; i++) {
//         data[i] = distrib_real(rng);
//         tags.push_back(i);
//     }
//     index.build(data, data_num, index_build_params, tags);
//     auto results = new uint32_t;
//     double correct = 0;
//     for (size_t i = 0; i < data_num; i++) {
//         index.search(data + i * data_dim, 1, L, results);
//         if (*results == i) {
//             correct += 1;
//         }
//     }
//     std::cout << "Reccall:" << correct / data_num;
//     return 0;
// }

// int vename_disk() {

//     diskann::Metric metric = diskann::Metric::L2;

//     size_t data_num = 10000, data_dim = 256;
//     int L = 200;
//     int R = 64;
//     std::string data_type = "float";
//     std::string label_type = "uint32";

//     auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
//             .build();

//     diskann::Index<float, uint32_t, uint32_t> index(metric, data_dim, data_num, false, false, false, false, 4, false);

//     std::mt19937 rng;
//     rng.seed(47);
//     std::uniform_real_distribution<> distrib_real;
//     float * data = new float[data_dim * data_num];
//     std::vector<uint32_t> tags;
//     for (uint32_t i = 0; i < data_dim * data_num; i++) {
//         data[i] = distrib_real(rng);
//         tags.push_back(i);
//     }
//     index.build(data, data_num, index_build_params, tags);

//     std::stringstream graph_stream;
//     std::stringstream tag_stream;
//     std::stringstream data_stream;



//     index.save(graph_stream, tag_stream, data_stream);

//     std::cout << graph_stream.str().size() << std::endl;
//     std::cout << tag_stream.str().size() << std::endl;
//     std::cout << data_stream.str().size() << std::endl;

//     diskann::Index<float, uint32_t, uint32_t> index2(metric, data_dim, data_num, false, false, false, false, 4, false);

//     index2.load(graph_stream, tag_stream, data_stream, 1, 200);
//     auto results = new uint32_t;
//     double correct = 0;
//     for (size_t i = 0; i < data_num; i++) {
//         index2.search(data + i * data_dim, 1, L, results);
//         if (*results == i) {
//             correct += 1;
//         }
//     }
//     std::cout << "Reccall:" << correct / data_num;
//     return 0;
// }

// int vename_disk_pq() {

//     diskann::Metric metric = diskann::Metric::L2;

//     size_t data_num = 10000, data_dim = 256;
//     int L = 200;
//     int R = 64;
//     std::string data_type = "float";
//     std::string label_type = "uint32";
//     float p_val = 0.5;
//     size_t disk_pq_dims = 8;


//     auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
//             .build();

//     diskann::Index<float, uint32_t, uint32_t> index(metric, data_dim, data_num, false, false, false, false, 4, false);

//     std::mt19937 rng;
//     rng.seed(47);
//     std::uniform_real_distribution<> distrib_real;
//     float * data = new float[data_dim * data_num];
//     std::vector<uint32_t> tags;
//     for (uint32_t i = 0; i < data_dim * data_num; i++) {
//         data[i] = distrib_real(rng);
//         tags.push_back(i);
//     }
//     index.build(data, data_num, index_build_params, tags);

//     std::stringstream graph_stream;
//     std::stringstream tag_stream;
//     std::stringstream data_stream;

//     index.save(graph_stream, tag_stream, data_stream);

//     std::stringstream pq_pivots_stream;
//     std::stringstream disk_pq_compressed_vectors;
//     diskann::generate_disk_quantized_data<float>(data_stream, pq_pivots_stream,
//                                  disk_pq_compressed_vectors, metric, p_val, disk_pq_dims);

//     std::stringstream diskann_stream;


//     diskann::create_disk_layout<float>(data_stream, graph_stream, diskann_stream, "");

//     std::string filename = "index.out";
//     std::ofstream output(filename);

//     output << diskann_stream.rdbuf();
//     output.close();

//     std::shared_ptr<AlignedFileReader> reader = nullptr;
//     reader.reset(new LinuxAlignedFileReader());

//     diskann::PQFlashIndex<float> index1(reader, metric);

//     index1.load_from_separate_paths(omp_get_num_procs(), filename.c_str(), pq_pivots_stream, disk_pq_compressed_vectors);

//     std::cout << diskann_stream.str().size() << std::endl;

//     auto results = new uint64_t ;
//     auto dists = new float;
//     double correct = 0;
//     for (size_t i = 0; i < data_num; i++) {
//         index1.cached_beam_search(data + i * data_dim, 1, L, results, dists, 4);
//         if (*results == i) {
//             correct += 1;
//         }
//     }
//     std::cout << "Reccall:" << correct / data_num;
//     return 0;
// }

void float_diskann() {
    int dim = 256;               // Dimension of the elements
    int max_elements = 10000;    // Maximum number of elements, should be known beforehand
    int M = 16;                 // Tightly connected with internal dimensionality of the data
    // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int ef_runtime = 200;
    int p_val = 0.5;  // p_val represents how much original data is selected during the training of pq compressed vectors.
    int chunks_num = 32; // chunks_num represents the dimensionality of the compressed vector.
    std::string disk_layout_file = "/tmp/index.out";
    // Initing index
    nlohmann::json index_parameters{
            {"dtype", "float32"},
            {"metric_type", "l2"},
            {"dim", dim},
            {"max_elements", max_elements},
            {"R", M},
            {"L", ef_construction},
            {"p_val", 0.5},
            {"disk_pq_dims", chunks_num},
            {"disk_layout_file", disk_layout_file},
    };
    auto diskann = vsag::Factory::create("diskann", index_parameters.dump());

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
    diskann->Build(dataset);

    // Serialize
    {
        vsag::BinarySet bs = diskann->Serialize();
        diskann = nullptr;

        vsag::Binary pq_b = bs.Get(vsag::DISKANN_PQ);
        std::ofstream pq("diskann_pq.index", std::ios::binary);
        pq.write((const char*)pq_b.data.get(), pq_b.size);
        pq.close();

        vsag::Binary compressed_vector_b = bs.Get(vsag::DISKANN_COMPRESSED_VECTOR);
        std::ofstream compressed("diskann_compressed_vector.index", std::ios::binary);
        compressed.write((const char*)compressed_vector_b.data.get(), compressed_vector_b.size);
        compressed.close();

        vsag::Binary layout_file_b = bs.Get(vsag::DISKANN_LAYOUT_FILE);
        std::ofstream layout(disk_layout_file, std::ios::binary);
        layout.write((const char*)layout_file_b.data.get(), layout_file_b.size);
        layout.close();
    }
    // Deserialize
    {
        vsag::BinarySet bs;

        std::ifstream pq("diskann_pq.index", std::ios::binary);
        pq.seekg(0, std::ios::end);
        size_t size = pq.tellg();
        pq.seekg(0, std::ios::beg);
        std::shared_ptr<int8_t[]> buff(new int8_t[size]);
        pq.read(reinterpret_cast<char *>(buff.get()), size);
        vsag::Binary pq_b{
                .data = buff,
                .size = size,
        };
        bs.Set(vsag::DISKANN_PQ, pq_b);

        std::ifstream compressed("diskann_compressed_vector.index", std::ios::binary);
        compressed.seekg(0, std::ios::end);
        size = compressed.tellg();
        compressed.seekg(0, std::ios::beg);
        buff.reset(new int8_t[size]);
        compressed.read(reinterpret_cast<char *>(buff.get()), size);
        vsag::Binary compressed_vector_b{
                .data = buff,
                .size = size,
        };
        bs.Set(vsag::DISKANN_COMPRESSED_VECTOR, compressed_vector_b);

        diskann = vsag::Factory::create("diskann", index_parameters.dump());

        std::cout << "#####" << std::endl;
        diskann->Deserialize(bs);
    }
    // Query the elements for themselves and measure recall 1@2
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        vsag::Dataset query;
        query.SetNumElements(1);
        query.SetDim(dim);
        query.SetFloat32Vectors(data + i * dim);
        query.SetOwner(false);
        nlohmann::json parameters{
                {"data_num", 1},
                {"ef_search", ef_runtime},
                {"beam_search", 4},
                {"io_limit", 200}
        };
        int64_t k = 2;
        auto result = diskann->KnnSearch(query, k, parameters.dump());
        if (result.GetNumElements() == 1) {
            if (result.GetIds()[0] == i) {
                correct++;
            }
        }
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << std::endl;
}


int main() {
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