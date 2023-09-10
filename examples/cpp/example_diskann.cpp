//
// Created by inabao on 2023/8/21.
//
#include "vsag/vsag.h"
#include <sstream>
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

void stream_test() {
    std::stringstream originalStream;
    originalStream.write("1234", 4);
    std::cout << originalStream.str().size() << std::endl;
    std::cout << "Stream content after restoring: " << originalStream.str() << std::endl;

}

int main() {
    vename_disk_pq();
    return 0;
}