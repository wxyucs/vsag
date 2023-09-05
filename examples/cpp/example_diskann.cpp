//
// Created by inabao on 2023/8/21.
//


#include <cstring>
#include <iostream>
#include <omp.h>

#include <hnswlib/hnswlib.h>
#include <ann_exception.h>
#include <index.h>
#include <index_factory.h>
#include <memory_mapper.h>
#include <utils.h>

#include <vsag/vsag.h>

int
venama_memory() {
    diskann::Metric metric = diskann::Metric::L2;

    size_t data_num = 10000, data_dim = 256;
    int L = 200;
    int R = 64;
    std::string data_type = "float";
    std::string label_type = "uint32";

    auto index_build_params = diskann::IndexWriteParametersBuilder(L, R).build();

    diskann::Index<float, uint32_t, uint32_t> index(
        metric, data_dim, data_num, false, false, false, false, 4, false);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[data_dim * data_num];
    std::vector<uint32_t> tags;
    for (uint32_t i = 0; i < data_dim * data_num; i++) {
        data[i] = distrib_real(rng);
        tags.push_back(i);
    }
    index.build(data, data_num, index_build_params, tags);
    auto results = new uint32_t;
    double correct = 0;
    for (size_t i = 0; i < data_num; i++) {
        index.search(data + i * data_dim, 1, L, results);
        if (*results == i) {
            correct += 1;
        }
    }
    std::cout << "Reccall:" << correct / data_num;
    return 0;
}

int
vename_disk() {
    diskann::Metric metric = diskann::Metric::L2;

    size_t data_num = 10000, data_dim = 256;
    int L = 200;
    int R = 64;
    std::string data_type = "float";
    std::string label_type = "uint32";

    auto index_build_params = diskann::IndexWriteParametersBuilder(L, R).build();

    diskann::Index<float, uint32_t, uint32_t> index(
        metric, data_dim, data_num, false, false, false, false, 4, false);

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    float* data = new float[data_dim * data_num];
    std::vector<uint32_t> tags;
    for (uint32_t i = 0; i < data_dim * data_num; i++) {
        data[i] = distrib_real(rng);
        tags.push_back(i);
    }
    index.build(data, data_num, index_build_params, tags);

    std::stringstream graph_stream;
    std::stringstream tag_stream;
    std::stringstream data_stream;

    index.save(graph_stream, tag_stream, data_stream);

    std::cout << graph_stream.str().size() << std::endl;
    std::cout << tag_stream.str().size() << std::endl;
    std::cout << data_stream.str().size() << std::endl;

    diskann::Index<float, uint32_t, uint32_t> index2(
        metric, data_dim, data_num, false, false, false, false, 4, false);

    index2.load(graph_stream, tag_stream, data_stream, 1, 200);
    auto results = new uint32_t;
    double correct = 0;
    for (size_t i = 0; i < data_num; i++) {
        index2.search(data + i * data_dim, 1, L, results);
        if (*results == i) {
            correct += 1;
        }
    }
    std::cout << "Reccall:" << correct / data_num;
    return 0;
}

void
stream_test() {
    std::stringstream originalStream;
    originalStream << "Hello, World!";

    // 输出原始流的内容
    std::cout << "Original stream content: " << originalStream.str() << std::endl;

    // 复制原始流的内容到临时流
    std::stringstream tempStream;
    tempStream << originalStream.rdbuf();

    // 在这里可以根据需要读取临时流的内容
    std::string s;
    //    tempStream >> s;
    tempStream.read((char*)s.c_str(), 2);

    // 输出恢复后的原始流内容
    std::cout << "Stream content after restoring: " << originalStream.str() << std::endl;
}

int
main() {
    vename_disk();
    return 0;
}
