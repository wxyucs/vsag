//
// Created by root on 9/6/23.
//

#include "diskann.h"

#include <nlohmann/json.hpp>
namespace vsag {

DiskANN::DiskANN(
    diskann::Metric metric, std::string data_type, int L, int R, float p_val, size_t disk_pq_dims)
    : metric_(metric),
      L_(L),
      R_(R),
      p_val_(p_val),
      data_type_(data_type),
      disk_pq_dims_(disk_pq_dims) {
}

void
DiskANN::Build(const Dataset& base) {
    auto vectors = base.GetFloat32Vectors();
    auto ids = base.GetIds();
    auto data_num = base.GetNumElements();
    auto data_dim = base.GetDim();
    diskann::Index<float, uint32_t, uint32_t> build_index(
        metric_, data_dim, data_num, false, true, false, false, 4, false);
    std::vector<uint32_t> tags;
    for (int i = 0; i < data_num; ++i) {
        tags.push_back(static_cast<uint32_t>(ids[i]));
    }
    auto index_build_params = diskann::IndexWriteParametersBuilder(L_, R_).build();
    build_index.build(vectors, data_num, index_build_params, tags);

    std::stringstream graph_stream;
    std::stringstream tag_stream;
    std::stringstream data_stream;

    build_index.save(graph_stream, tag_stream, data_stream);

    std::stringstream pq_pivots_stream;
    std::stringstream disk_pq_compressed_vectors;
    diskann::generate_disk_quantized_data<float>(
        data_stream, pq_pivots_stream, disk_pq_compressed_vectors, metric_, p_val_, disk_pq_dims_);

    std::stringstream diskann_stream;

    diskann::create_disk_layout<float>(data_stream, graph_stream, diskann_stream, "");

    std::string filename = "index.out";
    std::ofstream output(filename);

    output << diskann_stream.rdbuf();
    output.close();
    reader.reset(new LinuxAlignedFileReader());
    index = std::make_shared<diskann::PQFlashIndex<float>>(reader, metric_);
    index->load_from_separate_paths(
        omp_get_num_procs(), filename.c_str(), pq_pivots_stream, disk_pq_compressed_vectors);
}

Dataset
DiskANN::KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) {
    nlohmann::json param = nlohmann::json::parse(parameters);
    Dataset result;
    if (!index)
        return result;
    auto query_num = query.GetNumElements();
    auto query_dim = query.GetDim();
    size_t beam_search = param["beam_search"]; 
    size_t io_limit = param["io_limit"];
    size_t ef_search = param["ef_search"];
    uint64_t labels[query_num * k];
    auto distances = new float[query_num * k];
    auto ids = new int64_t[query_num * k];
    auto stats = new diskann::QueryStats[query_num];
    for (int i = 0; i < query_num; i ++) {
       index->cached_beam_search(query.GetFloat32Vectors() + i * query_dim, k, ef_search, labels + i * k, distances + i * k, beam_search, io_limit, false, stats + i); 
       distances[i * k] = static_cast<float>(stats->n_ios); 
    }
    for (int i = 0; i < query_num * k; ++i) {
        ids[i] = static_cast<int64_t>(labels[i]);
    }

    result.SetOwner(false);
    result.SetNumElements(query_num);
    result.SetDim(k);
    result.SetDistances(distances);
    result.SetIds(ids);
    return result;
}


}  // namespace vsag