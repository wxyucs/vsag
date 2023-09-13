//
// Created by root on 9/6/23.
//

#include "diskann.h"
#include "vsag/index.h"
#include <nlohmann/json.hpp>
#include <utility>
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

    // build diskann layout
    std::ifstream file(disk_layout_file);

    std::stringstream graph_stream;
    std::stringstream tag_stream;
    std::stringstream data_stream;
        // build graph
    diskann::Index<float, int64_t, int64_t> build_index(
        metric_, data_dim, data_num, false, true, false, false, 0, false);
    std::vector<int64_t> tags(ids, ids + data_num);

    auto index_build_params = diskann::IndexWriteParametersBuilder(L_, R_).build();
    build_index.build(vectors, data_num, index_build_params, tags);
    build_index.save(graph_stream, tag_stream, data_stream);

    // build disk layout
    auto diskann_stream_ptr = new std::stringstream;
    std::stringstream &diskann_stream = *diskann_stream_ptr;
    diskann::create_disk_layout<float>(data_stream, graph_stream, diskann_stream, "");
    std::ofstream output(disk_layout_file);
    output << diskann_stream.rdbuf();
    output.close();
    delete diskann_stream_ptr;

    diskann::generate_disk_quantized_data<float>(
        data_stream, pq_pivots_stream_, disk_pq_compressed_vectors_, metric_, p_val_, disk_pq_dims_);

    reader.reset(new LinuxAlignedFileReader());
    index = std::make_shared<diskann::PQFlashIndex<float>>(reader, metric_);
    index->load_from_separate_paths(
        omp_get_num_procs(), disk_layout_file.c_str(), pq_pivots_stream_, disk_pq_compressed_vectors_);
}

Dataset
DiskANN::KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) {
    nlohmann::json param = nlohmann::json::parse(parameters);
    Dataset result;
    if (!index)
        return std::move(result);
    auto query_num = query.GetNumElements();
    auto query_dim = query.GetDim();
    size_t beam_search = param["beam_search"];
    size_t io_limit = param["io_limit"];
    size_t ef_search = param["ef_search"];
    uint64_t labels[query_num * k];
    auto distances = new float[query_num * k];
    auto ids = new int64_t[query_num * k];
    auto stats = new diskann::QueryStats[query_num];
    for (int i = 0; i < query_num; i++) {
        index->cached_beam_search(query.GetFloat32Vectors() + i * query_dim,
                                  k,
                                  ef_search,
                                  labels + i * k,
                                  distances + i * k,
                                  beam_search,
                                  io_limit,
                                  false,
                                  stats + i);
        distances[i * k] = static_cast<float>(stats->n_ios);
    }
    for (int i = 0; i < query_num * k; ++i) {
        ids[i] = static_cast<int64_t>(labels[i]);
    }

    result.SetNumElements(query_num);
    result.SetDim(k);
    result.SetDistances(distances);
    result.SetIds(ids);
    return std::move(result);
}

BinarySet
DiskANN::Serialize() {
    BinarySet bs;
    std::string pq_str = pq_pivots_stream_.str();
    std::shared_ptr<int8_t[]> pq_pivots(new int8_t[pq_str.size()]);
    std::copy(pq_str.begin(), pq_str.end(), pq_pivots.get());
    Binary pq_binary{
        .data = pq_pivots,
        .size = pq_str.size(),
    };

    bs.Set(DISKANN_PQ, pq_binary);

    pq_str = disk_pq_compressed_vectors_.str();
    std::shared_ptr<int8_t[]> compressed_vectors(new int8_t[pq_str.size()]);
    std::copy(pq_str.begin(), pq_str.end(), compressed_vectors.get());
    Binary compressed_binary{
        .data = compressed_vectors,
        .size = pq_str.size(),
    };
    bs.Set(DISKANN_COMPRESSED_VECTOR, compressed_binary);

    std::shared_ptr<int8_t[]> layout_file(new int8_t[disk_layout_file.size()]);
    std::copy(disk_layout_file.begin(), disk_layout_file.end(), layout_file.get());
    Binary layout_binary{
        .data = layout_file,
        .size = disk_layout_file.size(),
    };
    bs.Set(DISKANN_COMPRESSED_VECTOR, layout_binary);
    return bs;
}

void
DiskANN::Deserialize(const BinarySet& binary_set) {
    auto pq_pivots = binary_set.Get(DISKANN_PQ);
    pq_pivots_stream_.write((char *)pq_pivots.data.get(), pq_pivots.size);


    auto compressed_vector = binary_set.Get(DISKANN_COMPRESSED_VECTOR);
    disk_pq_compressed_vectors_.write((char *)compressed_vector.data.get(), compressed_vector.size);

    std::ostringstream oss;

    auto layout_file_data = binary_set.Get(DISKANN_LAYOUT_FILE);
    oss.write((char *)layout_file_data.data.get(), layout_file_data.size);
    disk_layout_file = oss.str();

    reader.reset(new LinuxAlignedFileReader());
    index.reset(new diskann::PQFlashIndex<float>(reader, metric_));
    index->load_from_separate_paths(
        omp_get_num_procs(), disk_layout_file.c_str(), pq_pivots_stream_, disk_pq_compressed_vectors_);
}

}  // namespace vsag
