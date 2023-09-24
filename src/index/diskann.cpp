//
// Created by root on 9/6/23.
//

#include "diskann.h"

#include <local_file_reader.h>

#include <functional>
#include <future>
#include <nlohmann/json.hpp>
#include <utility>

#include "vsag/index.h"
#include "vsag/readerset.h"

namespace vsag {

class LocalMemoryReader : public Reader {
public:
    LocalMemoryReader(std::stringstream& file) {
        file_ << file.rdbuf();
        file_.seekg(0, std::ios::end);
        size_ = file_.tellg();
    }

    ~LocalMemoryReader() = default;

    virtual void
    Read(uint64_t offset, uint64_t len, void* dest) override {
        file_.seekg(offset, std::ios::beg);
        file_.read((char*)dest, len);
    }

    virtual uint64_t
    Size() const override {
        return size_;
    }

private:
    std::stringstream file_;
    uint64_t size_;
};

DiskANN::DiskANN(
    diskann::Metric metric, std::string data_type, int L, int R, float p_val, size_t disk_pq_dims)
    : metric_(metric),
      L_(L),
      R_(R),
      p_val_(p_val),
      data_type_(data_type),
      disk_pq_dims_(disk_pq_dims) {
    batch_read = [&](const std::vector<read_request>& requests) -> void {
        std::vector<std::future<void>> futures;
        for (int i = 0; i < requests.size(); ++i) {
            futures.push_back(std::async(
                std::launch::async,
                [&](uint64_t a, uint64_t b, void* c) { disk_layout_reader->Read(a, b, c); },
                std::get<0>(requests[i]),
                std::get<1>(requests[i]),
                std::get<2>(requests[i])));
        }
        for (int i = 0; i < requests.size(); ++i) {
            futures[i].wait();
        }

        //        for (int i = 0; i < requests.size(); ++i) {
        //            disk_layout_reader->Read(
        //                std::get<0>(requests[i]), std::get<1>(requests[i]), std::get<2>(requests[i]));
        //        }
    };
}

void
DiskANN::Build(const Dataset& base) {
    auto vectors = base.GetFloat32Vectors();
    auto ids = base.GetIds();
    auto data_num = base.GetNumElements();
    auto data_dim = base.GetDim();

    std::stringstream graph_stream;
    std::stringstream tag_stream;
    std::stringstream data_stream;

    {  // build graph
        build_index = std::make_shared<diskann::Index<float, int64_t, int64_t>>(
            metric_, data_dim, data_num, false, true, false, false, 0, false);
        std::vector<int64_t> tags(ids, ids + data_num);
        auto index_build_params = diskann::IndexWriteParametersBuilder(L_, R_).build();
        build_index->build(vectors, data_num, index_build_params, tags);
        build_index->save(graph_stream, tag_stream, data_stream);
    }

    {  // build disk layout
        //        std::stringstream diskann_stream;
        //        diskann::create_disk_layout<float>(data_stream, graph_stream, diskann_stream, "");
        //        std::ofstream output(disk_layout_file);
        //        output << diskann_stream.rdbuf();
        //        output.close();
    }

    diskann::generate_disk_quantized_data<float>(data_stream,
                                                 pq_pivots_stream_,
                                                 disk_pq_compressed_vectors_,
                                                 metric_,
                                                 p_val_,
                                                 disk_pq_dims_);

    diskann::create_disk_layout<float>(data_stream, graph_stream, disk_layout_stream_, "");

    disk_layout_reader = std::make_shared<LocalMemoryReader>(disk_layout_stream_);

    reader.reset(new LocalFileReader(batch_read));
    index.reset(new diskann::PQFlashIndex<float>(reader, metric_));
    index->load_from_separate_paths(
        omp_get_num_procs(), pq_pivots_stream_, disk_pq_compressed_vectors_);
}

Dataset
DiskANN::KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) {
    nlohmann::json param = nlohmann::json::parse(parameters);
    Dataset result;
    if (!index)
        return std::move(result);
    auto query_num = query.GetNumElements();
    auto query_dim = query.GetDim();
    size_t beam_search = param["diskann"]["beam_search"];
    size_t io_limit = param["diskann"]["io_limit"];
    size_t ef_search = param["diskann"]["ef_search"];
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
        //        distances[i * k] = static_cast<float>(stats->n_ios);
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
    std::string pq_str = std::move(pq_pivots_stream_.str());
    std::shared_ptr<int8_t[]> pq_pivots(new int8_t[pq_str.size()]);
    std::copy(pq_str.begin(), pq_str.end(), pq_pivots.get());
    Binary pq_binary{
        .data = pq_pivots,
        .size = pq_str.size(),
    };

    bs.Set(DISKANN_PQ, pq_binary);

    pq_str = std::move(disk_pq_compressed_vectors_.str());
    std::shared_ptr<int8_t[]> compressed_vectors(new int8_t[pq_str.size()]);
    std::copy(pq_str.begin(), pq_str.end(), compressed_vectors.get());
    Binary compressed_binary{
        .data = compressed_vectors,
        .size = pq_str.size(),
    };
    bs.Set(DISKANN_COMPRESSED_VECTOR, compressed_binary);

    std::string disk_layout = std::move(disk_layout_stream_.str());
    std::shared_ptr<int8_t[]> layout_file(new int8_t[disk_layout.size()]);
    std::copy(disk_layout.begin(), disk_layout.end(), layout_file.get());

    Binary layout_binary{
        .data = layout_file,
        .size = disk_layout.size(),
    };
    bs.Set(DISKANN_LAYOUT_FILE, layout_binary);
    return bs;
}

void
DiskANN::Deserialize(const BinarySet& binary_set) {
    auto pq_pivots = binary_set.Get(DISKANN_PQ);
    pq_pivots_stream_.write((char*)pq_pivots.data.get(), pq_pivots.size);

    auto compressed_vector = binary_set.Get(DISKANN_COMPRESSED_VECTOR);
    disk_pq_compressed_vectors_.write((char*)compressed_vector.data.get(), compressed_vector.size);

    auto disk_layout = binary_set.Get(DISKANN_LAYOUT_FILE);
    disk_layout_stream_.write((char*)disk_layout.data.get(), disk_layout.size);

    disk_layout_reader = std::make_shared<LocalMemoryReader>(disk_layout_stream_);

    reader.reset(new LocalFileReader(batch_read));
    index.reset(new diskann::PQFlashIndex<float>(reader, metric_));
    index->load_from_separate_paths(
        omp_get_num_procs(), pq_pivots_stream_, disk_pq_compressed_vectors_);
}

void
DiskANN::Deserialize(const ReaderSet& reader_set) {
    {
        auto pq_reader = reader_set.Get(DISKANN_PQ);
        char pq_pivots_data[pq_reader->Size()];
        pq_reader->Read(0, pq_reader->Size(), pq_pivots_data);
        pq_pivots_stream_.write(pq_pivots_data, pq_reader->Size());
        pq_pivots_stream_.seekg(0);
    }

    {
        auto compressed_vector_reader = reader_set.Get(DISKANN_COMPRESSED_VECTOR);
        char compressed_vector_data[compressed_vector_reader->Size()];
        compressed_vector_reader->Read(0, compressed_vector_reader->Size(), compressed_vector_data);
        disk_pq_compressed_vectors_.write(compressed_vector_data, compressed_vector_reader->Size());
        disk_pq_compressed_vectors_.seekg(0);
    }
    disk_layout_reader = reader_set.Get(DISKANN_LAYOUT_FILE);
    reader.reset(new LocalFileReader(batch_read));
    index.reset(new diskann::PQFlashIndex<float>(reader, metric_));
    index->load_from_separate_paths(
        omp_get_num_procs(), pq_pivots_stream_, disk_pq_compressed_vectors_);
}

}  // namespace vsag
