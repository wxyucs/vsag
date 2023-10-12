//
// Created by root on 9/6/23.
//

#include "diskann.h"

#include <local_file_reader.h>
#include <spdlog/spdlog.h>

#include <exception>
#include <functional>
#include <future>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

#include "../utils.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
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
        std::lock_guard<std::mutex> lock(mutex_);
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
    std::mutex mutex_;
};

DiskANN::DiskANN(
    diskann::Metric metric, std::string data_type, int L, int R, float p_val, size_t disk_pq_dims)
    : metric_(metric),
      L_(L),
      R_(R),
      p_val_(p_val),
      data_type_(data_type),
      disk_pq_dims_(disk_pq_dims) {
    status = IndexStatus::EMPTY;
    batch_read = [&](const std::vector<read_request>& requests) -> void {
        std::vector<std::future<void>> futures;
        for (int i = 0; i < requests.size(); ++i) {
            futures.push_back(std::async(
                std::launch::async,
                [&](uint64_t offset, uint64_t len, void* dest) {
                    disk_layout_reader->Read(offset, len, dest);
                },
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

tl::expected<int64_t, index_error>
DiskANN::Build(const Dataset& base) {
    SlowTaskTimer t("diskann build");
    auto vectors = base.GetFloat32Vectors();
    auto ids = base.GetIds();
    auto data_num = base.GetNumElements();
    auto data_dim = base.GetDim();

    std::stringstream graph_stream;
    std::stringstream tag_stream;
    std::stringstream data_stream;

    try {
        {  // build graph
            build_index = std::make_shared<diskann::Index<float, int64_t, int64_t>>(
                metric_, data_dim, data_num, false, true, false, false, 0, false);
            std::vector<int64_t> tags(ids, ids + data_num);
            auto index_build_params = diskann::IndexWriteParametersBuilder(L_, R_).build();
            build_index->build(vectors, data_num, index_build_params, tags);
            build_index->save(graph_stream, tag_stream, data_stream);
        }

        diskann::generate_disk_quantized_data<float>(data_stream,
                                                     pq_pivots_stream_,
                                                     disk_pq_compressed_vectors_,
                                                     metric_,
                                                     p_val_,
                                                     disk_pq_dims_);

        memory_usage_ += pq_pivots_stream_.str().size() + disk_pq_compressed_vectors_.str().size();

        diskann::create_disk_layout<float>(data_stream, graph_stream, disk_layout_stream_, "");

        memory_usage_ += disk_layout_stream_.str().size();

        disk_layout_reader = std::make_shared<LocalMemoryReader>(disk_layout_stream_);

        reader.reset(new LocalFileReader(batch_read));
        index.reset(new diskann::PQFlashIndex<float>(reader, metric_));
        index->load_from_separate_paths(
            omp_get_num_procs(), pq_pivots_stream_, disk_pq_compressed_vectors_);
        status = IndexStatus::MEMORY;
        memory_usage_ += pq_pivots_stream_.str().size() + disk_pq_compressed_vectors_.str().size();
    } catch (std::runtime_error e) {
        return tl::unexpected(index_error::internal_error);
    }

    return this->GetNumElements();
}

tl::expected<Dataset, index_error>
DiskANN::KnnSearch(const Dataset& query, int64_t k, const std::string& parameters) const {
    SlowTaskTimer t("diskann search", 100);
    nlohmann::json param = nlohmann::json::parse(parameters);
    Dataset result;
    if (!index)
        return std::move(result);
    k = std::min(k, GetNumElements());
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
        try {
            index->cached_beam_search(query.GetFloat32Vectors() + i * query_dim,
                                      k,
                                      ef_search,
                                      labels + i * k,
                                      distances + i * k,
                                      beam_search,
                                      io_limit,
                                      false,
                                      stats + i);
        } catch (std::runtime_error e) {
            spdlog::error(std::string("failed to perform knn search on diskann: ") + e.what());
        }
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

tl::expected<BinarySet, index_error>
DiskANN::Serialize() const {
    SlowTaskTimer t("diskann serialize");
    try {
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
    } catch (const std::bad_alloc& e) {
        return tl::unexpected(index_error::no_enough_memory);
    }
}

tl::expected<void, index_error>
DiskANN::Deserialize(const BinarySet& binary_set) {
    SlowTaskTimer t("diskann deserialize");
    try {
        auto pq_pivots = binary_set.Get(DISKANN_PQ);
        pq_pivots_stream_.write((char*)pq_pivots.data.get(), pq_pivots.size);

        auto compressed_vector = binary_set.Get(DISKANN_COMPRESSED_VECTOR);
        disk_pq_compressed_vectors_.write((char*)compressed_vector.data.get(),
                                          compressed_vector.size);

        auto disk_layout = binary_set.Get(DISKANN_LAYOUT_FILE);
        disk_layout_stream_.write((char*)disk_layout.data.get(), disk_layout.size);

        disk_layout_reader = std::make_shared<LocalMemoryReader>(disk_layout_stream_);

        reader.reset(new LocalFileReader(batch_read));
        index.reset(new diskann::PQFlashIndex<float>(reader, metric_));
        index->load_from_separate_paths(
            omp_get_num_procs(), pq_pivots_stream_, disk_pq_compressed_vectors_);
        status = IndexStatus::MEMORY;
    } catch (std::runtime_error e) {
        spdlog::error(std::string("failed to deserialize: ") + e.what());
        return tl::unexpected(index_error::internal_error);
    }

    return {};
    reader.reset(new LocalFileReader(batch_read));
    index.reset(new diskann::PQFlashIndex<float>(reader, metric_));
    index->load_from_separate_paths(
        omp_get_num_procs(), pq_pivots_stream_, disk_pq_compressed_vectors_);

    memory_usage_ +=
        (pq_pivots_stream_.str().size() + disk_pq_compressed_vectors_.str().size()) * 2 +
        disk_layout_stream_.str().size();
    status = IndexStatus::MEMORY;
}

tl::expected<void, index_error>
DiskANN::Deserialize(const ReaderSet& reader_set) {
    SlowTaskTimer t("diskann deserialize");

    std::stringstream pq_pivots_stream, disk_pq_compressed_vectors;
    memory_usage_ = 0;
    {
        auto pq_reader = reader_set.Get(DISKANN_PQ);
        char pq_pivots_data[pq_reader->Size()];
        pq_reader->Read(0, pq_reader->Size(), pq_pivots_data);
        pq_pivots_stream.write(pq_pivots_data, pq_reader->Size());
        pq_pivots_stream.seekg(0);
        memory_usage_ += pq_reader->Size();
    }

    {
        auto compressed_vector_reader = reader_set.Get(DISKANN_COMPRESSED_VECTOR);
        char compressed_vector_data[compressed_vector_reader->Size()];
        compressed_vector_reader->Read(0, compressed_vector_reader->Size(), compressed_vector_data);
        disk_pq_compressed_vectors.write(compressed_vector_data, compressed_vector_reader->Size());
        disk_pq_compressed_vectors.seekg(0);
        memory_usage_ += compressed_vector_reader->Size();
    }
    disk_layout_reader = reader_set.Get(DISKANN_LAYOUT_FILE);
    reader.reset(new LocalFileReader(batch_read));
    index.reset(new diskann::PQFlashIndex<float>(reader, metric_));
    index->load_from_separate_paths(
        omp_get_num_procs(), pq_pivots_stream, disk_pq_compressed_vectors);
    status = IndexStatus::HYBRID;

    return {};
}

}  // namespace vsag
