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

const static float MACRO_TO_MILLI = 1000;
const static int64_t DATA_LIMIT = 2;
const static size_t MAXIMAL_BEAM_SEARCH = 64;
const static size_t MINIMAL_BEAM_SEARCH = 1;
const static int MINIMAL_R = 8;
const static int MAXIMAL_R = 64;

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

Binary
convertStreamToBinary(const std::stringstream& stream) {
    std::streambuf* buf = stream.rdbuf();
    std::streamsize size = buf->pubseekoff(0, stream.end, stream.in);  // get the stream buffer size
    buf->pubseekpos(0, stream.in);                                     // reset pointer pos
    std::shared_ptr<int8_t[]> binary_data(new int8_t[size]);
    buf->sgetn((char*)binary_data.get(), size);
    Binary binary{
        .data = binary_data,
        .size = (size_t)size,
    };
    return std::move(binary);
}

DiskANN::DiskANN(diskann::Metric metric,
                 std::string data_type,
                 int L,
                 int R,
                 float p_val,
                 size_t disk_pq_dims,
                 int64_t dim,
                 bool preload)
    : metric_(metric),
      L_(L),
      R_(R),
      p_val_(p_val),
      data_type_(data_type),
      disk_pq_dims_(disk_pq_dims),
      dim_(dim),
      preload_(preload) {
    status_ = IndexStatus::EMPTY;
    batch_read_ = [&](const std::vector<read_request>& requests) -> void {
        std::vector<std::future<void>> futures;
        for (int i = 0; i < requests.size(); ++i) {
            futures.push_back(std::async(
                std::launch::async,
                [&](uint64_t offset, uint64_t len, void* dest) {
                    disk_layout_reader_->Read(offset, len, dest);
                },
                std::get<0>(requests[i]),
                std::get<1>(requests[i]),
                std::get<2>(requests[i])));
        }
        for (int i = 0; i < requests.size(); ++i) {
            futures[i].wait();
        }
    };

    R_ = std::min(MAXIMAL_R, std::max(MINIMAL_R, R_));
}

tl::expected<std::vector<int64_t>, index_error>
DiskANN::Build(const Dataset& base) {
    SlowTaskTimer t("diskann build");

    std::vector<int64_t> failed_ids;
    if (base.GetNumElements() < DATA_LIMIT) {
        return tl::unexpected(index_error::no_enough_data);
    }

    if (this->index_) {
        return tl::unexpected(index_error::build_twice);
    }

    auto vectors = base.GetFloat32Vectors();
    auto ids = base.GetIds();
    auto data_num = base.GetNumElements();
    auto data_dim = base.GetDim();

    if (data_dim != dim_) {
        spdlog::error("dimension not equal: base(" + std::to_string(data_dim) + ") index(" +
                      std::to_string(dim_) + ")");
        return tl::unexpected(index_error::dimension_not_equal);
    }

    std::stringstream data_stream;

    try {
        {  // build graph
            build_index_ = std::make_shared<diskann::Index<float, int64_t, int64_t>>(
                metric_, data_dim, data_num, false, true, false, false, 0, false);
            std::vector<int64_t> tags(ids, ids + data_num);
            auto index_build_params = diskann::IndexWriteParametersBuilder(L_, R_).build();
            failed_ids = build_index_->build(vectors, data_num, index_build_params, tags);
            build_index_->save(graph_stream_, tag_stream_, data_stream);
        }

        diskann::generate_disk_quantized_data<float>(data_stream,
                                                     pq_pivots_stream_,
                                                     disk_pq_compressed_vectors_,
                                                     metric_,
                                                     p_val_,
                                                     disk_pq_dims_);

        diskann::create_disk_layout<float>(data_stream, graph_stream_, disk_layout_stream_, "");

        disk_layout_reader_ = std::make_shared<LocalMemoryReader>(disk_layout_stream_);
        reader_.reset(new LocalFileReader(batch_read_));
        index_.reset(new diskann::PQFlashIndex<float, int64_t>(reader_, metric_));
        index_->load_from_separate_paths(
            omp_get_num_procs(), pq_pivots_stream_, disk_pq_compressed_vectors_, tag_stream_);
        if (preload_) {
            index_->load_graph(graph_stream_);
        } else {
            graph_stream_.clear();
        }
        status_ = IndexStatus::MEMORY;
    } catch (std::runtime_error e) {
        return tl::unexpected(index_error::internal_error);
    }

    return std::move(failed_ids);
}

tl::expected<Dataset, index_error>
DiskANN::KnnSearch(const Dataset& query,
                   int64_t k,
                   const std::string& parameters,
                   BitsetPtr invalid) const {
    SlowTaskTimer t("diskann search", 100);
    nlohmann::json param = nlohmann::json::parse(parameters);
    if (!index_) {
        spdlog::error("failed to search: diskann index is empty");
        return tl::unexpected(index_error::index_empty);
    }

    if (k <= 0) {
        spdlog::error("invalid parameter: k (" + std::to_string(k) + ")");
        return tl::unexpected(index_error::invalid_parameter);
    }

    k = std::min(k, GetNumElements());
    auto query_num = query.GetNumElements();
    auto query_dim = query.GetDim();

    if (query_dim != dim_) {
        spdlog::error("dimension not equal: query(" + std::to_string(query_dim) + ") index(" +
                      std::to_string(dim_) + ")");
        return tl::unexpected(index_error::dimension_not_equal);
    }

    std::function<bool(uint32_t)> filter_ = nullptr;
    if (invalid) {
        if (invalid->Capcity() < GetNumElements()) {
            spdlog::error("number of invalid is less than the size of index");
            return tl::unexpected(index_error::internal_error);
        }
        filter_ = [&](uint32_t offset) -> bool { return invalid->Get(offset); };
    }

    if (!param.contains(INDEX_DISKANN)) {
        spdlog::error("missing parameter: {}", INDEX_DISKANN);
        return tl::unexpected(index_error::invalid_parameter);
    }

    if (!param[INDEX_DISKANN].contains(DISKANN_PARAMETER_BEAM_SEARCH)) {
        spdlog::error("missing parameter: {}", DISKANN_PARAMETER_BEAM_SEARCH);
        return tl::unexpected(index_error::invalid_parameter);
    }

    if (!param[INDEX_DISKANN].contains(DISKANN_PARAMETER_IO_LIMIT)) {
        spdlog::error("missing parameter: {}", DISKANN_PARAMETER_IO_LIMIT);
        return tl::unexpected(index_error::invalid_parameter);
    }

    if (!param[INDEX_DISKANN].contains(DISKANN_PARAMETER_EF_SEARCH)) {
        spdlog::error("missing parameter: {}", DISKANN_PARAMETER_EF_SEARCH);
        return tl::unexpected(index_error::invalid_parameter);
    }

    size_t beam_search = param[INDEX_DISKANN][DISKANN_PARAMETER_BEAM_SEARCH];
    int64_t io_limit = param[INDEX_DISKANN][DISKANN_PARAMETER_IO_LIMIT];
    int64_t ef_search = param[INDEX_DISKANN][DISKANN_PARAMETER_EF_SEARCH];

    // ensure that in the topK scenario, ef_search > k and io_limit > k.
    ef_search = std::max(ef_search, k);
    io_limit = std::max(io_limit, k);
    beam_search = std::min(beam_search, MAXIMAL_BEAM_SEARCH);
    beam_search = std::max(beam_search, MINIMAL_BEAM_SEARCH);

    uint64_t labels[query_num * k];
    auto distances = new float[query_num * k];
    auto ids = new int64_t[query_num * k];
    diskann::QueryStats query_stats[query_num];
    for (int i = 0; i < query_num; i++) {
        try {
            double time_cost = 0;
            {
                Timer timer(time_cost);
                if (preload_) {
                    k = index_->cached_beam_search_memory(query.GetFloat32Vectors() + i * dim_,
                                                          k,
                                                          ef_search,
                                                          labels + i * k,
                                                          distances + i * k,
                                                          beam_search,
                                                          filter_,
                                                          io_limit,
                                                          false,
                                                          query_stats + i);
                } else {
                    k = index_->cached_beam_search(query.GetFloat32Vectors() + i * dim_,
                                                   k,
                                                   ef_search,
                                                   labels + i * k,
                                                   distances + i * k,
                                                   beam_search,
                                                   filter_,
                                                   io_limit,
                                                   false,
                                                   query_stats + i);
                }
            }
            {
                std::lock_guard<std::mutex> lock(stats_mutex_);
                result_queues_[STATSTIC_KNN_IO].Push(query_stats[i].n_ios);
                result_queues_[STATSTIC_KNN_HOP].Push(query_stats[i].n_hops);
                result_queues_[STATSTIC_KNN_TIME].Push(time_cost);
                result_queues_[STATSTIC_KNN_CACHE_HIT].Push(query_stats[i].n_cache_hits);
                result_queues_[STATSTIC_KNN_IO_TIME].Push(
                    (query_stats[i].io_us / query_stats[i].n_ios) / MACRO_TO_MILLI);
            }

        } catch (std::runtime_error e) {
            spdlog::error(std::string("failed to perform knn search on diskann: ") + e.what());
        }
        //        distances[i * k] = static_cast<float>(stats->n_ios);
    }

    Dataset result;
    result.NumElements(query.GetNumElements()).Dim(0);

    if (k == 0) {
        delete[] distances;
        delete[] ids;
        return std::move(result);
    }
    for (int i = 0; i < query_num * k; ++i) {
        ids[i] = static_cast<int64_t>(labels[i]);
    }

    result.NumElements(query_num).Dim(k).Distances(distances).Ids(ids);
    return std::move(result);
}

tl::expected<Dataset, index_error>
DiskANN::RangeSearch(const Dataset& query,
                     float radius,
                     const std::string& parameters,
                     BitsetPtr invalid) const {
    nlohmann::json param = nlohmann::json::parse(parameters);

    if (radius <= 0) {
        spdlog::error("invalid parameter: radius (" + std::to_string(radius) + ")");
        return tl::unexpected(index_error::invalid_parameter);
    }

    int64_t query_num = query.GetNumElements();
    int64_t query_dim = query.GetDim();

    if (!index_) {
        spdlog::error("failed to search: {} index is empty", INDEX_DISKANN);
        return tl::unexpected(index_error::index_empty);
    }

    if (query_dim != dim_) {
        spdlog::error("dimension not equal: query(" + std::to_string(query_dim) + ") index(" +
                      std::to_string(dim_) + ")");
        return tl::unexpected(index_error::dimension_not_equal);
    }

    if (query_num != 1) {
        spdlog::error("number of elements is NOT 1 in query database");
        return tl::unexpected(index_error::internal_error);
    }

    std::function<bool(uint32_t)> filter_ = nullptr;
    if (invalid) {
        if (invalid->Capcity() < GetNumElements()) {
            spdlog::error("number of invalid is less than the size of index");
            return tl::unexpected(index_error::internal_error);
        }
        filter_ = [&](uint32_t offset) -> bool { return invalid->Get(offset); };
    }

    if (!param.contains(INDEX_DISKANN)) {
        spdlog::error("missing parameter: {}", INDEX_DISKANN);
        return tl::unexpected(index_error::invalid_parameter);
    }

    if (!param[INDEX_DISKANN].contains(DISKANN_PARAMETER_BEAM_SEARCH)) {
        spdlog::error("missing parameter: {}", DISKANN_PARAMETER_BEAM_SEARCH);
        return tl::unexpected(index_error::invalid_parameter);
    }

    if (!param[INDEX_DISKANN].contains(DISKANN_PARAMETER_IO_LIMIT)) {
        spdlog::error("missing parameter: {}", DISKANN_PARAMETER_IO_LIMIT);
        return tl::unexpected(index_error::invalid_parameter);
    }

    if (!param[INDEX_DISKANN].contains(DISKANN_PARAMETER_EF_SEARCH)) {
        spdlog::error("missing parameter: {}", DISKANN_PARAMETER_EF_SEARCH);
        return tl::unexpected(index_error::invalid_parameter);
    }

    size_t beam_search = param[INDEX_DISKANN][DISKANN_PARAMETER_BEAM_SEARCH];
    int64_t ef_search = param[INDEX_DISKANN][DISKANN_PARAMETER_EF_SEARCH];

    if (ef_search <= 0) {
        spdlog::error("invalid parameter: {} (" + std::to_string(ef_search) + ")", ef_search);
        return tl::unexpected(index_error::invalid_parameter);
    }

    beam_search = std::min(beam_search, MAXIMAL_BEAM_SEARCH);
    beam_search = std::max(beam_search, MINIMAL_BEAM_SEARCH);

    std::vector<uint64_t> labels;
    std::vector<float> range_distances;
    diskann::QueryStats query_stats;
    try {
        double time_cost = 0;
        {
            Timer timer(time_cost);
            index_->range_search(query.GetFloat32Vectors(),
                                 radius,
                                 ef_search,
                                 ef_search * 2,
                                 labels,
                                 range_distances,
                                 beam_search,
                                 filter_,
                                 preload_,
                                 &query_stats);
        }
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);

            result_queues_[STATSTIC_RANGE_IO].Push(query_stats.n_ios);
            result_queues_[STATSTIC_RANGE_HOP].Push(query_stats.n_hops);
            result_queues_[STATSTIC_RANGE_TIME].Push(time_cost);
            result_queues_[STATSTIC_RANGE_CACHE_HIT].Push(query_stats.n_cache_hits);
            result_queues_[STATSTIC_RANGE_IO_TIME].Push((query_stats.io_us / query_stats.n_ios) /
                                                        MACRO_TO_MILLI);
        }
    } catch (std::runtime_error e) {
        spdlog::error(std::string("failed to perform knn search on diskann: ") + e.what());
    }
    int64_t k = labels.size();

    Dataset result;
    result.Dim(0).NumElements(query_num);
    if (k == 0) {
        return std::move(result);
    }

    auto dis = new float[k];
    auto ids = new int64_t[k];
    for (int i = 0; i < k; ++i) {
        ids[i] = static_cast<int64_t>(labels[i]);
        dis[i] = range_distances[i];
    }

    result.NumElements(query_num).Dim(k).Distances(dis).Ids(ids);
    return std::move(result);
}

tl::expected<BinarySet, index_error>
DiskANN::Serialize() const {
    if (status_ == IndexStatus::EMPTY) {
        spdlog::error("failed to serialize: {} index is empty", INDEX_DISKANN);
        return tl::unexpected(index_error::index_empty);
    }
    SlowTaskTimer t("diskann serialize");
    try {
        BinarySet bs;

        bs.Set(DISKANN_PQ, convertStreamToBinary(pq_pivots_stream_));
        bs.Set(DISKANN_COMPRESSED_VECTOR, convertStreamToBinary(disk_pq_compressed_vectors_));
        bs.Set(DISKANN_LAYOUT_FILE, convertStreamToBinary(disk_layout_stream_));
        bs.Set(DISKANN_TAG_FILE, convertStreamToBinary(tag_stream_));
        if (preload_) {
            bs.Set(DISKANN_GRAPH, convertStreamToBinary(graph_stream_));
        }
        return bs;
    } catch (const std::bad_alloc& e) {
        return tl::unexpected(index_error::no_enough_memory);
    }
}

tl::expected<void, index_error>
DiskANN::Deserialize(const BinarySet& binary_set) {
    SlowTaskTimer t("diskann deserialize");

    if (this->index_) {
        spdlog::error("failed to deserialize: index is not empty");
        return tl::unexpected(index_error::index_not_empty);
    }

    try {
        auto pq_pivots = binary_set.Get(DISKANN_PQ);
        pq_pivots_stream_.write((char*)pq_pivots.data.get(), pq_pivots.size);

        auto compressed_vector = binary_set.Get(DISKANN_COMPRESSED_VECTOR);
        disk_pq_compressed_vectors_.write((char*)compressed_vector.data.get(),
                                          compressed_vector.size);

        auto disk_layout = binary_set.Get(DISKANN_LAYOUT_FILE);
        disk_layout_stream_.write((char*)disk_layout.data.get(), disk_layout.size);

        auto tag_file = binary_set.Get(DISKANN_TAG_FILE);
        tag_stream_.write((char*)tag_file.data.get(), tag_file.size);

        disk_layout_reader_ = std::make_shared<LocalMemoryReader>(disk_layout_stream_);

        reader_.reset(new LocalFileReader(batch_read_));
        index_.reset(new diskann::PQFlashIndex<float, int64_t>(reader_, metric_));
        index_->load_from_separate_paths(
            omp_get_num_procs(), pq_pivots_stream_, disk_pq_compressed_vectors_, tag_stream_);

        auto graph = binary_set.Get(DISKANN_GRAPH);
        if (preload_) {
            if (graph.data) {
                graph_stream_.write((char*)graph.data.get(), graph.size);
                index_->load_graph(graph_stream_);
            } else {
                spdlog::error("missing file: {} when deserialize diskann index", DISKANN_GRAPH);
                return tl::unexpected(index_error::missing_file);
            }
        } else {
            if (graph.data) {
                spdlog::warn("serialize without using file: {} ", DISKANN_GRAPH);
            }
        }
        status_ = IndexStatus::MEMORY;
    } catch (std::runtime_error e) {
        spdlog::error(std::string("failed to deserialize: ") + e.what());
        return tl::unexpected(index_error::internal_error);
    }

    return {};
}

tl::expected<void, index_error>
DiskANN::Deserialize(const ReaderSet& reader_set) {
    SlowTaskTimer t("diskann deserialize");

    if (this->index_) {
        spdlog::error("failed to deserialize: {} is not empty", INDEX_DISKANN);
        return tl::unexpected(index_error::index_not_empty);
    }
    try {
        std::stringstream pq_pivots_stream, disk_pq_compressed_vectors, graph, tag_stream;

        {
            auto pq_reader = reader_set.Get(DISKANN_PQ);
            auto pq_pivots_data = std::make_unique<char[]>(pq_reader->Size());
            pq_reader->Read(0, pq_reader->Size(), pq_pivots_data.get());
            pq_pivots_stream.write(pq_pivots_data.get(), pq_reader->Size());
            pq_pivots_stream.seekg(0);
        }

        {
            auto compressed_vector_reader = reader_set.Get(DISKANN_COMPRESSED_VECTOR);
            auto compressed_vector_data =
                std::make_unique<char[]>(compressed_vector_reader->Size());
            compressed_vector_reader->Read(
                0, compressed_vector_reader->Size(), compressed_vector_data.get());
            disk_pq_compressed_vectors.write(compressed_vector_data.get(),
                                             compressed_vector_reader->Size());
            disk_pq_compressed_vectors.seekg(0);
        }

        {
            auto tag_reader = reader_set.Get(DISKANN_TAG_FILE);
            auto tag_data = std::make_unique<char[]>(tag_reader->Size());
            tag_reader->Read(0, tag_reader->Size(), tag_data.get());
            tag_stream.write(tag_data.get(), tag_reader->Size());
            tag_stream.seekg(0);
        }

        disk_layout_reader_ = reader_set.Get(DISKANN_LAYOUT_FILE);
        reader_.reset(new LocalFileReader(batch_read_));
        index_.reset(new diskann::PQFlashIndex<float, int64_t>(reader_, metric_));
        index_->load_from_separate_paths(
            omp_get_num_procs(), pq_pivots_stream, disk_pq_compressed_vectors, tag_stream);

        auto graph_reader = reader_set.Get(DISKANN_GRAPH);
        if (preload_) {
            if (graph_reader) {
                auto graph_data = std::make_unique<char[]>(graph_reader->Size());
                graph_reader->Read(0, graph_reader->Size(), graph_data.get());
                graph.write(graph_data.get(), graph_reader->Size());
                graph.seekg(0);
                index_->load_graph(graph);
            } else {
                spdlog::error("miss file: {} when deserialize diskann index", DISKANN_GRAPH);
                return tl::unexpected(index_error::missing_file);
            }
        } else {
            if (graph_reader) {
                spdlog::warn("serialize without using file: {} ", DISKANN_GRAPH);
            }
        }
        status_ = IndexStatus::HYBRID;
    } catch (std::exception e) {
        spdlog::error(std::string("failed to deserialize: ") + e.what());
        return tl::unexpected(index_error::read_error);
    }

    return {};
}

std::string
DiskANN::GetStats() const {
    nlohmann::json j;
    j[STATSTIC_DATA_NUM] = GetNumElements();
    j[STATSTIC_INDEX_NAME] = INDEX_DISKANN;
    j[STATSTIC_MEMORY] = GetMemoryUsage();

    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        for (auto& item : result_queues_) {
            j[item.first] = item.second.GetAvgResult();
        }
    }

    return j.dump();
}

}  // namespace vsag
