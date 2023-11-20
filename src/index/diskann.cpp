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
    // FIXME: A exception will be throwed out when there are less data.
    if (base.GetNumElements() < DATA_LIMIT) {
        return tl::unexpected(index_error::no_enough_data);
    }

    if (this->index) {
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

    std::stringstream tag_stream;
    std::stringstream data_stream;

    try {
        {  // build graph
            build_index = std::make_shared<diskann::Index<float, int64_t, int64_t>>(
                metric_, data_dim, data_num, false, true, false, false, 0, false);
            std::vector<int64_t> tags(ids, ids + data_num);
            auto index_build_params = diskann::IndexWriteParametersBuilder(L_, R_).build();
            build_index->build(vectors, data_num, index_build_params, tags);
            build_index->save(graph_stream_, tag_stream, data_stream);
        }

        diskann::generate_disk_quantized_data<float>(data_stream,
                                                     pq_pivots_stream_,
                                                     disk_pq_compressed_vectors_,
                                                     metric_,
                                                     p_val_,
                                                     disk_pq_dims_);

        diskann::create_disk_layout<float>(data_stream, graph_stream_, disk_layout_stream_, "");

        disk_layout_reader = std::make_shared<LocalMemoryReader>(disk_layout_stream_);
        reader.reset(new LocalFileReader(batch_read));
        index.reset(new diskann::PQFlashIndex<float>(reader, metric_));
        index->load_from_separate_paths(
            omp_get_num_procs(), pq_pivots_stream_, disk_pq_compressed_vectors_);
        if (preload_) {
            index->load_graph(graph_stream_);
        } else {
            graph_stream_.clear();
        }
        status = IndexStatus::MEMORY;
    } catch (std::runtime_error e) {
        return tl::unexpected(index_error::internal_error);
    }

    return this->GetNumElements();
}

tl::expected<Dataset, index_error>
DiskANN::KnnSearch(const Dataset& query,
                   int64_t k,
                   const std::string& parameters,
                   BitsetPtr invalid) const {
    SlowTaskTimer t("diskann search", 100);
    nlohmann::json param = nlohmann::json::parse(parameters);
    Dataset result;
    if (!index)
        return std::move(result);
    k = std::min(k, GetNumElements());
    auto query_num = query.GetNumElements();
    auto query_dim = query.GetDim();

    if (query_dim != dim_) {
        spdlog::error("dimension not equal: query(" + std::to_string(query_dim) + ") index(" +
                      std::to_string(dim_) + ")");
        return tl::unexpected(index_error::dimension_not_equal);
    }

    size_t beam_search = param["diskann"]["beam_search"];
    size_t io_limit = param["diskann"]["io_limit"];
    size_t ef_search = param["diskann"]["ef_search"];
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
                    index->cached_beam_search_memory(query.GetFloat32Vectors() + i * dim_,
                                                     k,
                                                     ef_search,
                                                     labels + i * k,
                                                     distances + i * k,
                                                     beam_search,
                                                     false,
                                                     0,
                                                     io_limit,
                                                     false,
                                                     query_stats + i);
                } else {
                    index->cached_beam_search(query.GetFloat32Vectors() + i * dim_,
                                              k,
                                              ef_search,
                                              labels + i * k,
                                              distances + i * k,
                                              beam_search,
                                              false,
                                              0,
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
    for (int i = 0; i < query_num * k; ++i) {
        ids[i] = static_cast<int64_t>(labels[i]);
    }

    result.SetNumElements(query_num);
    result.SetDim(k);
    result.SetDistances(distances);
    result.SetIds(ids);
    return std::move(result);
}

tl::expected<Dataset, index_error>
DiskANN::RangeSearch(const Dataset& query, float radius, const std::string& parameters) const {
    Dataset result;
    nlohmann::json param = nlohmann::json::parse(parameters);

    auto query_num = query.GetNumElements();
    auto query_dim = query.GetDim();

    result.SetDim(0);
    result.SetNumElements(query_num);

    if (!index)
        return std::move(result);

    if (query_dim != dim_) {
        spdlog::error("dimension not equal: query(" + std::to_string(query_dim) + ") index(" +
                      std::to_string(dim_) + ")");
        return tl::unexpected(index_error::dimension_not_equal);
    }

    assert(query_num == 1);

    size_t beam_search = param["diskann"]["beam_search"];
    size_t ef_search = param["diskann"]["ef_search"];

    std::vector<uint64_t> labels;
    std::vector<float> range_distances;
    diskann::QueryStats query_stats;
    try {
        double time_cost = 0;
        {
            Timer timer(time_cost);
            index->range_search(query.GetFloat32Vectors(),
                                radius,
                                ef_search,
                                ef_search * 2,
                                labels,
                                range_distances,
                                beam_search,
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

    if (k == 0) {
        return std::move(result);
    }

    auto dis = new float[k];
    auto ids = new int64_t[k];
    for (int i = 0; i < k; ++i) {
        ids[i] = static_cast<int64_t>(labels[i]);
        dis[i] = range_distances[i];
    }

    result.SetNumElements(query_num);
    result.SetDim(k);
    result.SetDistances(dis);
    result.SetIds(ids);
    return std::move(result);
}

tl::expected<BinarySet, index_error>
DiskANN::Serialize() const {
    if (status == IndexStatus::EMPTY) {
        spdlog::error("failed to serialize: diskann index is empty");
        return tl::unexpected(index_error::index_empty);
    }
    SlowTaskTimer t("diskann serialize");
    try {
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

        std::string disk_layout = disk_layout_stream_.str();
        std::shared_ptr<int8_t[]> layout_file(new int8_t[disk_layout.size()]);
        std::copy(disk_layout.begin(), disk_layout.end(), layout_file.get());

        Binary layout_binary{
            .data = layout_file,
            .size = disk_layout.size(),
        };
        bs.Set(DISKANN_LAYOUT_FILE, layout_binary);

        if (preload_) {
            std::string graph = graph_stream_.str();

            std::shared_ptr<int8_t[]> graph_content(new int8_t[graph.size()]);
            std::copy(graph.begin(), graph.end(), graph_content.get());

            Binary graph_binary{
                .data = graph_content,
                .size = graph.size(),
            };
            bs.Set(DISKANN_GRAPH, graph_binary);
        }

        return bs;
    } catch (const std::bad_alloc& e) {
        return tl::unexpected(index_error::no_enough_memory);
    }
}

tl::expected<void, index_error>
DiskANN::Deserialize(const BinarySet& binary_set) {
    SlowTaskTimer t("diskann deserialize");

    if (this->index) {
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

        disk_layout_reader = std::make_shared<LocalMemoryReader>(disk_layout_stream_);

        reader.reset(new LocalFileReader(batch_read));
        index.reset(new diskann::PQFlashIndex<float>(reader, metric_));
        index->load_from_separate_paths(
            omp_get_num_procs(), pq_pivots_stream_, disk_pq_compressed_vectors_);

        auto graph = binary_set.Get(DISKANN_GRAPH);
        if (preload_) {
            if (graph.data) {
                graph_stream_.write((char*)graph.data.get(), graph.size);
                index->load_graph(graph_stream_);
            } else {
                spdlog::error("missing file: {} when deserialize diskann index", DISKANN_GRAPH);
                return tl::unexpected(index_error::missing_file);
            }
        } else {
            if (graph.data) {
                spdlog::warn("serialize without using file: {} ", DISKANN_GRAPH);
            }
        }
        status = IndexStatus::MEMORY;
    } catch (std::runtime_error e) {
        spdlog::error(std::string("failed to deserialize: ") + e.what());
        return tl::unexpected(index_error::internal_error);
    }

    return {};
}

tl::expected<void, index_error>
DiskANN::Deserialize(const ReaderSet& reader_set) {
    SlowTaskTimer t("diskann deserialize");

    if (this->index) {
        spdlog::error("failed to deserialize: index is not empty");
        return tl::unexpected(index_error::index_not_empty);
    }
    try {
        std::stringstream pq_pivots_stream, disk_pq_compressed_vectors, graph;

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

        disk_layout_reader = reader_set.Get(DISKANN_LAYOUT_FILE);
        reader.reset(new LocalFileReader(batch_read));
        index.reset(new diskann::PQFlashIndex<float>(reader, metric_));
        index->load_from_separate_paths(
            omp_get_num_procs(), pq_pivots_stream, disk_pq_compressed_vectors);

        auto graph_reader = reader_set.Get(DISKANN_GRAPH);
        if (preload_) {
            if (graph_reader) {
                auto graph_data = std::make_unique<char[]>(graph_reader->Size());
                graph_reader->Read(0, graph_reader->Size(), graph_data.get());
                graph.write(graph_data.get(), graph_reader->Size());
                graph.seekg(0);
                index->load_graph(graph);
            } else {
                spdlog::error("miss file: {} when deserialize diskann index", DISKANN_GRAPH);
                return tl::unexpected(index_error::missing_file);
            }
        } else {
            if (graph_reader) {
                spdlog::warn("serialize without using file: {} ", DISKANN_GRAPH);
            }
        }
        status = IndexStatus::HYBRID;
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
