//
// Created by root on 9/6/23.
//

#include "diskann.h"

#include <local_file_reader.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <exception>
#include <functional>
#include <future>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <utility>

#include "../common.h"
#include "../utils.h"
#include "./diskann_zparameters.h"
#include "vsag/constants.h"
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
const static int64_t MAX_IO_LIMIT = 512;
const static int VECTOR_PER_BLOCK = 1;
const static size_t MINIMAL_SECTOR_LEN = 4096;

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
                 bool preload,
                 bool use_reference,
                 bool use_opq)
    : metric_(metric),
      L_(L),
      R_(R),
      p_val_(p_val),
      data_type_(data_type),
      disk_pq_dims_(disk_pq_dims),
      dim_(dim),
      preload_(preload),
      use_reference_(use_reference),
      use_opq_(use_opq) {
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

    // When the length of the vector is too long, set sector_len_ to the size of storing a vector along with its linkage list.
    sector_len_ =
        std::max(MINIMAL_SECTOR_LEN,
                 (size_t)((dim + 1) * sizeof(float) + R_ * sizeof(uint32_t)) * VECTOR_PER_BLOCK);
}

tl::expected<std::vector<int64_t>, Error>
DiskANN::build(const Dataset& base) {
    SlowTaskTimer t("diskann build");

    try {
        auto data_dim = base.GetDim();
        CHECK_ARGUMENT(data_dim == dim_,
                       fmt::format("base.dim({}) must be equal to index.dim({})", data_dim, dim_));

        CHECK_ARGUMENT(
            base.GetNumElements() >= DATA_LIMIT,
            "number of elements must be greater equal than " + std::to_string(DATA_LIMIT));

        if (this->index_) {
            LOG_ERROR_AND_RETURNS(ErrorType::BUILD_TWICE, "failed to build index: build twice");
        }

        auto vectors = base.GetFloat32Vectors();
        auto ids = base.GetIds();
        auto data_num = base.GetNumElements();

        std::vector<size_t> failed_indexs;
        {
            // build graph
            build_index_ = std::make_shared<diskann::Index<float, int64_t, int64_t>>(
                metric_, data_dim, data_num, false, true, false, false, 0, false);
            std::vector<int64_t> tags(ids, ids + data_num);
            auto index_build_params = diskann::IndexWriteParametersBuilder(L_, R_).build();
            failed_indexs =
                build_index_->build(vectors, data_num, index_build_params, tags, use_reference_);
            build_index_->save(graph_stream_, tag_stream_);
            build_index_.reset();
        }

        diskann::generate_disk_quantized_data<float>(vectors,
                                                     data_num,
                                                     data_dim,
                                                     failed_indexs,
                                                     pq_pivots_stream_,
                                                     disk_pq_compressed_vectors_,
                                                     metric_,
                                                     p_val_,
                                                     disk_pq_dims_,
                                                     use_opq_);

        diskann::create_disk_layout<float>(vectors,
                                           data_num,
                                           data_dim,
                                           failed_indexs,
                                           graph_stream_,
                                           disk_layout_stream_,
                                           sector_len_,
                                           "");

        std::vector<int64_t> failed_ids;
        std::transform(failed_indexs.begin(),
                       failed_indexs.end(),
                       std::back_inserter(failed_ids),
                       [&ids](const auto& index) { return ids[index]; });

        disk_layout_reader_ = std::make_shared<LocalMemoryReader>(disk_layout_stream_);
        reader_.reset(new LocalFileReader(batch_read_));
        index_.reset(new diskann::PQFlashIndex<float, int64_t>(reader_, metric_, sector_len_));
        index_->set_sector_size(Option::Instance().GetSectorSize());
        index_->load_from_separate_paths(
            omp_get_num_procs(), pq_pivots_stream_, disk_pq_compressed_vectors_, tag_stream_);
        if (preload_) {
            index_->load_graph(graph_stream_);
        } else {
            graph_stream_.clear();
        }
        status_ = IndexStatus::MEMORY;
        return failed_ids;
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(
            ErrorType::INVALID_ARGUMENT, "failed to build(invalid argument): ", e.what());
    }
}

tl::expected<Dataset, Error>
DiskANN::knn_search(const Dataset& query,
                    int64_t k,
                    const std::string& parameters,
                    BitsetPtr invalid) const {
    SlowTaskTimer t("diskann knnsearch", 100);
    try {
        if (!index_) {
            LOG_ERROR_AND_RETURNS(ErrorType::INDEX_EMPTY,
                                  "failed to search: diskann index is empty");
        }

        // check query vector
        auto query_num = query.GetNumElements();
        auto query_dim = query.GetDim();
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));

        // check k
        CHECK_ARGUMENT(k > 0, fmt::format("k({}) must be greater than 0", k))
        k = std::min(k, GetNumElements());

        // check search parameters
        auto params = DiskannSearchParameters::FromJson(parameters);
        int64_t ef_search = params.ef_search;
        size_t beam_search = params.beam_search;
        int64_t io_limit = params.io_limit;
        bool reorder = params.use_reorder;

        // check filter
        std::function<bool(int64_t)> filter_ = nullptr;
        if (invalid) {
            CHECK_ARGUMENT(
                invalid->Capacity() >= GetNumElements(),
                fmt::format("invalid.capcity({}) must be greater equal than index.size({})",
                            invalid->Capacity(),
                            GetNumElements()));
            filter_ = [&](int64_t offset) -> bool { return invalid->Get(offset & ROW_ID_MASK); };
        }

        // ensure that in the topK scenario, ef_search > k and io_limit > k.
        ef_search = std::max(ef_search, k);
        io_limit = std::min(MAX_IO_LIMIT, std::max(io_limit, k));
        if (reorder && preload_) {
            io_limit = std::min((int64_t)Option::Instance().GetSectorSize(), io_limit);
        }
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
                                                              reorder,
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

            } catch (const std::runtime_error& e) {
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
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              "failed to perform knn_search(invalid argument): ",
                              e.what());
    }
}

tl::expected<Dataset, Error>
DiskANN::range_search(const Dataset& query,
                      float radius,
                      const std::string& parameters,
                      BitsetPtr invalid) const {
    SlowTaskTimer t("diskann rangesearch", 100);
    try {
        if (!index_) {
            LOG_ERROR_AND_RETURNS(
                ErrorType::INDEX_EMPTY,
                fmt::format("failed to search: {} index is empty", INDEX_DISKANN));
        }

        // check query vector
        int64_t query_num = query.GetNumElements();
        int64_t query_dim = query.GetDim();
        CHECK_ARGUMENT(
            query_dim == dim_,
            fmt::format("query.dim({}) must be equal to index.dim({})", query_dim, dim_));

        // check radius
        CHECK_ARGUMENT(radius >= 0, fmt::format("radius({}) must be greater equal than 0", radius))
        CHECK_ARGUMENT(query_num == 1, fmt::format("query.num({}) must be equal to 1", query_num));

        // check search parameters
        auto params = DiskannSearchParameters::FromJson(parameters);
        size_t beam_search = params.beam_search;
        int64_t ef_search = params.ef_search;
        CHECK_ARGUMENT(ef_search > 0,
                       fmt::format("ef_search({}) must be greater than 0", ef_search));

        // check filter
        std::function<bool(int64_t)> filter = nullptr;
        if (invalid) {
            CHECK_ARGUMENT(
                invalid->Capacity() >= GetNumElements(),
                fmt::format("invalid.capcity({}) must be greater equal than index.size({})",
                            invalid->Capacity(),
                            GetNumElements()));

            filter = [&](int64_t offset) -> bool { return invalid->Get(offset & ROW_ID_MASK); };
        }

        bool reorder = params.use_reorder;
        int64_t io_limit = params.io_limit;

        io_limit = std::min(MAX_IO_LIMIT, io_limit);
        if (reorder && preload_) {
            io_limit = std::min((int64_t)Option::Instance().GetSectorSize(), io_limit);
        }

        beam_search = std::min(beam_search, MAXIMAL_BEAM_SEARCH);
        beam_search = std::max(beam_search, MINIMAL_BEAM_SEARCH);

        std::vector<uint64_t> labels;
        std::vector<float> range_distances;
        diskann::QueryStats query_stats;

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
                                 io_limit,
                                 reorder,
                                 filter,
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
    } catch (const std::invalid_argument& e) {
        LOG_ERROR_AND_RETURNS(ErrorType::INVALID_ARGUMENT,
                              "falied to perform range_search(invalid argument): ",
                              e.what());
    }
}

tl::expected<BinarySet, Error>
DiskANN::serialize() const {
    if (status_ == IndexStatus::EMPTY) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_EMPTY,
                              fmt::format("failed to serialize: {} index is empty", INDEX_DISKANN))
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
        return tl::unexpected(Error(ErrorType::NO_ENOUGH_MEMORY, ""));
    }
}

tl::expected<void, Error>
DiskANN::deserialize(const BinarySet& binary_set) {
    SlowTaskTimer t("diskann deserialize");

    if (this->index_) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              "failed to deserialize: index is not empty")
    }

    auto pq_pivots = binary_set.Get(DISKANN_PQ);
    pq_pivots_stream_.write((char*)pq_pivots.data.get(), pq_pivots.size);

    auto compressed_vector = binary_set.Get(DISKANN_COMPRESSED_VECTOR);
    disk_pq_compressed_vectors_.write((char*)compressed_vector.data.get(), compressed_vector.size);

    auto disk_layout = binary_set.Get(DISKANN_LAYOUT_FILE);
    disk_layout_stream_.write((char*)disk_layout.data.get(), disk_layout.size);

    auto tag_file = binary_set.Get(DISKANN_TAG_FILE);
    tag_stream_.write((char*)tag_file.data.get(), tag_file.size);

    disk_layout_reader_ = std::make_shared<LocalMemoryReader>(disk_layout_stream_);

    reader_.reset(new LocalFileReader(batch_read_));
    index_.reset(new diskann::PQFlashIndex<float, int64_t>(reader_, metric_, sector_len_));
    index_->set_sector_size(Option::Instance().GetSectorSize());
    index_->load_from_separate_paths(
        omp_get_num_procs(), pq_pivots_stream_, disk_pq_compressed_vectors_, tag_stream_);

    auto graph = binary_set.Get(DISKANN_GRAPH);
    if (preload_) {
        if (graph.data) {
            graph_stream_.write((char*)graph.data.get(), graph.size);
            index_->load_graph(graph_stream_);
        } else {
            LOG_ERROR_AND_RETURNS(
                ErrorType::MISSING_FILE,
                fmt::format("missing file: {} when deserialize diskann index", DISKANN_GRAPH));
        }
    } else {
        if (graph.data) {
            spdlog::warn("serialize without using file: {} ", DISKANN_GRAPH);
        }
    }
    status_ = IndexStatus::MEMORY;

    return {};
}

tl::expected<void, Error>
DiskANN::deserialize(const ReaderSet& reader_set) {
    SlowTaskTimer t("diskann deserialize");

    if (this->index_) {
        LOG_ERROR_AND_RETURNS(ErrorType::INDEX_NOT_EMPTY,
                              fmt::format("failed to deserialize: {} is not empty", INDEX_DISKANN));
    }

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
        auto compressed_vector_data = std::make_unique<char[]>(compressed_vector_reader->Size());
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
    index_.reset(new diskann::PQFlashIndex<float, int64_t>(reader_, metric_, sector_len_));
    index_->set_sector_size(Option::Instance().GetSectorSize());
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
            LOG_ERROR_AND_RETURNS(
                ErrorType::MISSING_FILE,
                fmt::format("miss file: {} when deserialize diskann index", DISKANN_GRAPH));
        }
    } else {
        if (graph_reader) {
            spdlog::warn("serialize without using file: {} ", DISKANN_GRAPH);
        }
    }
    status_ = IndexStatus::HYBRID;

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

int64_t
DiskANN::GetEstimateBuildMemory(const int64_t num_elements) const {
    int64_t estimate_memory_usage = 0;
    // Memory usage of graph
    estimate_memory_usage +=
        num_elements * R_ * sizeof(uint32_t) + num_elements * (R_ + 1) * sizeof(uint32_t);
    // Memory usage of disk layout
    if (sector_len_ > MINIMAL_SECTOR_LEN) {
        estimate_memory_usage += num_elements * sector_len_ * sizeof(uint8_t);
    } else {
        size_t single_node =
            (size_t)((dim_ + 1) * sizeof(float) + R_ * sizeof(uint32_t)) * VECTOR_PER_BLOCK;
        size_t node_per_sector = MINIMAL_SECTOR_LEN / single_node;
        size_t sector_size = num_elements / node_per_sector + 1;
        estimate_memory_usage += sector_size * sector_len_ * sizeof(uint8_t);
    }
    // Memory usage of the ID mapping.
    estimate_memory_usage += num_elements * sizeof(int64_t) * 2;
    // Memory usage of the compressed PQ vectors.
    estimate_memory_usage += disk_pq_dims_ * num_elements * sizeof(uint8_t) * 2;
    // Memory usage of the PQ centers and chunck offsets.
    estimate_memory_usage += 256 * dim_ * sizeof(float) * (3 + 1) + dim_ * sizeof(uint32_t);
    return estimate_memory_usage;
}

}  // namespace vsag
