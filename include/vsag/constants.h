#pragma once

namespace vsag {

constexpr const char* INDEX_DISKANN = "diskann";
constexpr const char* INDEX_HNSW = "hnsw";
constexpr const char* DIM = "dim";
constexpr const char* NUM_ELEMENTS = "num_elements";
constexpr const char* IDS = "ids";
constexpr const char* DISTS = "dists";
constexpr const char* FLOAT32_VECTORS = "f32_vectors";
constexpr const char* INT8_VECTORS = "i8_vectors";
constexpr const char* HNSW_DATA = "hnsw_data";
constexpr const char* DISKANN_PQ = "diskann_qp";
constexpr const char* DISKANN_COMPRESSED_VECTOR = "diskann_compressed_vector";
constexpr const char* DISKANN_LAYOUT_FILE = "diskann_layout_file";
constexpr const char* SIMPLEFLAT_VECTORS = "simpleflat_vectors";
constexpr const char* SIMPLEFLAT_IDS = "simpleflat_ids";
constexpr const char* METRIC_L2 = "l2";
constexpr const char* METRIC_IP = "ip";

// statstic key

constexpr const char* STATSTIC_MEMORY = "memory";
constexpr const char* STATSTIC_INDEX_NAME = "index_name";
constexpr const char* STATSTIC_DATA_NUM = "data_num";

constexpr const char* STATSTIC_KNN_TIME = "knn_time";
constexpr const char* STATSTIC_KNN_IO = "knn_io";
constexpr const char* STATSTIC_KNN_HOP = "knn_hop";
constexpr const char* STATSTIC_KNN_IO_TIME = "knn_io_time";
constexpr const char* STATSTIC_KNN_CACHE_HIT = "knn_cache_hit";
constexpr const char* STATSTIC_RANGE_TIME = "range_time";
constexpr const char* STATSTIC_RANGE_IO = "range_io";
constexpr const char* STATSTIC_RANGE_HOP = "range_hop";
constexpr const char* STATSTIC_RANGE_CACHE_HIT = "range_cache_hit";
constexpr const char* STATSTIC_RANGE_IO_TIME = "range_io_time";

}  // namespace vsag
