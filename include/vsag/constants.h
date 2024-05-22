#pragma once

namespace vsag {

constexpr const char* INDEX_DISKANN = "diskann";
constexpr const char* INDEX_HNSW = "hnsw";
constexpr const char* INDEX_FRESH_HNSW = "fresh_hnsw";
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
constexpr const char* DISKANN_TAG_FILE = "diskann_tag_file";
constexpr const char* DISKANN_GRAPH = "diskann_graph";
constexpr const char* SIMPLEFLAT_VECTORS = "simpleflat_vectors";
constexpr const char* SIMPLEFLAT_IDS = "simpleflat_ids";
constexpr const char* METRIC_L2 = "l2";
constexpr const char* METRIC_IP = "ip";
constexpr const char* DATATYPE_FLOAT32 = "float32";
constexpr const char* BLANK_INDEX = "blank_index";

// parameters
constexpr const char* PARAMETER_DTYPE = "dtype";
constexpr const char* PARAMETER_DIM = "dim";
constexpr const char* PARAMETER_METRIC_TYPE = "metric_type";
constexpr const char* PARAMETER_USE_CONJUGATE_GRAPH = "use_conjugate_graph";
constexpr const char* PARAMETER_USE_CONJUGATE_GRAPH_SEARCH = "use_conjugate_graph_search";

constexpr const char* DISKANN_PARAMETER_L = "ef_construction";
constexpr const char* DISKANN_PARAMETER_R = "max_degree";
constexpr const char* DISKANN_PARAMETER_P_VAL = "pq_sample_rate";
constexpr const char* DISKANN_PARAMETER_DISK_PQ_DIMS = "pq_dims";
constexpr const char* DISKANN_PARAMETER_PRELOAD = "use_pq_search";
constexpr const char* DISKANN_PARAMETER_USE_REFERENCE = "use_reference";
constexpr const char* DISKANN_PARAMETER_USE_OPQ = "use_opq";
constexpr const char* DISKANN_PARAMETER_USE_BSA = "use_bsa";

constexpr const char* DISKANN_PARAMETER_BEAM_SEARCH = "beam_search";
constexpr const char* DISKANN_PARAMETER_IO_LIMIT = "io_limit";
constexpr const char* DISKANN_PARAMETER_EF_SEARCH = "ef_search";
constexpr const char* DISKANN_PARAMETER_REORDER = "use_reorder";

constexpr const char* HNSW_PARAMETER_EF_RUNTIME = "ef_search";
constexpr const char* HNSW_PARAMETER_M = "max_degree";
constexpr const char* HNSW_PARAMETER_CONSTRUCTION = "ef_construction";
constexpr const char* HNSW_PARAMETER_USE_STATIC = "use_static";
constexpr const char* HNSW_PARAMETER_REVERSED_EDGES = "use_reversed_edges";

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

//Error message
constexpr const char* MESSAGE_PARAMETER = "invalid parameter";

}  // namespace vsag
