#pragma once

#include <nlohmann/json.hpp>

#include "../index/diskann.h"
#include "../index/hnsw.h"
#include "vsag/constants.h"

namespace vsag {

inline void
check_common_parameters(const nlohmann::json& params) {
    CHECK_ARGUMENT(params.contains(PARAMETER_DTYPE),
                   fmt::format("parameters must contains {}", PARAMETER_DTYPE));
    CHECK_ARGUMENT(
        params[PARAMETER_DTYPE] == DATATYPE_FLOAT32,
        fmt::format("parameters[{}] supports {} only now", PARAMETER_DTYPE, DATATYPE_FLOAT32));
    CHECK_ARGUMENT(params.contains(PARAMETER_METRIC_TYPE),
                   fmt::format("parameters must contains {}", PARAMETER_METRIC_TYPE));
    CHECK_ARGUMENT(params.contains(PARAMETER_DIM),
                   fmt::format("parameters must contains {}", PARAMETER_DIM));
}

struct CreateHnswParameters {
public:
    static CreateHnswParameters
    FromJson(const std::string& json_string) {
        nlohmann::json params = nlohmann::json::parse(json_string);
        check_common_parameters(params);

        CreateHnswParameters obj;

        // set obj.space
        CHECK_ARGUMENT(params.contains(INDEX_HNSW),
                       fmt::format("parameters must contains {}", INDEX_HNSW));
        if (params[PARAMETER_METRIC_TYPE] == METRIC_L2) {
            obj.space = std::make_shared<hnswlib::L2Space>(params[PARAMETER_DIM]);
        } else if (params[PARAMETER_METRIC_TYPE] == METRIC_IP) {
            obj.space = std::make_shared<hnswlib::InnerProductSpace>(params[PARAMETER_DIM]);
        } else {
            std::string metric = params[PARAMETER_METRIC_TYPE];
            throw std::invalid_argument(fmt::format("parameters[{}] must in [{}, {}], now is {}",
                                                    PARAMETER_METRIC_TYPE,
                                                    METRIC_L2,
                                                    METRIC_IP,
                                                    metric));
        }

        // set obj.max_degree
        CHECK_ARGUMENT(
            params[INDEX_HNSW].contains(HNSW_PARAMETER_M),
            fmt::format("parameters[{}] must contains {}", INDEX_HNSW, HNSW_PARAMETER_M));
        obj.max_degree = params[INDEX_HNSW][HNSW_PARAMETER_M];

        // set obj.ef_construction
        CHECK_ARGUMENT(
            params[INDEX_HNSW].contains(HNSW_PARAMETER_CONSTRUCTION),
            fmt::format(
                "parameters[{}] must contains {}", INDEX_HNSW, HNSW_PARAMETER_CONSTRUCTION));
        obj.ef_construction = params[INDEX_HNSW][HNSW_PARAMETER_CONSTRUCTION];

        return obj;
    }

public:
    // required vars
    std::shared_ptr<hnswlib::SpaceInterface> space;
    int64_t max_degree;
    int64_t ef_construction;

private:
    CreateHnswParameters() = default;
};

struct CreateDiskannParameters {
public:
    static CreateDiskannParameters
    FromJson(const std::string& json_string) {
        nlohmann::json params = nlohmann::json::parse(json_string);
        check_common_parameters(params);

        CreateDiskannParameters obj;

        // set ojb.dim
        obj.dim = params[PARAMETER_DIM];

        // set ojb.dtype
        obj.dtype = params[PARAMETER_DTYPE];

        // set obj.metric
        CHECK_ARGUMENT(params.contains(INDEX_DISKANN),
                       fmt::format("parameters must contains {}", INDEX_DISKANN));
        if (params[PARAMETER_METRIC_TYPE] == METRIC_L2) {
            obj.metric = diskann::Metric::L2;
        } else if (params[PARAMETER_METRIC_TYPE] == METRIC_IP) {
            obj.metric = diskann::Metric::INNER_PRODUCT;
        } else {
            std::string metric = params[PARAMETER_METRIC_TYPE];
            throw std::invalid_argument(fmt::format("parameters[{}] must in [{}, {}], now is {}",
                                                    PARAMETER_METRIC_TYPE,
                                                    METRIC_L2,
                                                    METRIC_IP,
                                                    metric));
        }

        // set obj.max_degree
        CHECK_ARGUMENT(
            params[INDEX_DISKANN].contains(DISKANN_PARAMETER_L),
            fmt::format("parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_L));
        obj.max_degree = params[INDEX_DISKANN][DISKANN_PARAMETER_L];

        // set obj.ef_construction
        CHECK_ARGUMENT(
            params[INDEX_DISKANN].contains(DISKANN_PARAMETER_R),
            fmt::format("parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_R));
        obj.ef_construction = params[INDEX_DISKANN][DISKANN_PARAMETER_R];

        // set obj.pq_dims
        CHECK_ARGUMENT(
            params[INDEX_DISKANN].contains(DISKANN_PARAMETER_DISK_PQ_DIMS),
            fmt::format(
                "parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_DISK_PQ_DIMS));
        obj.pq_dims = params[INDEX_DISKANN][DISKANN_PARAMETER_DISK_PQ_DIMS];

        // set obj.pq_sample_rate
        CHECK_ARGUMENT(
            params[INDEX_DISKANN].contains(DISKANN_PARAMETER_P_VAL),
            fmt::format("parameters[{}] must contains {}", INDEX_DISKANN, DISKANN_PARAMETER_P_VAL));
        obj.pq_sample_rate = params[INDEX_DISKANN][DISKANN_PARAMETER_P_VAL];

        // optional
        // set obj.use_preload
        if (params[INDEX_DISKANN].contains(DISKANN_PARAMETER_PRELOAD)) {
            obj.use_preload = params[INDEX_DISKANN][DISKANN_PARAMETER_PRELOAD];
        }
        // set obj.use_reference
        if (params[INDEX_DISKANN].contains(DISKANN_PARAMETER_USE_REFERENCE)) {
            obj.use_reference = params[INDEX_DISKANN][DISKANN_PARAMETER_USE_REFERENCE];
        }
        // set obj.use_opq
        if (params[INDEX_DISKANN].contains(DISKANN_PARAMETER_USE_OPQ)) {
            obj.use_opq = params[INDEX_DISKANN][DISKANN_PARAMETER_USE_OPQ];
        }
        return obj;
    }

public:
    // require vars
    int64_t dim;
    std::string dtype;
    diskann::Metric metric;
    int64_t max_degree;
    int64_t ef_construction;
    int64_t pq_dims;
    float pq_sample_rate;

    // optional vars with default value
    bool use_preload = false;
    bool use_reference = true;
    bool use_opq = false;

private:
    CreateDiskannParameters() = default;
};

}  // namespace vsag
