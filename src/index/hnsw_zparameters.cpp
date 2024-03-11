#include "hnsw_zparameters.h"

#include <fmt/format-inl.h>

#include <nlohmann/json.hpp>

#include "../common.h"
#include "vsag/constants.h"

namespace vsag {

CreateHnswParameters
CreateHnswParameters::FromJson(const std::string& json_string) {
    nlohmann::json params = nlohmann::json::parse(json_string);

    CHECK_ARGUMENT(params.contains(PARAMETER_DTYPE),
                   fmt::format("parameters must contains {}", PARAMETER_DTYPE));
    CHECK_ARGUMENT(
        params[PARAMETER_DTYPE] == DATATYPE_FLOAT32,
        fmt::format("parameters[{}] supports {} only now", PARAMETER_DTYPE, DATATYPE_FLOAT32));
    CHECK_ARGUMENT(params.contains(PARAMETER_METRIC_TYPE),
                   fmt::format("parameters must contains {}", PARAMETER_METRIC_TYPE));
    CHECK_ARGUMENT(params.contains(PARAMETER_DIM),
                   fmt::format("parameters must contains {}", PARAMETER_DIM));

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
    CHECK_ARGUMENT(params[INDEX_HNSW].contains(HNSW_PARAMETER_M),
                   fmt::format("parameters[{}] must contains {}", INDEX_HNSW, HNSW_PARAMETER_M));
    obj.max_degree = params[INDEX_HNSW][HNSW_PARAMETER_M];
    CHECK_ARGUMENT((5 <= obj.max_degree) and (obj.max_degree <= 64),
                   fmt::format("max_degree({}) must in range[5, 64]", obj.max_degree));

    // set obj.ef_construction
    CHECK_ARGUMENT(
        params[INDEX_HNSW].contains(HNSW_PARAMETER_CONSTRUCTION),
        fmt::format("parameters[{}] must contains {}", INDEX_HNSW, HNSW_PARAMETER_CONSTRUCTION));
    obj.ef_construction = params[INDEX_HNSW][HNSW_PARAMETER_CONSTRUCTION];
    CHECK_ARGUMENT((obj.max_degree <= obj.ef_construction) and (obj.ef_construction <= 1000),
                   fmt::format("ef_construction({}) must in range[$max_degree({}), 64]",
                               obj.ef_construction,
                               obj.max_degree));

    return obj;
}

HnswSearchParameters
HnswSearchParameters::FromJson(const std::string& json_string) {
    nlohmann::json params = nlohmann::json::parse(json_string);

    HnswSearchParameters obj;

    // set obj.ef_search
    CHECK_ARGUMENT(params.contains(INDEX_HNSW),
                   fmt::format("parameters must contains {}", INDEX_HNSW));

    CHECK_ARGUMENT(
        params[INDEX_HNSW].contains(HNSW_PARAMETER_EF_RUNTIME),
        fmt::format("parameters[{}] must contains {}", INDEX_HNSW, HNSW_PARAMETER_EF_RUNTIME));
    obj.ef_search = params[INDEX_HNSW][HNSW_PARAMETER_EF_RUNTIME];
    CHECK_ARGUMENT((1 <= obj.ef_search) and (obj.ef_search <= 1000),
                   fmt::format("ef_search({}) must in range[1, 1000]", obj.ef_search));

    return obj;
}

}  // namespace vsag
