#include <stdexcept>

#include "index/diskann_zparameters.h"
#include "index/hnsw_zparameters.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/utils.h"

namespace vsag {

template <typename IndexOpParameters>
tl::expected<bool, Error>
check_parameters(const std::string& json_string) {
    try {
        IndexOpParameters::FromJson(json_string);
    } catch (const std::invalid_argument& e) {
        return tl::unexpected<Error>(ErrorType::INVALID_ARGUMENT, e.what());
    }

    return true;
}

tl::expected<bool, Error>
check_diskann_hnsw_build_parameters(const std::string& json_string) {
    if (auto ret = check_parameters<CreateHnswParameters>(json_string); not ret.has_value()) {
        return ret;
    }
    if (auto ret = check_parameters<CreateDiskannParameters>(json_string); not ret.has_value()) {
        return ret;
    }
    return true;
}

tl::expected<bool, Error>
check_diskann_hnsw_search_parameters(const std::string& json_string) {
    if (auto ret = check_parameters<HnswSearchParameters>(json_string); not ret.has_value()) {
        return ret;
    }
    if (auto ret = check_parameters<DiskannSearchParameters>(json_string); not ret.has_value()) {
        return ret;
    }
    return true;
}

}  // namespace vsag
