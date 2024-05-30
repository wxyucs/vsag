
#include <sstream>

#include "logger.h"
#include "utils.h"
#include "vsag/errors.h"
#include "vsag/expected.hpp"
#include "vsag/index.h"

namespace vsag {

bool
is_multiple_of_four(int n) {
    return (n > 0) && (n % 4 == 0);
}

std::string
parameter_string(const std::string& metric_type,
                 int64_t dimension,
                 int64_t hnsw_max_degree,
                 int64_t hnsw_ef_construction,
                 int64_t diskann_max_degree,
                 int64_t diskann_ef_construction,
                 int64_t diskann_pq_dims,
                 float diskann_pq_sample_rate,
                 bool use_conjugate_graph) {
    // use {{ to escape curlies
    return fmt::format(R"(
                        {{
                            "dtype": "float32",
                            "metric_type": "{}",
                            "dim": {},
                            "hnsw": {{
                            	"max_degree": {},
                            	"ef_construction": {},
                                "use_conjugate_graph": {}
                            }},
                            "diskann": {{
                                "max_degree": {},
                                "ef_construction": {},
                                "pq_dims": {},
                                "pq_sample_rate": {} 
                            }}
                        }}
                        )",
                       metric_type,
                       dimension,
                       hnsw_max_degree,
                       hnsw_ef_construction,
                       use_conjugate_graph,
                       diskann_max_degree,
                       diskann_ef_construction,
                       diskann_pq_dims,
                       diskann_pq_sample_rate);
}

tl::expected<std::string, Error>
generate_build_parameters(std::string metric_type,
                          int64_t num_elements,
                          int64_t dim,
                          bool use_conjugate_graph) {
    logger::debug("metric_type: {}, num_elements: {}, dim: {}, use_conjugate_graph: {}",
                  metric_type,
                  num_elements,
                  dim,
                  use_conjugate_graph);

    // check metric_type
    std::transform(
        metric_type.begin(), metric_type.end(), metric_type.begin(), [](unsigned char c) {
            return std::tolower(c);
        });
    if (metric_type != "l2" and metric_type != "ip") {
        return tl::unexpected(Error(
            ErrorType::INVALID_ARGUMENT,
            fmt::format("failed to generate build parameter: metric_type({}) is not in [l2, ip]",
                        metric_type)));
    }

    // check dimension
    if (not is_multiple_of_four(dim)) {
        return tl::unexpected(
            Error(ErrorType::INVALID_ARGUMENT,
                  "failed to generate build parameter: dimension is not multiple of 4"));
    }

    // compression ratio: 1/16
    // dim*FP32 -> pq_dims*INT8
    int64_t pq_dims = dim / 4;
    logger::debug("pq_dims: {}", pq_dims);

    // rule-based parameters
    if (Number(num_elements).in_range(1, 2'000'000)) {
        return parameter_string(
            metric_type, dim, 12, 100, 12, 100, pq_dims, 0.1, use_conjugate_graph);
    } else if (Number(num_elements).in_range(2'000'000, 5'000'000)) {
        return parameter_string(
            metric_type, dim, 16, 200, 16, 200, pq_dims, 0.1, use_conjugate_graph);
    } else if (Number(num_elements).in_range(5'000'000, 10'000'000)) {
        return parameter_string(
            metric_type, dim, 24, 300, 24, 300, pq_dims, 0.1, use_conjugate_graph);
    } else if (Number(num_elements).in_range(10'000'000, 17'000'000)) {
        return parameter_string(
            metric_type, dim, 48, 500, 48, 500, pq_dims, 0.1, use_conjugate_graph);
    } else {
        return tl::unexpected(
            Error(ErrorType::INVALID_ARGUMENT,
                  fmt::format(
                      "failed to generate build parameter: unsupported num_elements({}) or dim({})",
                      num_elements,
                      dim)));
    }
}

}  // namespace vsag
