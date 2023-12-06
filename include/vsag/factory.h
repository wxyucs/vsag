#pragma once

#include <memory>

#include "index.h"
#include "readerset.h"

namespace vsag {

class Factory {
public:
    /*
      HNSW.parameters:
	- dtype: string, required, one of [float32]
	- metric_type: string, required, one of [l2, ip]
	- max_elements: integer, required
	- M: integer, required
	- ef_construction: integer, required
	- ef_runtime: integer, optional
      Vamana: Not supported
     */
    static tl::expected<std::shared_ptr<Index>, index_error>
    CreateIndex(const std::string& name, const std::string& parameters);

    static std::shared_ptr<Reader>
    CreateLocalFileReader(const std::string& filename, int64_t base_offset, int64_t size);

private:
    Factory() = default;
};

}  // namespace vsag
