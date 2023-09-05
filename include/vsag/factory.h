#pragma once

#include <memory>
#include <nlohmann/json.hpp>

#include "index.h"

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
    static std::shared_ptr<Index>
    create(const std::string& name, nlohmann::json parameters);

private:
    Factory() = default;
};

}  // namespace vsag
