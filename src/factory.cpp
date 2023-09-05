

#include "vsag/factory.h"

#include <memory>
#include <stdexcept>
#include <string>

#include "index/hnsw.h"
#include "index/vamana.h"

namespace vsag {

std::shared_ptr<Index>
Factory::create(const std::string& name, nlohmann::json parameters) {
    if (parameters["dtype"] != "float32") {
        return nullptr;
    }
    if (name == "hnsw") {
        std::shared_ptr<hnswlib::SpaceInterface> space = nullptr;
        if (parameters["metric_type"] == "l2") {
            space = std::make_shared<hnswlib::L2Space>(parameters["dim"]);
        } else {
            space = std::make_shared<hnswlib::InnerProductSpace>(parameters["dim"]);
        }
        auto index = std::make_shared<HNSW>(
            space, parameters["max_elements"], parameters["M"], parameters["ef_construction"]);
        if (parameters.contains("ef_runtime")) {
            index->SetEfRuntime(parameters["ef_runtime"]);
        }
        return index;
    } else if (name == "vamana") {
        std::string dtype = "float32";
        return std::make_shared<Vamana>(diskann::Metric::FAST_L2, 1, 1, dtype);
    } else {
        // not support
        return nullptr;
    }
}

}  // namespace vsag
