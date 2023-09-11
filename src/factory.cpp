

#include "vsag/factory.h"

#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "index/hnsw.h"
#include "index/vamana.h"
#include "index/diskann.h"
namespace vsag {

std::shared_ptr<Index>
Factory::create(const std::string& name, const std::string& parameters) {
    nlohmann::json params = nlohmann::json::parse(parameters);

    if (params["dtype"] != "float32") {
        return nullptr;
    }
    if (name == "hnsw") {
        std::shared_ptr<hnswlib::SpaceInterface> space = nullptr;
        if (params["metric_type"] == "l2") {
            space = std::make_shared<hnswlib::L2Space>(params["dim"]);
        } else {
            space = std::make_shared<hnswlib::InnerProductSpace>(params["dim"]);
        }
        auto index = std::make_shared<HNSW>(
            space, params["max_elements"], params["M"], params["ef_construction"]);
        if (params.contains("ef_runtime")) {
            index->SetEfRuntime(params["ef_runtime"]);
        }
        return index;
    } else if (name == "vamana") {
        std::string dtype = "float32";
        return std::make_shared<Vamana>(diskann::Metric::FAST_L2, 1, 1, dtype);
    } else if (name == "diskann") {
        std::string dtype = "float32";
        auto index = std::make_shared<DiskANN>(diskann::Metric::L2,
                                               dtype,
                                               params["L"],
                                               params["R"],
                                               params["p_val"],
                                               params["disk_pq_dims"]);
        return index;
    } else {
        // not support
        return nullptr;
    }
}

}  // namespace vsag
