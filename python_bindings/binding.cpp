//
// Created by inabao on 2023/7/31.
//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <nlohmann/json.hpp>

#include "vsag/vsag.h"

namespace py = pybind11;

int
add(int a, int b) {
    return a + b;
}
py::array_t<float>
kmeans(py::array_t<float, py::array::c_style | py::array::forcecast>& datas,
       int clusters,
       const std::string& dis_type) {
    auto data_shape = datas.shape();
    py::ssize_t py_clusters(clusters);
    auto data_size = data_shape[0];
    auto dimension = data_shape[1];
    auto centroids = py::array_t<float>(py::array::ShapeContainer{py_clusters, dimension});
    vsag::kmeans_clustering(
        dimension, data_size, clusters, datas.data(), centroids.mutable_data(), dis_type);
    return centroids;
}

class HNSWIndex {
public:
    HNSWIndex(int dim,
              int max_elements,
              const std::string& dist_type,
              const std::string& data_type,
              int M,
              int ef_construction,
              int ef_runtime) {
        nlohmann::json index_parameters{
            {"dtype", data_type},
            {"metric_type", dist_type},
            {"dim", dim},
            {"max_elements", max_elements},
            {"M", M},
            {"ef_construction", ef_construction},
            {"ef_runtime", ef_runtime},
        };
        hnsw_ = vsag::Factory::create("hnsw", index_parameters.dump());
    }
    void
    addPoint(py::array_t<float> point, size_t label) {
	throw std::runtime_error("not support yet");
    }

    void
    setEfRuntime(size_t ef_runtime) {
	throw std::runtime_error("not support yet");
    }

    py::object
    searchTopK(py::array_t<float> point, size_t k) {
	throw std::runtime_error("not support yet");
        // auto result = hnsw->searchTopK(point.data(), k);
        // auto labels = py::array_t<size_t>(k);
        // auto dists = py::array_t<float>(k);
        // auto labels_data = labels.mutable_data();
        // auto dists_data = dists.mutable_data();
        // for (int i = 0; i < k; ++i) {
        //     auto item = result.top();
        //     labels_data[i] = item.second;
        //     dists_data[i] = item.first;
        //     result.pop();
        // }
        // return py::make_tuple(labels, dists);
    }

private:
    std::shared_ptr<vsag::Index> hnsw_;
};

class VamanaIndex {
public:
    VamanaIndex(int dim,
                int max_elements,
                const std::string& dist_type,
                const std::string& data_type) {
	nlohmann::json index_parameters{
            {"dtype", data_type},
            {"metric_type", dist_type},
            {"dim", dim},
            {"max_elements", max_elements},
        };

        vamana_ = vsag::Factory::create("vamana", index_parameters);
    }

    void
    build(py::array_t<float> datas, int M, int ef_construction) {
	throw std::runtime_error("not support yet");
    }

    py::object
    searchTopK(py::array_t<float> point, size_t k) {
	throw std::runtime_error("not support yet");
        // auto result = vamana->searchTopK(point.data(), k);
        // auto labels = py::array_t<uint32_t>(k);
        // auto dists = py::array_t<float>(k);
        // auto labels_data = labels.mutable_data();
        // auto dists_data = dists.mutable_data();
        // for (int i = 0; i < k; ++i) {
        //     auto item = result.top();
        //     labels_data[i] = item.second;
        //     dists_data[i] = item.first;
        //     result.pop();
        // }
        // return py::make_tuple(labels, dists);
    }

    void
    setEfRuntime(size_t ef_runtime) {
	throw std::runtime_error("not support yet");
    }

private:
    std::shared_ptr<vsag::Index> vamana_;
};

PYBIND11_MODULE(pyvsag, m) {
    m.def("add", &add, "A function which adds two numbers");
    m.def("kmeans", &kmeans, "Kmeans");

    py::class_<HNSWIndex>(m, "HNSWIndex")
        .def(py::init<int, int, const std::string&, const std::string&, int, int, int>(),
             py::arg("dim"),
             py::arg("max_elements"),
             py::arg("dist_type"),
             py::arg("data_type"),
             py::arg("M") = 16,
             py::arg("ef_construction") = 200,
             py::arg("ef_runtime") = 200)
        .def("addPoint", &HNSWIndex::addPoint, py::arg("point"), py::arg("label"))
        .def("searchTopK", &HNSWIndex::searchTopK, py::arg("point"), py::arg("k"))
        .def("setEfRuntime", &HNSWIndex::setEfRuntime, py::arg("ef_runtime"));

    py::class_<VamanaIndex>(m, "VamanaIndex")
        .def(py::init<int, int, const std::string&, const std::string&>(),
             py::arg("dim"),
             py::arg("max_elements"),
             py::arg("dist_type"),
             py::arg("data_type"))
        .def("searchTopK", &VamanaIndex::searchTopK, py::arg("point"), py::arg("k"))
        .def("build",
             &VamanaIndex::build,
             py::arg("datas"),
             py::arg("M"),
             py::arg("ef_construction"))
        .def("setEfRuntime", &VamanaIndex::setEfRuntime, py::arg("ef_runtime"));
}
