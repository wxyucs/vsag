//
// Created by inabao on 2023/7/31.
//


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "vsag/vsag.h"

namespace py = pybind11;


int add(int a, int b) {
    return a + b;
}
py::array_t<float> kmeans(py::array_t<float, py::array::c_style | py::array::forcecast>& datas, int clusters, const std::string& dis_type) {
    auto data_shape = datas.shape();
    py::ssize_t py_clusters(clusters);
    auto data_size = data_shape[0];
    auto dimension = data_shape[1];
    auto centroids = py::array_t<float>(py::array::ShapeContainer{py_clusters, dimension});
    vsag::kmeans_clustering(dimension, data_size, clusters, datas.data(), centroids.mutable_data(), dis_type);
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
        if (dist_type == "l2" && data_type == "float32") {
            space = std::make_shared<hnswlib::L2Space>(dim);
        } else if (dist_type == "ip" && data_type == "int8") {
            space = std::make_shared<hnswlib::InnerProductSpaceInt8>(dim);
        } else if (dist_type == "ip" && data_type == "float") {
            space = std::make_shared<hnswlib::InnerProductSpace>(dim);
        } else {
            throw std::runtime_error("no matching space");
        }
        hnsw = std::make_shared<vsag::HNSW>(space, max_elements, M, ef_construction, ef_runtime);
    }
    void
    addPoint(py::array_t<float> point, size_t label) {
        hnsw->addPoint(point.data(), label);
    }


    void
    setEfRuntime(size_t ef_runtime) {
        hnsw->setEfRuntime(ef_runtime);
    }

    py::object
    searchTopK(py::array_t<float> point, size_t k) {
        auto result = hnsw->searchTopK(point.data(), k);
        auto labels = py::array_t<size_t>(k);
        auto dists = py::array_t<float>(k);
        auto labels_data = labels.mutable_data();
        auto dists_data = dists.mutable_data();
        for (int i = 0; i < k; ++i) {
            auto item = result.top();
            labels_data[i] = item.second;
            dists_data[i] = item.first;
            result.pop();
        }
        return py::make_tuple(labels, dists);
    }
private:
    std::shared_ptr<vsag::HNSW> hnsw;
    std::shared_ptr<hnswlib::SpaceInterface> space;
};


class VamanaIndex {
public:
    VamanaIndex(int dim,
                int max_elements,
                const std::string& dist_type,
                const std::string& data_type) {
        diskann::Metric metric;
        if (dist_type == "l2") {
            metric = diskann::Metric::L2;
        } else if (dist_type == "cosine") {
            metric = diskann::Metric::COSINE;
        } else if (dist_type == "ip") {
            metric = diskann::Metric::INNER_PRODUCT;
        }
        vamana = std::make_shared<vsag::Vamana>(metric, dim, max_elements, data_type);
    }

    void build(
            py::array_t<float> datas,
            int M,
            int ef_construction) {
        vamana->build(datas.data(), ef_construction, M);
    }

    py::object
    searchTopK(py::array_t<float> point, size_t k) {
        auto result = vamana->searchTopK(point.data(), k);
        auto labels = py::array_t<uint32_t>(k);
        auto dists = py::array_t<float>(k);
        auto labels_data = labels.mutable_data();
        auto dists_data = dists.mutable_data();
        for (int i = 0; i < k; ++i) {
            auto item = result.top();
            labels_data[i] = item.second;
            dists_data[i] = item.first;
            result.pop();
        }
        return py::make_tuple(labels, dists);
    }

    void
    setEfRuntime(size_t ef_runtime) {
        vamana->setEfRuntime(ef_runtime);
    }

private:
    std::shared_ptr<vsag::Vamana> vamana;
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
        .def("addPoint", &HNSWIndex::addPoint,
             py::arg("point"),
             py::arg("label"))
        .def("searchTopK", &HNSWIndex::searchTopK,
             py::arg("point"),
             py::arg("k"))
        .def("setEfRuntime", &HNSWIndex::setEfRuntime,
             py::arg("ef_runtime"));

    py::class_<VamanaIndex>(m, "VamanaIndex")
            .def(py::init<int, int, const std::string&, const std::string&>(),
                 py::arg("dim"),
                 py::arg("max_elements"),
                 py::arg("dist_type"),
                 py::arg("data_type"))
            .def("searchTopK", &VamanaIndex::searchTopK,
                 py::arg("point"),
                 py::arg("k"))
            .def("build",&VamanaIndex::build,
                 py::arg("datas"),
                 py::arg("M"),
                 py::arg("ef_construction"))
            .def("setEfRuntime", &VamanaIndex::setEfRuntime,
                 py::arg("ef_runtime"));
}
