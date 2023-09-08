//
// Created by inabao on 2023/7/31.
//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include <nlohmann/json.hpp>
#include "vsag/vsag.h"
#include "iostream"
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
        hnsw_ = vsag::Factory::create("hnsw", index_parameters);
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


class DiskAnnIndex {
public:
    DiskAnnIndex() {

    }

    void
    build(py::array_t<float> datas, int max_elements, int dim, int ef_construction, std::string dist_type,  std::string data_type, int R, float p_val, size_t disk_pq_dims) {

        nlohmann::json index_parameters{
                {"dtype", data_type},
                {"metric_type", dist_type},
                {"dim", dim},
                {"max_elements", max_elements},
                {"L", ef_construction},
                {"R", R},
                {"p_val", p_val},
                {"disk_pq_dims", disk_pq_dims}
        };
        diskann_ = vsag::Factory::create("diskann", index_parameters);
        vsag::Dataset dataset;
        long ids[max_elements];
        for (int i = 0; i < max_elements; ++i) {
            ids[i] = i;
        }
        dataset.SetOwner(false);
        dataset.SetDim(dim);
        dataset.SetNumElements(max_elements);
        dataset.SetIds(ids);
        dataset.SetFloat32Vectors(datas.mutable_data());
        diskann_->Build(dataset);
    }

    py::object
    searchTopK(py::array_t<float> point, size_t k, size_t ef_search, size_t beam_search, size_t io_limit) {
        nlohmann::json index_parameters{
            {"ef_search", ef_search},
            {"beam_search", beam_search},
            {"io_limit", io_limit}
        };
        vsag::Dataset query;
        
        query.SetNumElements(1);
        query.SetDim(point.size());
        query.SetFloat32Vectors(point.mutable_data());
        query.SetOwner(false);
        auto result = diskann_->KnnSearch(query, k, index_parameters);
        auto labels = py::array_t<uint32_t>(k);
        auto dists = py::array_t<float>(k);
        auto labels_data = labels.mutable_data();
        auto dists_data = dists.mutable_data();
        auto ids = result.GetIds();
        auto distances = result.GetDistances();
        for (int i = 0; i < k; ++i) {
            labels_data[i] = static_cast<uint32_t>(ids[i]);
            dists_data[i] = distances[i];
        }

        return py::make_tuple(labels, dists);
    }


private:
    std::shared_ptr<vsag::Index> diskann_;
};


class Index {
public:
    Index(std::string index_name, nlohmann::json &index_parameters) {
        index = vsag::Factory::create(index_name, index_parameters);
    }
    void
    build(py::array_t<float> datas, int max_elements, int dim) {
        vsag::Dataset dataset;
        long ids[max_elements];
        for (int i = 0; i < max_elements; ++i) {
            ids[i] = i;
        }
        dataset.SetOwner(false);
        dataset.SetDim(dim);
        dataset.SetNumElements(max_elements);
        dataset.SetIds(ids);
        dataset.SetFloat32Vectors(datas.mutable_data());
        index->Build(dataset);
    }

    py::object
    searchTopK(py::array_t<float> point, size_t k, nlohmann::json search_parameters) {
        vsag::Dataset query;
        int data_num = search_parameters["data_num"]; 
        query.SetNumElements(data_num);
        query.SetDim(point.size());
        query.SetFloat32Vectors(point.mutable_data());
        query.SetOwner(false);

        auto result = index->KnnSearch(query, k, search_parameters);
        
        auto labels = py::array_t<int64_t>(k);
        auto dists = py::array_t<float>(k);
        auto labels_data = labels.mutable_data();
        auto dists_data = dists.mutable_data();
        auto ids = result.GetIds();
        auto distances = result.GetDistances();
        for (int i = 0; i < data_num * k; ++i) {
            labels_data[i] = ids[i];
            dists_data[i] = distances[i];
        }
        return py::make_tuple(labels, dists); 
    }

private:
    std::shared_ptr<vsag::Index> index;
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

    py::class_<DiskAnnIndex>(m, "DiskAnnIndex")
            .def(py::init<>())
            .def("searchTopK", &DiskAnnIndex::searchTopK,
             py::arg("point"), 
             py::arg("k"), 
             py::arg("ef_search"), 
             py::arg("beam_search"), 
             py::arg("io_limit"))
            .def("build",
                 &DiskAnnIndex::build,
                 py::arg("datas"),
                 py::arg("max_elements"),
                 py::arg("dim"),
                 py::arg("ef_construction"),
                 py::arg("dist_type"),
                 py::arg("data_type"),
                 py::arg("R"),
                 py::arg("p_val"),
                 py::arg("disk_pq_dims"));

    py::class_<Index>(m, "Index")
        .def(py::init<std::string, nlohmann::json&>(),
                py::arg("index_name"), 
                py::arg("index_parameters"))
        .def("searchTopK", &Index::searchTopK,
                py::arg("point"), 
                py::arg("k"), 
                py::arg("search_parameters"))
        .def("build", &Index::build,
                py::arg("datas"),
                py::arg("max_elements"),
                py::arg("dim"));
}
