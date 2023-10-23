//
// Created by inabao on 2023/7/31.
//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "iostream"
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

class Index {
public:
    Index(std::string index_name, const std::string& index_parameters) {
        index = vsag::Factory::CreateIndex(index_name, index_parameters);
    }
    void
    build(py::array_t<float> datas, py::array_t<int64_t> ids, size_t max_elements, size_t dim) {
        vsag::Dataset dataset;
        dataset.SetOwner(false);
        dataset.SetDim(dim);
        dataset.SetNumElements(max_elements);
        dataset.SetIds(ids.mutable_data());
        dataset.SetFloat32Vectors(datas.mutable_data());
        index->Build(dataset);
    }

    py::object
    KnnSearch(py::array_t<float> point, size_t k, std::string& search_parameters) {
        vsag::Dataset query;
        size_t data_num = 1;
        query.SetNumElements(data_num);
        query.SetDim(point.size());
        query.SetFloat32Vectors(point.mutable_data());
        query.SetOwner(false);

        auto labels = py::array_t<int64_t>(k);
        auto dists = py::array_t<float>(k);
        if (auto result = index->KnnSearch(query, k, search_parameters); result.has_value()) {
            auto labels_data = labels.mutable_data();
            auto dists_data = dists.mutable_data();
            auto ids = result->GetIds();
            auto distances = result->GetDistances();
            for (int i = 0; i < data_num * k; ++i) {
                labels_data[i] = ids[i];
                dists_data[i] = distances[i];
            }
        }

        return py::make_tuple(labels, dists);
    }

    py::object
    RangeSearch(py::array_t<float> point, float threshold, std::string& search_parameters) {
        vsag::Dataset query;
        size_t data_num = 1;
        query.SetNumElements(data_num);
        query.SetDim(point.size());
        query.SetFloat32Vectors(point.mutable_data());
        query.SetOwner(false);

        py::array_t<int64_t> labels;
        py::array_t<float> dists;
        if (auto result = index->RangeSearch(query, threshold, search_parameters);
            result.has_value()) {
            auto ids = result->GetIds();
            auto distances = result->GetDistances();
            auto k = result->GetDim();
            labels.resize({k});
            dists.resize({k});
            auto labels_data = labels.mutable_data();
            auto dists_data = dists.mutable_data();
            for (int i = 0; i < data_num * k; ++i) {
                labels_data[i] = ids[i];
                dists_data[i] = distances[i];
            }
        }

        return py::make_tuple(labels, dists);
    }

private:
    std::shared_ptr<vsag::Index> index;
};

PYBIND11_MODULE(pyvsag, m) {
    m.def("add", &add, "A function which adds two numbers");
    m.def("kmeans", &kmeans, "Kmeans");
    py::class_<Index>(m, "Index")
        .def(py::init<std::string, std::string&>(),
             py::arg("index_name"),
             py::arg("index_parameters"))
        .def("KnnSearch",
             &Index::KnnSearch,
             py::arg("point"),
             py::arg("k"),
             py::arg("search_parameters"))
        .def("RangeSearch",
             &Index::RangeSearch,
             py::arg("point"),
             py::arg("threshold"),
             py::arg("search_parameters"))
        .def("build",
             &Index::build,
             py::arg("datas"),
             py::arg("ids"),
             py::arg("max_elements"),
             py::arg("dim"));
}
