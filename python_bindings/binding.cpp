//
// Created by inabao on 2023/7/31.
//

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "iostream"
#include "vsag/vsag.h"

namespace py = pybind11;

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
    Index(std::string name, const std::string& parameters) {
        if (auto index = vsag::Factory::CreateIndex(name, parameters)) {
            index_ = index.value();
        } else {
            vsag::Error error_code = index.error();
            if (error_code.type == vsag::ErrorType::UNSUPPORTED_INDEX) {
                throw std::runtime_error("error type: UNSUPPORTED_INDEX");
            } else if (error_code.type == vsag::ErrorType::INVALID_ARGUMENT) {
                throw std::runtime_error("error type: invalid_parameter");
            } else {
                throw std::runtime_error("error type: unexpectedError");
            }
        }
    }

public:
    void
    Build(py::array_t<float> vectors, py::array_t<int64_t> ids, size_t num_elements, size_t dim) {
        vsag::Dataset dataset;
        dataset.Owner(false)
            .Dim(dim)
            .NumElements(num_elements)
            .Ids(ids.mutable_data())
            .Float32Vectors(vectors.mutable_data());
        index_->Build(dataset);
    }

    py::object
    KnnSearch(py::array_t<float> vector, size_t k, std::string& parameters) {
        vsag::Dataset query;
        size_t data_num = 1;
        query.NumElements(data_num)
            .Dim(vector.size())
            .Float32Vectors(vector.mutable_data())
            .Owner(false);

        auto labels = py::array_t<int64_t>(k);
        auto dists = py::array_t<float>(k);
        if (auto result = index_->KnnSearch(query, k, parameters); result.has_value()) {
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
    RangeSearch(py::array_t<float> point, float threshold, std::string& parameters) {
        vsag::Dataset query;
        size_t data_num = 1;
        query.NumElements(data_num)
            .Dim(point.size())
            .Float32Vectors(point.mutable_data())
            .Owner(false);

        py::array_t<int64_t> labels;
        py::array_t<float> dists;
        if (auto result = index_->RangeSearch(query, threshold, parameters); result.has_value()) {
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
    std::shared_ptr<vsag::Index> index_;
};

PYBIND11_MODULE(pyvsag, m) {
    m.def("kmeans", &kmeans, "Kmeans");
    py::class_<Index>(m, "Index")
        .def(py::init<std::string, std::string&>(), py::arg("name"), py::arg("parameters"))
        .def("build",
             &Index::Build,
             py::arg("vectors"),
             py::arg("ids"),
             py::arg("num_elements"),
             py::arg("dim"))
        .def(
            "knn_search", &Index::KnnSearch, py::arg("vector"), py::arg("k"), py::arg("parameters"))
        .def("range_search",
             &Index::RangeSearch,
             py::arg("vector"),
             py::arg("threshold"),
             py::arg("parameters"));
}
