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


PYBIND11_MODULE(pyvsag, m) {
    m.def("add", &add, "A function which adds two numbers");
    m.def("kmeans", &kmeans, "Kmeans");
}
