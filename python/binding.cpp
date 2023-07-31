//
// Created by inabao on 2023/7/31.
//


#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

int add(int a, int b) {
    return a + b;
}




PYBIND11_MODULE(pyVsag, m) {
    m.def("add", &add, "A function which adds two numbers");
}
