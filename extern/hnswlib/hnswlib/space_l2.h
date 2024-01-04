#pragma once
#include "hnswlib.h"

namespace vsag {

extern hnswlib::DISTFUNC  GetL2DistanceFunc(size_t dim);

}  // namespace vsag

namespace hnswlib {

class L2Space : public SpaceInterface {
    DISTFUNC fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2Space(size_t dim) {
        fstdistfunc_ = vsag::GetL2DistanceFunc(dim);
        dim_ = dim;
        data_size_ = dim * sizeof(float);
    }

    size_t get_data_size() override {
        return data_size_;
    }

    DISTFUNC get_dist_func() override {
        return fstdistfunc_;
    }

    void *get_dist_func_param() override {
        return &dim_;
    }

    ~L2Space() {}
};

}  // namespace hnswlib
