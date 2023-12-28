#pragma once
#include "hnswlib.h"

namespace vsag {

extern float L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr);
extern float(*L2SqrSIMD16Ext)(const void *, const void *, const void *);
extern float(*L2SqrSIMD16ExtResiduals)(const void *, const void *, const void *);
extern float(*L2SqrSIMD4Ext)(const void *, const void *, const void *);
extern float(*L2SqrSIMD4ExtResiduals)(const void *, const void *, const void *);

}  // namespace vsag

namespace hnswlib {

class L2Space : public SpaceInterface {
    DISTFUNC fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    L2Space(size_t dim) {
        if (dim % 16 == 0)
            fstdistfunc_ = vsag::L2SqrSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = vsag::L2SqrSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = vsag::L2SqrSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = vsag::L2SqrSIMD4ExtResiduals;
        else
            fstdistfunc_ = vsag::L2Sqr;
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
