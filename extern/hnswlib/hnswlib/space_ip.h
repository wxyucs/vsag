#pragma once
#include "hnswlib.h"

namespace vsag {

extern float InnerProductDistance(const void* pVect1, const void* pVect2, const void* qty_ptr);
extern float(*InnerProductDistanceSIMD16Ext)(const void *, const void *, const void *);
extern float(*InnerProductDistanceSIMD16ExtResiduals)(const void *, const void *, const void *);
extern float(*InnerProductDistanceSIMD4Ext)(const void *, const void *, const void *);
extern float(*InnerProductDistanceSIMD4ExtResiduals)(const void *, const void *, const void *);

} // namespace vsag

namespace hnswlib {
class InnerProductSpace : public SpaceInterface {
    DISTFUNC fstdistfunc_;
    size_t data_size_;
    size_t dim_;

 public:
    InnerProductSpace(size_t dim) {
        if (dim % 16 == 0)
            fstdistfunc_ = vsag::InnerProductDistanceSIMD16Ext;
        else if (dim % 4 == 0)
            fstdistfunc_ = vsag::InnerProductDistanceSIMD4Ext;
        else if (dim > 16)
            fstdistfunc_ = vsag::InnerProductDistanceSIMD16ExtResiduals;
        else if (dim > 4)
            fstdistfunc_ = vsag::InnerProductDistanceSIMD4ExtResiduals;
        else
            fstdistfunc_ = vsag::InnerProductDistance;
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

    ~InnerProductSpace() {}
};

}  // namespace hnswlib
