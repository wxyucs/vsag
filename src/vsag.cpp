#include "vsag/vsag.h"

#include <cpuinfo.h>
#include <spdlog/spdlog.h>

#include <sstream>

#include "simd/simd.h"
#include "version.h"

namespace vsag {

std::string
version() {
    return VSAG_VERSION;
}

float (*L2SqrSIMD16Ext)(const void*, const void*, const void*);
float (*L2SqrSIMD16ExtResiduals)(const void*, const void*, const void*);
float (*L2SqrSIMD4Ext)(const void*, const void*, const void*);
float (*L2SqrSIMD4ExtResiduals)(const void*, const void*, const void*);

float (*InnerProductSIMD4Ext)(const void*, const void*, const void*);
float (*InnerProductSIMD16Ext)(const void*, const void*, const void*);
float (*InnerProductDistanceSIMD16Ext)(const void*, const void*, const void*);
float (*InnerProductDistanceSIMD16ExtResiduals)(const void*, const void*, const void*);
float (*InnerProductDistanceSIMD4Ext)(const void*, const void*, const void*);
float (*InnerProductDistanceSIMD4ExtResiduals)(const void*, const void*, const void*);

void
setup_simd() {
    L2SqrSIMD16Ext = L2Sqr;
    L2SqrSIMD16ExtResiduals = L2Sqr;
    L2SqrSIMD4Ext = L2Sqr;
    L2SqrSIMD4ExtResiduals = L2Sqr;

    InnerProductSIMD4Ext = InnerProduct;
    InnerProductSIMD16Ext = InnerProduct;
    InnerProductDistanceSIMD16Ext = InnerProductDistance;
    InnerProductDistanceSIMD16ExtResiduals = InnerProductDistance;
    InnerProductDistanceSIMD4Ext = InnerProductDistance;
    InnerProductDistanceSIMD4ExtResiduals = InnerProductDistance;

#if defined(ENABLE_SSE)
    if (cpuinfo_has_x86_sse()) {
        L2SqrSIMD16Ext = L2SqrSIMD16ExtSSE;
        L2SqrSIMD16ExtResiduals = L2SqrSIMD16ExtResidualsSSE;
        L2SqrSIMD4Ext = L2SqrSIMD4ExtSSE;
        L2SqrSIMD4ExtResiduals = L2SqrSIMD4ExtResidualsSSE;

        InnerProductSIMD4Ext = InnerProductSIMD4ExtSSE;
        InnerProductSIMD16Ext = InnerProductSIMD16ExtSSE;
        InnerProductDistanceSIMD16Ext = InnerProductDistanceSIMD16ExtSSE;
        InnerProductDistanceSIMD16ExtResiduals = InnerProductDistanceSIMD16ExtResidualsSSE;
        InnerProductDistanceSIMD4Ext = InnerProductDistanceSIMD4ExtSSE;
        InnerProductDistanceSIMD4ExtResiduals = InnerProductDistanceSIMD4ExtResidualsSSE;
    }
#endif

#if defined(ENABLE_AVX)
    if (cpuinfo_has_x86_avx()) {
        L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;

        InnerProductSIMD4Ext = InnerProductSIMD4ExtAVX;
        InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX;
    }
#endif

#if defined(ENABLE_AVX512)
    if (cpuinfo_has_x86_avx512f()) {
        L2SqrSIMD16Ext = L2SqrSIMD16ExtAVX512;

        InnerProductSIMD16Ext = InnerProductSIMD16ExtAVX512;
    }
#endif
}

bool
init() {
    cpuinfo_initialize();
    std::stringstream ss;

    ss << std::boolalpha;
    ss << "====vsag start init====";
    ss << "\nrunning on " << cpuinfo_get_package(0)->name;
    ss << "\ncores count: " << cpuinfo_get_cores_count();
    ss << "\nsse: " << cpuinfo_has_x86_sse();
    ss << "\navx: " << cpuinfo_has_x86_avx();
    ss << "\navx2: " << cpuinfo_has_x86_avx2();
    ss << "\navx512f: " << cpuinfo_has_x86_avx512f();
    ss << "\navx512dq: " << cpuinfo_has_x86_avx512dq();
    ss << "\navx512bw: " << cpuinfo_has_x86_avx512bw();
    ss << "\navx512vl: " << cpuinfo_has_x86_avx512vl();
    ss << "\n====vsag init done====";
    spdlog::info(ss.str());

    setup_simd();

    return true;
}

static bool _init = init();

}  // namespace vsag
