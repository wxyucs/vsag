#include "vsag/vsag.h"

#include <cpuinfo.h>

#include <sstream>

#include "logger.h"
#include "simd/simd.h"
#include "version.h"

namespace vsag {

std::string
version() {
    return VSAG_VERSION;
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
    logger::debug(ss.str());

    setup_simd();

    return true;
}

static bool _init = init();

}  // namespace vsag
