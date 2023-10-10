#include "./utils.h"

#include <cstdint>

namespace vsag {

SlowTaskTimer::SlowTaskTimer(const std::string& n, int64_t log_threshold_ms)
    : name(n), threshold(log_threshold_ms) {
    start = std::chrono::steady_clock::now();
}

SlowTaskTimer::~SlowTaskTimer() {
    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = finish - start;
    if (duration.count() > threshold) {
        if (duration.count() >= 1000) {
            spdlog::info("{0} cost {1:.3f}s", name, duration.count() / 1000);
        } else {
            spdlog::info("{0} cost {1:.3f}ms", name, duration.count());
        }
    }
}

}  // namespace vsag
