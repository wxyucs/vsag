#pragma once

#include <chrono>
#include <cstdint>
#include <string>

#include "spdlog/spdlog.h"

namespace vsag {

struct SlowTaskTimer {
    SlowTaskTimer(const std::string& name, int64_t log_threshold_ms = 0);
    ~SlowTaskTimer();

    std::string name;
    int64_t threshold;
    std::chrono::steady_clock::time_point start;
};

struct Timer {
    Timer(double& ref);
    ~Timer();

    double& ref_;
    std::chrono::steady_clock::time_point start;
};

}  // namespace vsag
