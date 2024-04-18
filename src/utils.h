#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

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

class WindowResultQueue {
public:
    WindowResultQueue();

    WindowResultQueue(size_t window_size);

    void
    Push(float value);

    size_t
    ResizeWindowSize(size_t new_window_size_);

    float
    GetAvgResult() const;

private:
    size_t count_ = 0;
    std::vector<float> queue_;
};

template <typename T>
struct Number {
    Number(T n) : num(n) {
    }

    bool
    in_range(T lower, T upper) {
        return ((unsigned)(num - lower) <= (upper - lower));
    }

    T num;
};

}  // namespace vsag
