#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

#include "spdlog/spdlog.h"

namespace vsag {

const static int64_t DEFAULT_WATCH_WINDOW_SIZE = 20;

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
    WindowResultQueue() {
        window_size_ = DEFAULT_WATCH_WINDOW_SIZE;
        queue.resize(window_size_);
    }
    WindowResultQueue(size_t window_size) : window_size_(window_size) {
        if (window_size_ < 0) {
            window_size_ = DEFAULT_WATCH_WINDOW_SIZE;
        }
        queue.resize(window_size);
    }

    void
    push(float value) {
        queue.data()[count_ % window_size_] = value;
        count_++;
    }

    void
    resizeWindowSize(size_t new_window_size_) {
        if (new_window_size_ > window_size_) {
            if (count_ > window_size_) {
                count_ = window_size_;
            }
            window_size_ = new_window_size_;
            queue.resize(new_window_size_);
        }
    }

    float
    getAvgResult() const {
        int64_t statstic_num = std::min(count_, window_size_);
        float result = 0;
        for (int i = 0; i < statstic_num; i++) {
            result += queue.data()[i];
        }
        return result / statstic_num;
    }

private:
    int64_t window_size_;
    int64_t count_;
    std::vector<float> queue;
};

}  // namespace vsag
