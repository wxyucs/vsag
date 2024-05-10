#include "./utils.h"

#include <cstdint>
#include <unordered_set>

#include "./logger.h"

namespace vsag {

const static int64_t DEFAULT_WATCH_WINDOW_SIZE = 20;

SlowTaskTimer::SlowTaskTimer(const std::string& n, int64_t log_threshold_ms)
    : name(n), threshold(log_threshold_ms) {
    start = std::chrono::steady_clock::now();
}

SlowTaskTimer::~SlowTaskTimer() {
    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = finish - start;
    if (duration.count() > threshold) {
        if (duration.count() >= 1000) {
            logger::info("{0} cost {1:.3f}s", name, duration.count() / 1000);
        } else {
            logger::info("{0} cost {1:.3f}ms", name, duration.count());
        }
    }
}

Timer::Timer(double& ref) : ref_(ref) {
    start = std::chrono::steady_clock::now();
}

Timer::~Timer() {
    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = finish - start;
    ref_ = duration.count();
}

WindowResultQueue::WindowResultQueue() {
    queue_.resize(DEFAULT_WATCH_WINDOW_SIZE);
}

WindowResultQueue::WindowResultQueue(size_t window_size) {
    queue_.resize(window_size > 0 ? window_size : DEFAULT_WATCH_WINDOW_SIZE);
}

void
WindowResultQueue::Push(float value) {
    size_t window_size = queue_.size();
    queue_[count_ % window_size] = value;
    count_++;
}

size_t
WindowResultQueue::ResizeWindowSize(size_t new_window_size) {
    if (new_window_size > queue_.size()) {
        count_ = std::min(count_, queue_.size());
        queue_.resize(new_window_size);
    }
    return queue_.size();
}

float
WindowResultQueue::GetAvgResult() const {
    size_t statstic_num = std::min(count_, queue_.size());
    float result = 0;
    for (int i = 0; i < statstic_num; i++) {
        result += queue_[i];
    }
    return result / statstic_num;
}

}  // namespace vsag
