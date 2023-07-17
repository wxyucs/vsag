#pragma once

#include <chrono>
#include <iostream>
#include <string>

struct time_recorder {
public:
    time_recorder(int &var) : duration_(var) {
	start_ = std::chrono::high_resolution_clock::now();
    }

    ~time_recorder() {
	auto stop = std::chrono::high_resolution_clock::now();
	duration_ = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start_).count();

    }
private:
    std::chrono::time_point<std::chrono::system_clock> start_;
    int &duration_;
};

inline void
print_process(int i, int total, std::string prefix_string) {
    int one_percent = total / 100;
    if (i % one_percent == 0) {
	std::cout << "\r" << prefix_string << i / one_percent << "%" << std::flush;
    }
}
