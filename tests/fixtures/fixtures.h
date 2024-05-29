#pragma once

#include <cstdint>
#include <filesystem>
#include <random>
#include <tuple>
#include <vector>

#include "vsag/vsag.h"

namespace fixtures {

std::vector<float>
generate_vectors(int64_t num_vectors, int64_t dim);

std::tuple<std::vector<int64_t>, std::vector<float>>
generate_ids_and_vectors(int64_t num_elements, int64_t dim);

vsag::IndexPtr
generate_index(const std::string& name,
               const std::string& metric_type,
               int64_t num_vectors,
               int64_t dim,
               std::vector<int64_t>& ids,
               std::vector<float>& vectors);

float
test_knn_recall(const vsag::IndexPtr& index,
                const std::string& search_parameters,
                int64_t num_vectors,
                int64_t dim,
                std::vector<int64_t>& ids,
                std::vector<float>& vectors);

std::string
generate_hnsw_build_parameters_string(const std::string& metric_type, int64_t dim);

vsag::Dataset
brute_force(const vsag::Dataset& query,
            const vsag::Dataset& base,
            int64_t k,
            const std::string& metric_type);

struct temp_dir {
    temp_dir(const std::string& name) {
        auto epoch_time = std::chrono::system_clock::now().time_since_epoch();
        auto seconds = std::chrono::duration_cast<std::chrono::seconds>(epoch_time).count();

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dist(1000, 9999);
        int random_number = dist(gen);

        std::stringstream dirname;
        dirname << "vsagtest_" << std::setfill('0') << std::setw(14) << seconds << "_"
                << std::to_string(random_number);
        path = "/tmp/" + dirname.str() + "/";
        std::filesystem::create_directory(path);
    }

    ~temp_dir() {
        std::filesystem::remove_all(path);
    }

    std::string path;
};

struct dist_t {
    dist_t(float val) {
        this->value = val;
    }

    bool
    operator==(const dist_t& d) const {
        return std::fabs(this->value - d.value) < epsilon;
    }

    friend std::ostream&
    operator<<(std::ostream& os, const dist_t& obj) {
        os << obj.value;
        return os;
    }

    float value;
    const float epsilon = 1e-6;
};

}  // Namespace fixtures
