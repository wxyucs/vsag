// Copyright 2026-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <atomic>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

#include "impl/allocator/safe_allocator.h"
#include "quantization/scalar_quantization/scalar_quantizer.h"
#include "quantization/scalar_quantization/sq8_per_vector_quantizer.h"

namespace {
template <typename T>
std::vector<T>
read_binary(const std::string& path, uint64_t count) {
    std::vector<T> data(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
    if (!stream || stream.peek() != EOF) {
        throw std::runtime_error("invalid input size: " + path);
    }
    return data;
}

// Exhaustive ADC is an accuracy reference, not a throughput experiment. Ties use
// increasing IDs. Approximate search can occasionally beat this recall by missing
// a quantization-induced false positive, so this is not a strict mathematical bound.
template <typename Quantizer>
void
measure_reference(const std::string& dir, const std::string& type) {
    constexpr uint64_t n = 1000000;
    constexpr uint64_t nq = 1000;
    constexpr uint64_t dim = 960;
    constexpr uint64_t k = 10;
    constexpr uint64_t threads = 16;
    auto base = read_binary<float>(dir + "/train.f32", n * dim);
    auto queries = read_binary<float>(dir + "/test.f32", nq * dim);
    auto truth = read_binary<int64_t>(dir + "/neighbors.i64", nq * k);
    auto allocator = vsag::SafeAllocator::FactoryDefaultAllocator();
    Quantizer quantizer(dim, allocator.get());
    if (!quantizer.Train(base.data(), n)) {
        throw std::runtime_error("training failed");
    }
    const auto stride = quantizer.GetCodeSize();
    std::vector<uint8_t> codes(n * stride);
    if (!quantizer.EncodeBatch(base.data(), codes.data(), n)) {
        throw std::runtime_error("encoding failed");
    }
    base.clear();
    base.shrink_to_fit();
    std::atomic<uint64_t> next{0};
    std::vector<uint64_t> hits(nq, 0);
    std::vector<std::thread> workers;
    for (uint64_t worker = 0; worker < threads; ++worker) {
        workers.emplace_back([&]() {
            auto computer = quantizer.FactoryComputer();
            uint64_t q;
            while ((q = next.fetch_add(1)) < nq) {
                computer->SetQuery(queries.data() + q * dim);
                std::priority_queue<std::pair<float, int64_t>> nearest;
                for (uint64_t id = 0; id < n; ++id) {
                    float distance;
                    computer->ComputeDist(codes.data() + id * stride, &distance);
                    const auto entry = std::make_pair(distance, static_cast<int64_t>(id));
                    if (nearest.size() < k) {
                        nearest.push(entry);
                    } else if (entry < nearest.top()) {
                        nearest.pop();
                        nearest.push(entry);
                    }
                }
                while (!nearest.empty()) {
                    const auto id = nearest.top().second;
                    nearest.pop();
                    for (uint64_t t = 0; t < k; ++t) {
                        if (id == truth[q * k + t]) {
                            ++hits[q];
                            break;
                        }
                    }
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }
    const auto total = std::accumulate(hits.begin(), hits.end(), uint64_t{0});
    std::ofstream output(dir + "/" + type + "-adc-reference.json");
    output << std::setprecision(12) << R"({"quantizer":")" << type << R"(","base_count":)" << n
           << ",\"query_count\":" << nq << ",\"threads\":" << threads << ",\"hits\":" << total
           << ",\"recall\":" << static_cast<double>(total) / (nq * k)
           << ",\"method\":\"exhaustive FP32-query ADC; ties by ID; no QPS claim\"}\n";
    std::ofstream per_query(dir + "/" + type + "-adc-reference.csv");
    per_query << "query,hits\n";
    for (uint64_t q = 0; q < nq; ++q) {
        per_query << q << ',' << hits[q] << '\n';
    }
}
}  // namespace

int
main(int argc, char** argv) {
    try {
        if (argc != 3) {
            throw std::runtime_error("usage: sq8_adc_reference DATA_DIR TYPE");
        }
        const std::string type = argv[2];
        if (type == "sq8") {
            measure_reference<vsag::SQ8Quantizer<vsag::MetricType::METRIC_TYPE_L2SQR>>(argv[1],
                                                                                       type);
        } else if (type == "sq8_per_vector") {
            measure_reference<vsag::SQ8PerVectorQuantizer>(argv[1], type);
        } else {
            throw std::runtime_error("unsupported quantizer");
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
