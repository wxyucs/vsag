// Copyright 2026-present the vsag project
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "vsag/vsag.h"

#ifdef __linux__
#include <sched.h>
#endif

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

using Clock = std::chrono::steady_clock;
double
elapsed_seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
}  // namespace

// Input files are generated and validated by sq8_per_vector.py. Loading, building,
// serialization, warmup, result validation and recall computation are outside search timing.
int
main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6) {
            throw std::runtime_error(
                "usage: sq8_per_vector_benchmark DATA_DIR TYPE CPU EF_CSV [search]");
        }
        const std::string dir = argv[1];
        const std::string type = argv[2];
        const uint64_t n = 1000000;
        const uint64_t nq = 1000;
        const uint64_t dim = 960;
        const uint64_t k = 10;
        const bool search_only = argc == 6 && std::string(argv[5]) == "search";
        auto base =
            search_only ? std::vector<float>{} : read_binary<float>(dir + "/train.f32", n * dim);
        auto queries = read_binary<float>(dir + "/test.f32", nq * dim);
        auto truth = read_binary<int64_t>(dir + "/neighbors.i64", nq * k);
        std::vector<int64_t> ids(n);
        std::iota(ids.begin(), ids.end(), 0);
        const std::string config = R"({"dtype":"float32","metric_type":"l2","dim":960,
            "index_param":{"base_quantization_type":")" +
                                   type + R"(","use_reorder":false,
            "max_degree":32,"ef_construction":200,"build_thread_count":16}})";
        std::ofstream(dir + "/" + type + "-build.json") << config << '\n';
        auto made = vsag::Factory::CreateIndex("hgraph", config);
        if (!made.has_value()) {
            throw std::runtime_error(made.error().message);
        }
        auto index = made.value();
        Clock::time_point start;
        if (search_only) {
            std::ifstream serialized(dir + "/" + type + ".index", std::ios::binary);
            auto loaded = index->Deserialize(serialized);
            if (!loaded.has_value()) {
                throw std::runtime_error(loaded.error().message);
            }
        } else {
            auto dataset = vsag::Dataset::Make()
                               ->NumElements(n)
                               ->Dim(dim)
                               ->Float32Vectors(base.data())
                               ->Ids(ids.data())
                               ->Owner(false);
            start = Clock::now();
            auto built = index->Build(dataset);
            const double build_seconds = elapsed_seconds(start);
            if (!built.has_value() || !built.value().empty()) {
                throw std::runtime_error("build failed or omitted vectors");
            }
            std::ofstream serialized(dir + "/" + type + ".index", std::ios::binary);
            auto saved = index->Serialize(serialized);
            if (!saved.has_value()) {
                throw std::runtime_error(saved.error().message);
            }
            const auto bytes = serialized.tellp();
            serialized.close();
            std::ofstream footprint(dir + "/" + type + "-footprint.json");
            footprint << "{\"build_seconds\":" << build_seconds
                      << ",\"memory_bytes\":" << index->GetMemoryUsage()
                      << ",\"serialized_bytes\":" << bytes << "}\n";
            footprint.close();
        }
        base.clear();
        base.shrink_to_fit();
#ifdef __linux__
        cpu_set_t affinity;
        CPU_ZERO(&affinity);
        CPU_SET(std::stoi(argv[3]), &affinity);
        if (sched_setaffinity(0, sizeof(affinity), &affinity) != 0) {
            throw std::runtime_error("could not pin search CPU");
        }
#endif
        std::vector<vsag::DatasetPtr> query_sets;
        for (uint64_t q = 0; q < nq; ++q) {
            query_sets.push_back(vsag::Dataset::Make()
                                     ->NumElements(1)
                                     ->Dim(dim)
                                     ->Float32Vectors(queries.data() + q * dim)
                                     ->Owner(false));
        }
        std::ofstream csv(dir + "/" + type + "-raw.csv",
                          search_only ? std::ios::app : std::ios::out);
        if (!search_only) {
            csv << "quantizer,ef_search,repeat,queries,successful,failed,seconds,qps,recall\n";
        }
        csv << std::setprecision(12);
        std::ofstream failures(dir + "/" + type + "-failures.log",
                               search_only ? std::ios::app : std::ios::out);
        std::string sweep = argv[4];
        while (!sweep.empty()) {
            const auto delimiter = sweep.find(',');
            const int ef = std::stoi(sweep.substr(0, delimiter));
            sweep = delimiter == std::string::npos ? "" : sweep.substr(delimiter + 1);
            const std::string search = R"({"hgraph":{"ef_search":)" + std::to_string(ef) + "}}";
            std::ofstream(dir + "/search-" + std::to_string(ef) + ".json") << search << '\n';
            // One complete, equal warmup pass followed by three measured complete passes.
            for (int repeat = -1; repeat < 3; ++repeat) {
                std::vector<tl::expected<vsag::DatasetPtr, vsag::Error>> results;
                results.reserve(nq);
                start = Clock::now();
                for (const auto& query : query_sets) {
                    results.push_back(index->KnnSearch(query, k, search));
                }
                const double seconds = elapsed_seconds(start);
                uint64_t successful = 0;
                uint64_t hits = 0;
                for (uint64_t q = 0; q < nq; ++q) {
                    if (!results[q].has_value() || results[q].value()->GetDim() != k) {
                        failures << "ef=" << ef << " repeat=" << repeat << " query=" << q
                                 << " error="
                                 << (results[q].has_value() ? "short result"
                                                            : results[q].error().message)
                                 << '\n';
                        continue;
                    }
                    ++successful;
                    const auto* found = results[q].value()->GetIds();
                    for (uint64_t j = 0; j < k; ++j) {
                        for (uint64_t t = 0; t < k; ++t) {
                            if (found[j] == truth[q * k + t]) {
                                ++hits;
                                break;
                            }
                        }
                    }
                }
                if (repeat >= 0) {
                    csv << type << ',' << ef << ',' << repeat << ',' << nq << ',' << successful
                        << ',' << nq - successful << ',' << seconds << ','
                        << static_cast<double>(successful) / seconds << ','
                        << static_cast<double>(hits) / (nq * k) << '\n';
                    csv.flush();
                }
            }
            std::cout << type << " completed ef_search=" << ef << std::endl;
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
