
#include "fixtures.h"

#include <cstdint>
#include <string>

#include "../extern/hnswlib/hnswlib/hnswlib.h"
#include "fmt/format.h"
#include "vsag/dataset.h"

namespace fixtures {

void
normalize(float* input_vector, int64_t dim) {
    float magnitude = 0.0f;
    for (int64_t i = 0; i < dim; ++i) {
        magnitude += input_vector[i] * input_vector[i];
    }
    magnitude = std::sqrt(magnitude);

    for (int64_t i = 0; i < dim; ++i) {
        input_vector[i] = input_vector[i] / magnitude;
    }
}

std::vector<float>
generate_vectors(int64_t num_vectors, int64_t dim) {
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    std::vector<float> vectors(dim * num_vectors);
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }
    for (int64_t i = 0; i < num_vectors; ++i) {
        normalize(vectors.data() + i * dim, dim);
    }

    return vectors;
}

std::tuple<std::vector<int64_t>, std::vector<float>>
generate_ids_and_vectors(int64_t num_vectors, int64_t dim) {
    std::vector<int64_t> ids(num_vectors);
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }

    return {ids, generate_vectors(num_vectors, dim)};
}

vsag::IndexPtr
generate_index(const std::string& name,
               const std::string& metric_type,
               int64_t num_vectors,
               int64_t dim,
               std::vector<int64_t>& ids,
               std::vector<float>& vectors) {
    auto index = vsag::Factory::CreateIndex(
                     name, vsag::generate_build_parameters(metric_type, num_vectors, dim).value())
                     .value();

    vsag::Dataset base;
    base.NumElements(num_vectors)
        .Dim(dim)
        .Ids(ids.data())
        .Float32Vectors(vectors.data())
        .Owner(false);
    if (not index->Build(base).has_value()) {
        return nullptr;
    }

    return index;
}

float
test_knn_recall(const vsag::IndexPtr& index,
                const std::string& search_parameters,
                int64_t num_vectors,
                int64_t dim,
                std::vector<int64_t>& ids,
                std::vector<float>& vectors) {
    int64_t correct = 0;
    for (int64_t i = 0; i < num_vectors; ++i) {
        vsag::Dataset query;
        query.NumElements(1).Dim(dim).Float32Vectors(vectors.data() + i * dim).Owner(false);
        auto result = index->KnnSearch(query, 10, search_parameters).value();
        for (int64_t j = 0; j < result.GetDim(); ++j) {
            if (i == result.GetIds()[j]) {
                ++correct;
                break;
            }
        }
    }

    float recall = 1.0 * correct / num_vectors;
    return recall;
}

std::string
generate_hnsw_build_parameters_string(const std::string& metric_type, int64_t dim) {
    constexpr auto parameter_temp = R"(
    {{
        "dtype": "float32",
        "metric_type": "{}",
        "dim": {},
        "hnsw": {{
            "max_degree": 64,
            "ef_construction": 500
        }}
    }}
    )";
    auto build_parameters = fmt::format(parameter_temp, metric_type, dim);
    return build_parameters;
}

vsag::Dataset
brute_force(const vsag::Dataset& query,
            const vsag::Dataset& base,
            int64_t k,
            const std::string& metric_type) {
    assert(metric_type == "l2");
    assert(query.GetDim() == base.GetDim());
    assert(query.GetNumElements() == 1);

    hnswlib::L2Space space(base.GetDim());
    auto fstdistfunc_ = space.get_dist_func();

    vsag::Dataset result;
    int64_t* ids = new int64_t[k];
    float* dists = new float[k];
    result.Ids(ids).Distances(dists).NumElements(k);

    std::priority_queue<std::pair<float, int64_t>> bf_result;

    for (uint32_t i = 0; i < base.GetNumElements(); i++) {
        float dist = fstdistfunc_(query.GetFloat32Vectors(),
                                  base.GetFloat32Vectors() + i * base.GetDim(),
                                  space.get_dist_func_param());
        if (bf_result.size() < k) {
            bf_result.push({dist, base.GetIds()[i]});
        } else {
            if (dist < bf_result.top().first) {
                bf_result.pop();
                bf_result.push({dist, base.GetIds()[i]});
            }
        }
    }

    for (int i = k - 1; i >= 0; i--) {
        ids[i] = bf_result.top().second;
        dists[i] = bf_result.top().first;
        bf_result.pop();
    }

    return std::move(result);
}
}  // namespace fixtures
