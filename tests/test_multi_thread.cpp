
#include <catch2/catch_test_macros.hpp>
#include <future>
#include <nlohmann/json.hpp>
#include <thread>

#include "vsag/vsag.h"

class ThreadPool {
public:
    ThreadPool(size_t threads) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock,
                                             [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template <class F, class... Args>
    auto
    enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) worker.join();
    }

private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;

    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

float
query_knn(std::shared_ptr<vsag::Index> index,
          const vsag::Dataset& query,
          int64_t id,
          int64_t k,
          const std::string& parameters,
          vsag::BitsetPtr invalid) {
    if (auto result = index->KnnSearch(query, k, parameters, invalid); result.has_value()) {
        if (result->GetDim() != 0 && result->GetIds()[0] == id) {
            return 1.0;
        } else {
            std::cout << result->GetDim() << " " << result->GetIds()[0] << " " << id << std::endl;
        }
    } else if (result.error().type == vsag::ErrorType::INTERNAL_ERROR) {
        std::cerr << "failed to perform knn search on index" << std::endl;
    }
    return 0.0;
}

TEST_CASE("DiskAnn Multi-threading", "[diskann][test][diskann-ci-part4]") {
    int dim = 65;             // Dimension of the elements
    int max_elements = 1000;  // Maximum number of elements, should be known beforehand
    int max_degree = 16;      // Tightly connected with internal dimensionality of the data
    // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int ef_search = 100;
    int io_limit = 200;
    float threshold = 8.0;
    float pq_sample_rate =
        0.5;  // pq_sample_rate represents how much original data is selected during the training of pq compressed vectors.
    int pq_dims = 9;  // pq_dims represents the dimensionality of the compressed vector.
    nlohmann::json diskann_parameters{{"max_degree", max_degree},
                                      {"ef_construction", ef_construction},
                                      {"pq_sample_rate", pq_sample_rate},
                                      {"pq_dims", pq_dims}};
    nlohmann::json index_parameters{
        {"dtype", "float32"},
        {"metric_type", "l2"},
        {"dim", dim},
        {"diskann", diskann_parameters},
    };

    std::shared_ptr<vsag::Index> diskann;
    auto index = vsag::Factory::CreateIndex("diskann", index_parameters.dump()).value();

    std::shared_ptr<int64_t[]> ids(new int64_t[max_elements]);
    std::shared_ptr<float[]> data(new float[dim * max_elements]);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int i = 0; i < max_elements; i++) ids[i] = i;
    for (int i = 0; i < dim * max_elements; i++) data[i] = distrib_real(rng);

    // Build index
    vsag::Dataset dataset;
    dataset.Dim(dim)
        .NumElements(max_elements)
        .Ids(ids.get())
        .Float32Vectors(data.get())
        .Owner(false);
    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    ThreadPool pool(10);
    std::vector<std::future<float>> future_results;
    float correct = 0;
    nlohmann::json parameters{
        {"diskann", {{"ef_search", ef_search}, {"beam_search", 4}, {"io_limit", io_limit}}}};
    std::string str_parameters = parameters.dump();

    size_t bytes_count = max_elements / 8 + 1;
    auto bits_zeros = new uint8_t[bytes_count];
    std::memset(bits_zeros, 0, bytes_count);
    vsag::BitsetPtr zeros = std::make_shared<vsag::Bitset>(bits_zeros, bytes_count);
    for (int i = 0; i < max_elements; i++) {
        int64_t k = 2;
        future_results.push_back(
            pool.enqueue([&index, &ids, dim, &data, i, k, &str_parameters, &zeros]() -> float {
                vsag::Dataset query;
                query.NumElements(1).Dim(dim).Float32Vectors(data.get() + i * dim).Owner(false);
                return query_knn(index, query, *(ids.get() + i), k, str_parameters, zeros);
            }));
    }
    for (int i = 0; i < future_results.size(); ++i) {
        correct += future_results[i].get();
    }

    float recall = correct / max_elements;
    std::cout << index->GetStats() << std::endl;
    REQUIRE(recall == 1);
    delete[] bits_zeros;
}

TEST_CASE("HNSW Multi-threading", "[hnsw]") {
    int dim = 16;             // Dimension of the elements
    int max_elements = 1000;  // Maximum number of elements, should be known beforehand
    int max_degree = 16;      // Tightly connected with internal dimensionality of the data
    // strongly affects the memory consumption
    int ef_construction = 200;  // Controls index search speed/build speed tradeoff
    int ef_search = 100;
    float threshold = 8.0;
    nlohmann::json hnsw_parameters{
        {"max_degree", max_degree},
        {"ef_construction", ef_construction},
        {"ef_search", ef_search},
    };
    nlohmann::json index_parameters{
        {"dtype", "float32"}, {"metric_type", "l2"}, {"dim", dim}, {"hnsw", hnsw_parameters}};
    auto index = vsag::Factory::CreateIndex("hnsw", index_parameters.dump()).value();
    std::shared_ptr<int64_t[]> ids(new int64_t[max_elements]);
    std::shared_ptr<float[]> data(new float[dim * max_elements]);

    // Generate random data
    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int i = 0; i < max_elements; i++) ids[i] = i;
    for (int i = 0; i < dim * max_elements; i++) data[i] = distrib_real(rng);
    // Build index
    vsag::Dataset dataset;
    dataset.Dim(dim)
        .NumElements(max_elements)
        .Ids(ids.get())
        .Float32Vectors(data.get())
        .Owner(false);

    auto result = index->Build(dataset);
    REQUIRE(result.has_value());

    ThreadPool pool(10);
    std::vector<std::future<float>> future_results;
    float correct = 0;
    nlohmann::json parameters{
        {"hnsw", {{"ef_search", ef_search}}},
    };
    std::string str_parameters = parameters.dump();
    size_t bytes_count = max_elements / 8 + 1;
    auto bits_zeros = new uint8_t[bytes_count];
    std::memset(bits_zeros, 0, bytes_count);
    vsag::BitsetPtr zeros = std::make_shared<vsag::Bitset>(bits_zeros, bytes_count);
    for (int i = 0; i < max_elements; i++) {
        int64_t k = 2;
        future_results.push_back(
            pool.enqueue([&index, &ids, dim, &data, i, k, &str_parameters, &zeros]() -> float {
                vsag::Dataset query;
                query.NumElements(1).Dim(dim).Float32Vectors(data.get() + i * dim).Owner(false);
                return query_knn(index, query, *(ids.get() + i), k, str_parameters, zeros);
            }));
    }
    for (int i = 0; i < future_results.size(); ++i) {
        correct += future_results[i].get();
    }

    float recall = correct / max_elements;
    std::cout << index->GetStats() << std::endl;
    REQUIRE(recall == 1);
    delete[] bits_zeros;
}