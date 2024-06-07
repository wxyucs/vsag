# VSAG (Vector Search for AntGroup)

## Dependencies
### Ubuntu
```
$ ./scripts/deps/install_deps_ubuntu.sh
```

### AliOS/Centos
```
$ ./scripts/deps/install_deps_centos.sh
```

## Usage
```bash
Usage: make <target>

Targets:
help:                   ## Show the help.
debug:                  ## Build vsag with debug options.
release:                ## Build vsag with release options.
format:                 ## Format codes.
test:                   ## Build and run unit tests.
test_asan:              ## Build and run unit tests with AddressSanitizer option.
test_cov:               ## Build and run unit tests with code coverage enabled.
benchmark:              ## Run benchmarks.
clean:                  ## Clear build/ directory.
install: debug          ## Build and install the debug version of vsag.
```

## Example CMake Project with VSAG
https://code.alipay.com/octopus/vsag-cmake-example

## Example codes(hnsw index)
```cpp

// single header file
#include <vsag/vsag.h>

int
main(int argc, char** argv) {
    vsag::init();

    int64_t num_vectors = 10000;
    int64_t dim = 128;

    // prepare ids and vectors
    auto ids = new int64_t[num_vectors];
    auto vectors = new float[dim * num_vectors];

    std::mt19937 rng;
    rng.seed(47);
    std::uniform_real_distribution<> distrib_real;
    for (int64_t i = 0; i < num_vectors; ++i) {
        ids[i] = i;
    }
    for (int64_t i = 0; i < dim * num_vectors; ++i) {
        vectors[i] = distrib_real(rng);
    }

    // create index
    auto hnsw_build_paramesters = R"(
    {
        "dtype": "float32",
        "metric_type": "l2",
        "dim": 128,
        "hnsw": {
            "max_degree": 16,
            "ef_construction": 100
        }
    }
    )";
    auto index = vsag::Factory::CreateIndex("hnsw", hnsw_build_paramesters).value();
    vsag::Dataset base;
    // ownership of ids and vectors moved to base, by default
    base.NumElements(num_vectors)
        .Dim(dim)
        .Ids(ids)
        .Float32Vectors(vectors);
    index->Build(base);

    // prepare a query vector
    auto query_vector = new float[dim];  // memory will be released by query the dataset
    for (int64_t i = 0; i < dim; ++i) {
        query_vector[i] = distrib_real(rng);
    }

    // search on the index
    auto hnsw_search_parameters = R"(
    {
        "hnsw": {
            "ef_search": 100
        }
    }
    )";
    int64_t topk = 10;
    vsag::Dataset query;
    // ownership of query_vector moved to query, by default
    query.NumElements(1)
         .Dim(dim)
         .Float32Vectors(query_vector);
    auto result = index->KnnSearch(query, topk, hnsw_search_parameters).value();

    // print the results
    std::cout << "results: " << std::endl;
    for (int64_t i = 0; i < result.GetDim(); ++i) {
        std::cout << result.GetIds()[i] << ": " << result.GetDistances()[i] << std::endl;
    }

    return 0;
}
```
