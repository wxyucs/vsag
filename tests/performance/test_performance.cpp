
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_set>

#include "H5Cpp.h"
#include "nlohmann/json.hpp"
#include "spdlog/spdlog.h"
#include "vsag/vsag.h"

using namespace nlohmann;
using namespace spdlog;
using namespace vsag;

json
run_test(const std::string& index_name,
         const std::string& build_parameters,
         const std::string& search_parameters,
         const std::string& dataset_path);

int
main(int argc, char* argv[]) {
    set_level(level::off);
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset_file_path> <index_name> <build_param> <search_param>" << std::endl;
        return -1;
    }

    std::string dataset_filename = argv[1];
    std::string index_name = argv[2];
    std::string build_parameters = argv[3];
    std::string search_parameters = argv[4];

    auto result = run_test(dataset_filename, index_name, build_parameters, search_parameters);
    spdlog::debug("done");
    std::cout << result.dump(4) << std::endl;

    return 0;
}

class TestDataset;
using TestDatasetPtr = std::shared_ptr<TestDataset>;
class TestDataset {
public:
    static TestDatasetPtr
    Load(const std::string& filename) {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        // check datasets exist
        {
            auto datasets = get_datasets(file);
            assert(datasets.count("train"));
            assert(datasets.count("test"));
            assert(datasets.count("neighbors"));
        }

        // get and (should check shape)
        auto train_shape = get_shape(file, "train");
        spdlog::debug("train.shape: " + to_string(train_shape));
        auto test_shape = get_shape(file, "test");
        spdlog::debug("test.shape: " + to_string(test_shape));
        auto neighbors_shape = get_shape(file, "neighbors");
        spdlog::debug("neighbors.shape: " + to_string(neighbors_shape));
        assert(train_shape.second == test_shape.second);

        auto obj = std::make_shared<TestDataset>();
        obj->train_shape_ = train_shape;
        obj->test_shape_ = test_shape;
        obj->neighbors_shape_ = neighbors_shape;
        obj->dim_ = train_shape.second;
        obj->number_of_base_ = train_shape.first;
        obj->number_of_query_ = test_shape.first;

        // alloc memory
        {
            obj->train_ =
                std::shared_ptr<float[]>(new float[train_shape.first * train_shape.second]);
            obj->test_ = std::shared_ptr<float[]>(new float[test_shape.first * test_shape.second]);
            obj->neighbors_ = std::shared_ptr<int64_t[]>(
                new int64_t[neighbors_shape.first * neighbors_shape.second]);
        }

        // read from file
        {
            H5::DataSet dataset = file.openDataSet("/train");
            H5::DataSpace dataspace = dataset.getSpace();
            H5::FloatType datatype(H5::PredType::NATIVE_FLOAT);
            dataset.read(obj->train_.get(), datatype, dataspace);
        }
        {
            H5::DataSet dataset = file.openDataSet("/test");
            H5::DataSpace dataspace = dataset.getSpace();
            H5::FloatType datatype(H5::PredType::NATIVE_FLOAT);
            dataset.read(obj->test_.get(), datatype, dataspace);
        }
        {
            H5::DataSet dataset = file.openDataSet("/neighbors");
            H5::DataSpace dataspace = dataset.getSpace();
            H5::FloatType datatype(H5::PredType::NATIVE_INT64);
            dataset.read(obj->neighbors_.get(), datatype, dataspace);
        }

        return obj;
    }

public:
    std::shared_ptr<float[]>
    GetTrain() const {
        return train_;
    }

    std::shared_ptr<float[]>
    GetTest() const {
        return test_;
    }

    int64_t
    GetNearestNeighbor(int64_t i) const {
        return neighbors_[i * neighbors_shape_.second];
    }

    int64_t
    GetNumberOfBase() const {
        return number_of_base_;
    }

    int64_t
    GetNumberOfQuery() const {
        return number_of_query_;
    }

    int64_t
    GetDim() const {
        return dim_;
    }

private:
    using shape_t = std::pair<int64_t, int64_t>;
    static std::unordered_set<std::string>
    get_datasets(const H5::H5File& file) {
        std::unordered_set<std::string> datasets;
        H5::Group root = file.openGroup("/");
        hsize_t numObj = root.getNumObjs();
        for (unsigned i = 0; i < numObj; ++i) {
            std::string objname = root.getObjnameByIdx(i);
            H5O_info_t objinfo;
            root.getObjinfo(objname, objinfo);
            if (objinfo.type == H5O_type_t::H5O_TYPE_DATASET) {
                datasets.insert(objname);
            }
        }
        return datasets;
    }

    static shape_t
    get_shape(const H5::H5File& file, const std::string& dataset_name) {
        H5::DataSet dataset = file.openDataSet(dataset_name);
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims_out[2];
        int ndims = dataspace.getSimpleExtentDims(dims_out, NULL);
        return std::make_pair<int64_t, int64_t>(dims_out[0], dims_out[1]);
    }

    static std::string
    to_string(const shape_t& shape) {
        return "[" + std::to_string(shape.first) + "," + std::to_string(shape.second) + "]";
    }

private:
    std::shared_ptr<float[]> train_;
    std::shared_ptr<float[]> test_;
    std::shared_ptr<int64_t[]> neighbors_;
    shape_t train_shape_;
    shape_t test_shape_;
    shape_t neighbors_shape_;
    int64_t number_of_base_;
    int64_t number_of_query_;
    int64_t dim_;
};

class Test {
public:
    static json
    Run(const std::string& dataset_path,
        const std::string& index_name,
        const std::string& build_parameters,
        const std::string& search_parameters) {
        spdlog::debug("index_name: " + index_name);
        spdlog::debug("build_parameters: " + build_parameters);
        auto index = Factory::CreateIndex(index_name, build_parameters).value();

        spdlog::debug("dataset_path: " + dataset_path);
        auto test_dataset = TestDataset::Load(dataset_path);

        // build
        int64_t total_base = test_dataset->GetNumberOfBase();
        auto ids = range(total_base);
        Dataset base;
        base.NumElements(total_base)
            .Dim(test_dataset->GetDim())
            .Ids(ids.get())
            .Float32Vectors(test_dataset->GetTrain().get())
            .Owner(false);
        auto build_start = std::chrono::steady_clock::now();
        if (auto buildindex = index->Build(base); not buildindex.has_value()) {
            std::cerr << "build error: " << buildindex.error().message << std::endl;
            exit(-1);
        }
        auto build_finish = std::chrono::steady_clock::now();

        // search
        auto search_start = std::chrono::steady_clock::now();
        int64_t correct = 0;
        int64_t total = test_dataset->GetNumberOfQuery();
        spdlog::debug("total: " + std::to_string(total));
        std::vector<Dataset> results;
        for (int64_t i = 0; i < total; ++i) {
            Dataset query;
            query.NumElements(1)
                .Dim(test_dataset->GetDim())
                .Float32Vectors(test_dataset->GetTest().get() + i * test_dataset->GetDim())
                .Owner(false);

            auto result = index->KnnSearch(query, 10, search_parameters);
            if (not result.has_value()) {
                std::cerr << "query error: " << result.error().message << std::endl;
                exit(-1);
            }
            results.emplace_back(std::move(result.value()));
        }
        auto search_finish = std::chrono::steady_clock::now();

        // calculate recall
        for (int64_t i = 0; i < total; ++i) {
            for (int64_t j = 0; j < results[i].GetDim(); ++j) {
                // 1@10
                if (results[i].GetIds()[j] == test_dataset->GetNearestNeighbor(i)) {
                    ++correct;
                    break;
                }
            }
        }
        spdlog::debug("correct: " + std::to_string(correct));
        float recall = 1.0 * correct / total;

        json output;
        // input
        output["index_name"] = index_name;
        output["build_parameters"] = build_parameters;
        output["search_parameters"] = search_parameters;
        output["dataset"] = dataset_path;
        // for debugging
        double search_time_in_second =
            std::chrono::duration<double>(search_finish - search_start).count();
        output["search_time_in_second"] = search_time_in_second;
        output["correct"] = correct;
        output["num_base"] = total_base;
        output["num_query"] = total;
        // key results
        double build_time_in_second =
            std::chrono::duration<double>(build_finish - build_start).count();
        output["build_time_in_second"] = build_time_in_second;
        output["recall"] = recall;
        output["tps"] = total_base / build_time_in_second;
        output["qps"] = total / search_time_in_second;
        return output;
    }

private:
    static std::shared_ptr<int64_t[]>
    range(int64_t length) {
        auto result = std::shared_ptr<int64_t[]>(new int64_t[length]);
        for (int64_t i = 0; i < length; ++i) {
            result[i] = i;
        }
        return result;
    }
};

nlohmann::json
run_test(const std::string& dataset_path,
         const std::string& index_name,
         const std::string& build_parameters,
         const std::string& search_parameters) {
    return Test::Run(dataset_path, index_name, build_parameters, search_parameters);
}
