
#include "./dataset_impl.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>

namespace vsag {

DatasetPtr
Dataset::Make() {
    return std::make_shared<DatasetImpl>();
}

};  // namespace vsag
