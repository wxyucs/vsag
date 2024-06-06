
#include "./dataset_impl.h"

#include <catch2/catch_test_macros.hpp>

#include "vsag/dataset.h"

TEST_CASE("test dataset", "[ut][dataset]") {
    vsag::Dataset::Make();
}
