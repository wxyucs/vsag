
#include <cpuinfo.h>

#include <catch2/catch_test_macros.hpp>
#include <iostream>

TEST_CASE("CPU info", "[ft][cpuinfo]") {
    cpuinfo_initialize();
    std::cout << cpuinfo_get_processors_count() << std::endl;
    cpuinfo_deinitialize();
}
