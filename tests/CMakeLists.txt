
#  unittests
file (GLOB_RECURSE UNIT_TESTS "../src/*_test.cpp")
add_executable (unittests ${UNIT_TESTS}
    fixtures/fixtures.cpp
)
target_include_directories (unittests PRIVATE "./fixtures")
target_link_libraries (unittests PRIVATE Catch2::Catch2WithMain vsag simd)
add_dependencies (unittests spdlog Catch2)

# function tests
file (GLOB FUNCTION_TESTS "*.cpp")
add_executable (functests
    ${FUNCTION_TESTS}
    fixtures/fixtures.cpp
)
target_include_directories (functests PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/spdlog/install/include
    ${HDF5_INCLUDE_DIRS})
target_link_libraries (functests PRIVATE Catch2::Catch2WithMain vsag simd)
add_dependencies (functests spdlog Catch2)

# test performance tool
add_executable (test_performance performance/test_performance.cpp)
target_link_libraries (test_performance PRIVATE vsag simd libhdf5_cpp.a libhdf5.a z)
add_dependencies (test_performance hdf5)
