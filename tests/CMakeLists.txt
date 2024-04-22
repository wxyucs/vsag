
# function regression and unittests
file (GLOB_RECURSE SRC_TESTS "../src/*_test.cpp")
add_executable (tests
    ${SRC_TESTS}
    test_json.cpp
    test_log.cpp
    test_hnsw.cpp
    test_diskann.cpp
    test_kmeans.cpp
    test_factory.cpp
    test_utils.cpp
    test_bitset.cpp
    test_random_index.cpp
    test_cpuinfo.cpp
    test_simd.cpp
    test_index.cpp
    test_multi_thread.cpp
    fixtures/fixtures.cpp
)
target_include_directories (tests PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/spdlog/install/include
    ${HDF5_INCLUDE_DIRS})
target_link_libraries (tests PRIVATE Catch2::Catch2WithMain vsag simd)
add_dependencies (tests spdlog Catch2)

# test performance tool
add_executable (test_performance performance/test_performance.cpp)
target_link_libraries (test_performance PRIVATE vsag simd libhdf5_cpp.a libhdf5.a z)
add_dependencies (test_performance hdf5)
