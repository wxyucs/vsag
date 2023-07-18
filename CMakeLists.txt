cmake_minimum_required (VERSION 3.14.5)
project (VSAG
  VERSION 0.1.0
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)

include_directories(extern/hnswlib)
include_directories(include)

file (GLOB CPP_SRCS "src/*.cpp")
add_library (vsag SHARED ${CPP_SRCS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast")

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
    add_compile_options(-mavx512f)
    add_compile_options(-mavx512dq)
    add_compile_options(-mavx512bw)
    add_compile_options(-mavx512vl)
    add_compile_options(-mavx)
    add_compile_options(-msse)
endif()

add_executable (example_test examples/cpp/example_test.cpp)
target_link_libraries (example_test vsag)

add_executable (bench_random1m benchs/bench_random1m.cpp)
target_link_libraries (bench_random1m vsag)

add_subdirectory (extern/Catch2)
add_executable (tests tests/test_hnsw.cpp)
target_link_libraries (tests PRIVATE Catch2::Catch2WithMain)
