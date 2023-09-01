cmake_minimum_required (VERSION 3.14.5)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
	cmake_policy(SET CMP0135 NEW)
endif()

project (VSAG
  VERSION 0.6.0
  LANGUAGES CXX
)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()
set(NUM_BUILDING_JOBS 4)

set(CMAKE_CXX_STANDARD 17)

include(ExternalProject)
include(extern/boost/boost.cmake)
include(extern/openblas/openblas.cmake)
include(extern/aio/aio.cmake)
include(extern/diskann/diskann.cmake)

include_directories(extern/hnswlib)
include_directories(extern/json/single_include)
include_directories(extern/faiss)
include_directories(extern/spdlog/include)
include_directories(include)

file (GLOB CPP_SRCS "src/*.cpp")
add_library (vsag SHARED ${CPP_SRCS})

target_link_libraries (vsag diskann
        pthread
        m
        dl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -fPIC")

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
# FIXME: only work on machine with AVX512 support
#    add_compile_options(-mavx512f)
#    add_compile_options(-mavx512dq)
#    add_compile_options(-mavx512bw)
#    add_compile_options(-mavx512vl)
#    add_compile_options(-mavx)
#    add_compile_options(-msse)
endif()

add_executable (example_hnsw examples/cpp/example_hnsw.cpp)
target_link_libraries (example_hnsw vsag)

add_executable (example_kmeans examples/cpp/example_kmeans.cpp)
target_link_libraries (example_kmeans vsag)


add_executable (example_diskann examples/cpp/example_diskann.cpp)
target_link_libraries (example_diskann vsag)

add_subdirectory (extern/Catch2)
add_executable (tests tests/test_hnsw.cpp tests/test_json.cpp tests/test_kmeans.cpp tests/test_log.cpp tests/test_diskann.cpp)
target_include_directories(tests PRIVATE third-party/install/include/boost/)
target_link_libraries (tests PRIVATE Catch2::Catch2WithMain vsag)

#find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
#add_subdirectory(extern/pybind11)
#add_library(pyvsag SHARED python_bindings/binding.cpp)
#target_link_libraries(pyvsag PRIVATE pybind11::module vsag)

