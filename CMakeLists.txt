cmake_minimum_required (VERSION 3.14.5)
project (VSAG
  VERSION 0.6.0
  LANGUAGES CXX
)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

set(CMAKE_CXX_STANDARD 17)

include_directories(extern/hnswlib)
include_directories(extern/json/single_include)
include_directories(extern/faiss)
include_directories(extern/spdlog/include)
include_directories(include)
include_directories(extern/DiskANN/include)

link_directories(/opt/intel/lib/intel64_lin/)
link_directories(/opt/intel/mkl/lib/intel64/)
add_subdirectory(extern/DiskANN)

set(BOOST_ROOT "third-party/build/boost/source")
set(Boost_NO_SYSTEM_PATHS ON)
find_package(Boost REQUIRED)

include_directories(${Boost_INCLUDE_DIR})

file (GLOB CPP_SRCS "src/*.cpp")
add_library (vsag SHARED ${CPP_SRCS})

target_link_libraries (vsag diskann
        /opt/intel/mkl/lib/intel64/libmkl_def.so
        /opt/intel/mkl/lib/intel64/libmkl_intel_ilp64.a
        /opt/intel/mkl/lib/intel64/libmkl_intel_thread.a
        /opt/intel/mkl/lib/intel64/libmkl_core.a
        iomp5
        pthread
        m
        dl)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Ofast -fPIC")

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
    add_compile_options(-mavx512f)
    add_compile_options(-mavx512dq)
    add_compile_options(-mavx512bw)
    add_compile_options(-mavx512vl)
    add_compile_options(-mavx)
    add_compile_options(-msse)
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

