cmake_minimum_required (VERSION 3.26)
project (VSAG
  VERSION 0.1.0
  LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)

include_directories(extern/hnswlib)
include_directories(include)

file (GLOB CPP_SRCS "src/*.cpp")
add_library (vsag SHARED ${CPP_SRCS})

add_executable (example_test examples/cpp/example_test.cpp)
target_link_libraries (example_test vsag)
