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
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()
set(NUM_BUILDING_JOBS 4)

set(CMAKE_CXX_STANDARD 17)

include(ExternalProject)
include(extern/boost/boost.cmake)
include(extern/openblas/openblas.cmake)
include(extern/aio/aio.cmake)
include(extern/diskann/diskann.cmake)
include(extern/spdlog/spdlog.cmake)
include(extern/catch2/catch2.cmake)

include_directories(extern/hnswlib)
include_directories(extern/json/single_include)
include_directories(extern/faiss)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/spdlog/install/include)
include_directories(include)

file (GLOB CPP_SRCS "src/*.cpp")
file (GLOB CPP_INDEX_SRCS "src/index/*.cpp")
add_library (vsag SHARED ${CPP_SRCS} ${CPP_INDEX_SRCS})
target_link_libraries (vsag diskann pthread m dl)
add_dependencies(vsag spdlog)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
endif()

if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "(x86_64)|(AMD64|amd64)|(^i.86$)")
# FIXME: only work on machine with AVX512 support
#    add_compile_options(-mavx512f)
#    add_compile_options(-mavx512dq)
#    add_compile_options(-mavx512bw)
#    add_compile_options(-mavx512vl)
#    add_compile_options(-mavx)
#    add_compile_options(-msse)
endif()

add_subdirectory (examples/cpp)

add_executable (tests
  tests/test_json.cpp
  tests/test_log.cpp
  tests/test_hnsw.cpp
  tests/test_diskann.cpp
  tests/test_kmeans.cpp
  )
target_include_directories(tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/spdlog/install/include)
target_link_libraries (tests PRIVATE Catch2::Catch2WithMain vsag)
add_dependencies(tests spdlog Catch2)

if (ENABLE_PYBINDS)
    # find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    add_subdirectory(extern/pybind11)
    pybind11_add_module(pyvsag python_bindings/binding.cpp)
    target_link_libraries(pyvsag PRIVATE pybind11::module vsag)
endif ()

install(DIRECTORY include/
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS vsag
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
