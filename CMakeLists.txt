cmake_minimum_required (VERSION 3.14.5)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
	cmake_policy(SET CMP0135 NEW)
endif()

project (VSAG
  VERSION 0.9.0
  LANGUAGES CXX
)


set(NUM_BUILDING_JOBS 4)

set(CMAKE_CXX_STANDARD 17)

include (cmake/CheckSIMDCompilerFlag.cmake)

option(ENABLE_JEMALLOC "Whether to link jemalloc to all executables" OFF)
option(ENABLE_ASAN "Whether to turn AddressSanitizer ON or OFF" OFF)
option(ENABLE_COVERAGE "Whether to turn unit test coverage ON or OFF" OFF)
option(ENABLE_FUZZ_TEST "Whether to turn Fuzz Test ON or OFF" OFF)
option(ENABLE_WERROR "Whether to error on warnings" ON)
option(ENABLE_TSAN "Whether to turn Thread Sanitizer ON or OFF" OFF)
option(ENABLE_FRAME_POINTER "Whether to build with -fno-omit-frame-pointer" ON)
option(ENABLE_THIN_LTO "Whether to build with thin lto -flto=thin" OFF)

option(ENABLE_CCACHE "Whether to open ccache" OFF)
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND AND ENABLE_CCACHE)
 message(STATUS "Compiling with ccache")
 set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
 set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif()

option(ENABLE_COVERAGE "Enable gcov (debug, Linux builds only)" OFF)
if (ENABLE_COVERAGE AND NOT WIN32 AND NOT APPLE)
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --coverage -DCOVERAGE_TEST_USE_GCOV")
  SET(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS} --coverage -DCOVERAGE_TEST_USE_GCOV")
ENDIF()

if (ENABLE_ASAN)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0))
    message(STATUS "Compiling with AddressSanitizer and UndefinedBehaviorSanitizer")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize-recover=address -fno-sanitize=vptr")
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize-recover=address -fno-sanitize=vptr")
    SET (CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize-recover=address -fno-sanitize=vptr")
  else()
    message(STATUS "Compiling with AddressSanitizer")
    SET (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer -static-libasan")
    SET (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer -static-libasan")
    SET (CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer -static-libasan")
  endif()
endif()

if (ENABLE_TSAN)
    if (ENABLE_ASAN)
        MESSAGE(FATAL_ERROR "ENABLE_TSAN cannot be combined with ENABLE_ASAN")
    endif()
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=thread")
    set(ENV{TSAN_OPTIONS} "report_atomic_races=0")
    add_compile_options(-fsanitize=thread)
    add_compile_options(-g)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
include(ExternalProject)
include(extern/json/json.cmake)
include(extern/boost/boost.cmake)
include(extern/openblas/openblas.cmake)
include(extern/aio/aio.cmake)
include(extern/diskann/diskann.cmake)
include(extern/spdlog/spdlog.cmake)
include(extern/catch2/catch2.cmake)
include(extern/cpuinfo/cpuinfo.cmake)
include(extern/fmt/fmt.cmake)

include_directories(extern/hnswlib)
include_directories(extern/faiss)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/spdlog/install/include)
include_directories(include)

add_subdirectory (src/simd)

file (GLOB CPP_SRCS "src/*.cpp")
file (GLOB CPP_FACTORY_SRCS "src/factory/*.cpp")
file (GLOB CPP_INDEX_SRCS "src/index/*.cpp")
list (FILTER CPP_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_FACTORY_SRCS EXCLUDE REGEX "_test.cpp")
list (FILTER CPP_INDEX_SRCS EXCLUDE REGEX "_test.cpp")
add_library (vsag SHARED ${CPP_SRCS} ${CPP_FACTORY_SRCS} ${CPP_INDEX_SRCS})
add_library (vsag_static STATIC ${CPP_SRCS} ${CPP_FACTORY_SRCS} ${CPP_INDEX_SRCS})
target_link_libraries (vsag diskann pthread m dl simd fmt::fmt-header-only)
target_link_libraries (vsag_static diskann pthread m dl simd fmt::fmt-header-only)
add_dependencies(vsag spdlog)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fPIC -Werror=suggest-override")
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
endif()

add_subdirectory (mockimpl)

message("CPU ARCH: ${CMAKE_HOST_SYSTEM_PROCESSOR}")

add_subdirectory (examples/cpp)

if (ENABLE_TESTS)
    file (GLOB_RECURSE SRC_TESTS "src/*_test.cpp")
    add_executable (tests
	${SRC_TESTS}
	tests/test_json.cpp
	tests/test_log.cpp
	tests/test_hnsw.cpp
	tests/test_diskann.cpp
	tests/test_kmeans.cpp
	tests/test_factory.cpp
	tests/test_utils.cpp
	tests/test_bitset.cpp
	tests/test_random_index.cpp
	tests/test_cpuinfo.cpp
	tests/test_simd.cpp
	tests/test_index.cpp
    tests/test_option.cpp
    tests/test_multi_thread.cpp
    )
    target_include_directories(tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/spdlog/install/include)
    target_link_libraries (tests PRIVATE Catch2::Catch2WithMain vsag simd)
    add_dependencies(tests spdlog Catch2)
endif ()

if (ENABLE_PYBINDS)
    include(extern/pybind11/pybind11.cmake)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
    pybind11_add_module(pyvsag python_bindings/binding.cpp)
    target_link_libraries(pyvsag PRIVATE pybind11::module vsag)
endif ()

install(DIRECTORY include/
	DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(TARGETS vsag
	LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")

find_package (Git)
add_custom_target (version
  ${CMAKE_COMMAND} -D SRC=${CMAKE_SOURCE_DIR}/src/version.h.in
                   -D DST=${CMAKE_SOURCE_DIR}/src/version.h
                   -D GIT_EXECUTABLE=${GIT_EXECUTABLE}
                   -P ${CMAKE_SOURCE_DIR}/cmake/GenerateVersionHeader.cmake
  )
add_dependencies (vsag version)
