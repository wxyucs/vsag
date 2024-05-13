cmake_minimum_required (VERSION 3.18)
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

project (VSAG
  VERSION 0.9.0
  LANGUAGES CXX
)

set (NUM_BUILDING_JOBS 4)
set (CMAKE_CXX_STANDARD 17)
message ("CPU ARCH: ${CMAKE_HOST_SYSTEM_PROCESSOR}")

# debug or release
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fPIC -Werror=suggest-override")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast")
endif ()

# options
option (ENABLE_JEMALLOC "Whether to link jemalloc to all executables" OFF)
option (ENABLE_ASAN "Whether to turn AddressSanitizer ON or OFF" OFF)
option (ENABLE_COVERAGE "Whether to turn unit test coverage ON or OFF" OFF)
option (ENABLE_FUZZ_TEST "Whether to turn Fuzz Test ON or OFF" OFF)
option (ENABLE_WERROR "Whether to error on warnings" ON)
option (ENABLE_TSAN "Whether to turn Thread Sanitizer ON or OFF" OFF)
option (ENABLE_FRAME_POINTER "Whether to build with -fno-omit-frame-pointer" ON)
option (ENABLE_THIN_LTO "Whether to build with thin lto -flto=thin" OFF)
option (ENABLE_CCACHE "Whether to open ccache" OFF)
option (ENABLE_COVERAGE "Enable gcov (debug, Linux builds only)" OFF)

# ccache
find_program (CCACHE_FOUND ccache)
if (CCACHE_FOUND AND ENABLE_CCACHE)
 message (STATUS "Compiling with ccache")
 set_property (GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
 set_property (GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
endif ()

# coverage
if (ENABLE_COVERAGE AND NOT WIN32 AND NOT APPLE)
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage -DCOVERAGE_TEST_USE_GCOV")
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --coverage -DCOVERAGE_TEST_USE_GCOV")
endif ()

# asan
if (ENABLE_ASAN)
  if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0))
    message(STATUS "Compiling with AddressSanitizer and UndefinedBehaviorSanitizer")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize-recover=address -fno-sanitize=vptr")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize-recover=address -fno-sanitize=vptr")
    set (CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-optimize-sibling-calls -fsanitize-recover=address -fno-sanitize=vptr")
  else ()
    message(STATUS "Compiling with AddressSanitizer")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer -static-libasan")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer -static-libasan")
    set (CMAKE_LD_FLAGS "${CMAKE_LD_FLAGS} -g -fsanitize=address -fno-omit-frame-pointer -static-libasan")
  endif ()
endif ()

# tsan
if (ENABLE_TSAN)
    if (ENABLE_ASAN)
        message (FATAL_ERROR "ENABLE_TSAN cannot be combined with ENABLE_ASAN")
    endif ()
    set (CMAKE_REQUIRED_FLAGS "-fsanitize=thread")
    set (ENV{TSAN_OPTIONS} "report_atomic_races=0")
    add_compile_options (-fsanitize=thread)
    add_compile_options (-g)
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fsanitize=thread")
endif ()

set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

# thirdparties
include (cmake/CheckSIMDCompilerFlag.cmake)
include (ExternalProject)
include (extern/json/json.cmake)
include (extern/boost/boost.cmake)
include (extern/openblas/openblas.cmake)
include (extern/aio/aio.cmake)
include (extern/diskann/diskann.cmake)
include (extern/spdlog/spdlog.cmake)
include (extern/catch2/catch2.cmake)
include (extern/cpuinfo/cpuinfo.cmake)
include (extern/fmt/fmt.cmake)
include (extern/hdf5/hdf5.cmake)

include_directories (extern/hnswlib)
include_directories (extern/faiss)
include_directories (${CMAKE_CURRENT_BINARY_DIR}/spdlog/install/include)
include_directories (include)

# sources
add_subdirectory (src)
add_subdirectory (mockimpl)
add_subdirectory (examples/cpp)

# tests
if (ENABLE_TESTS)
     add_subdirectory (tests)
endif ()

# pybinds
if (ENABLE_PYBINDS)
    include (extern/pybind11/pybind11.cmake)
    find_package (Python3 REQUIRED COMPONENTS Interpreter Development)
    pybind11_add_module (pyvsag python_bindings/binding.cpp)
    target_link_libraries (pyvsag PRIVATE pybind11::module vsag)
endif ()

# install
install (DIRECTORY include/
	 DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install (TARGETS vsag
	 LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install (TARGETS vsag_static
	 LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}")
if (ENABLE_TESTS)
    install (TARGETS test_performance
    	 LIBRARY DESTINATION "${CMAKE_INSTALL_BINDIR}")
endif ()

# version
find_package (Git)
add_custom_target (version
  ${CMAKE_COMMAND} -D SRC=${CMAKE_SOURCE_DIR}/src/version.h.in
                   -D DST=${CMAKE_SOURCE_DIR}/src/version.h
                   -D GIT_EXECUTABLE=${GIT_EXECUTABLE}
                   -P ${CMAKE_SOURCE_DIR}/cmake/GenerateVersionHeader.cmake
  )
add_dependencies (vsag version)
