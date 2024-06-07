Include(FetchContent)
FetchContent_Declare(
  cpuinfo
  URL      http://tbase.cn-hangzhou.alipay.aliyun-inc.com/vsag%2Fthirdparty%2Fcpuinfo_b8b29a1.tar.gz?OSSAccessKeyId=LTAILFuN8FJQZ4Eu&Expires=101701955278&Signature=J3Ix5G4T9ZYuFcFcI2LhyoXIpsY%3D
  URL_HASH MD5=b4b01be5dab40713942f3f57e8d58dbd
)

set(CPUINFO_BUILD_TOOLS OFF CACHE BOOL "Disable some option in the library" FORCE)
set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "Disable some option in the library" FORCE)
set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "Disable some option in the library" FORCE)
set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "Disable some option in the library" FORCE)
set(CPUINFO_BUILD_PKG_CONFIG OFF CACHE BOOL "Disable some option in the library" FORCE)

# exclude cpuinfo in vsag installation
FetchContent_GetProperties(cpuinfo)
if(NOT cpuinfo_POPULATED)
  FetchContent_Populate(cpuinfo)
  add_subdirectory(${cpuinfo_SOURCE_DIR} ${cpuinfo_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

include_directories(${cpuinfo_SOURCE_DIR}/include)
