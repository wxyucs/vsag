include(FetchContent)
FetchContent_Declare(
		roaringbitmap
		URL "http://tbase.cn-hangzhou.alipay.aliyun-inc.com/vsag%2Fthirdparty%2Froaringbitmap-3.0.1.tar.gz?OSSAccessKeyId=LTAILFuN8FJQZ4Eu&Expires=101716520277&Signature=QtNyVXkKV2F3vdeL%2F1NICPhV6qI%3D"
		URL_HASH MD5=463db911f97d5da69393d4a3190f9201
)

set(ROARING_USE_CPM OFF)
set(ENABLE_ROARING_TESTS OFF)

if (NOT COMPILER_AVX_SUPPORTED)
  set(ROARING_DISABLE_AVX ON)
endif ()

if (NOT COMPILER_AVX512_SUPPORTED)
  set (ROARING_DISABLE_AVX512 ON)
endif ()

FetchContent_MakeAvailable(roaringbitmap)
