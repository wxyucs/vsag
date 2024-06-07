
include(FetchContent)

FetchContent_Declare(
    fmt
    URL http://tbase.cn-hangzhou.alipay.aliyun-inc.com/vsag%2Fthirdparty%2Ffmt-10.2.1.tar.gz?OSSAccessKeyId=LTAILFuN8FJQZ4Eu&Expires=101708516779&Signature=XGALK%2BUIXDvPVOxp4XD2BgIy6cY%3D
    URL_HASH MD5=dc09168c94f90ea890257995f2c497a5
)

# exclude fmt in vsag installation
FetchContent_GetProperties(fmt)
if(NOT fmt_POPULATED)
  FetchContent_Populate(fmt)
  add_subdirectory(${fmt_SOURCE_DIR} ${fmt_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()

include_directories(${fmt_SOURCE_DIR}/include)
