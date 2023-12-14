include(FetchContent)

FetchContent_Declare(
        pybind11
        URL http://tbase.cn-hangzhou.alipay.aliyun-inc.com/vsag%2Fthirdparty%2Fpybind11-2.11.1.tar.gz?OSSAccessKeyId=LTAILFuN8FJQZ4Eu&Expires=4856060929&Signature=M8qxYNmmd6UqHOclwDDiaPC7y6g%3D
        URL_HASH MD5=6525a687fb8a6291b1c4c5a10027b4ee
)

FetchContent_MakeAvailable(pybind11)
