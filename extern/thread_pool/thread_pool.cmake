
include(FetchContent)

FetchContent_Declare(
        thread_pool
        URL http://tbase.cn-hangzhou.alipay.aliyun-inc.com/vsag%2Fthirdparty%2Fthread_pool.tar.gz?OSSAccessKeyId=LTAILFuN8FJQZ4Eu&Expires=4869367210&Signature=%2BoyedDFXrx%2BcHVzq8xyDxmube5s%3D
        URL_HASH MD5=8355f0ffdd13c4b6bb9a4ea3ef6b85c4
)

FetchContent_MakeAvailable(thread_pool)
include_directories(${thread_pool_SOURCE_DIR}/)

