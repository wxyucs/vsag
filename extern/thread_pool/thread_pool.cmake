
include(FetchContent)

# https://github.com/log4cplus/ThreadPool
FetchContent_Declare(
        thread_pool
        URL http://aivolvo-dev.cn-hangzhou-alipay-b.oss-cdn.aliyun-inc.com/vsag/third-party/ThreadPool_7ea1ee6b.zip
        URL_HASH MD5=044b9437bbc94c389149954f1831f126)

FetchContent_MakeAvailable(thread_pool)
include_directories(${thread_pool_SOURCE_DIR}/)

