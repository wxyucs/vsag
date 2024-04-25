
include(FetchContent)

FetchContent_Declare(
    nlohmann_json
    URL http://tbase.cn-hangzhou.alipay.aliyun-inc.com/vsag%2Fthirdparty%2Fnlohmann_json.tar.gz?OSSAccessKeyId=LTAILFuN8FJQZ4Eu&Expires=4856232163&Signature=UPdZn3PKzrJ5rH829vw6quQc6NQ%3D
    URL_HASH MD5=b61b15f49a1ea5c70b81696b85b1c89b
)

FetchContent_MakeAvailable(nlohmann_json)
include_directories(${nlohmann_json_SOURCE_DIR}/include)
