
set(name roaringbitmap)
set(source_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/source)
set(install_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/install)
ExternalProject_Add(
    ${name}
    URL http://tbase.cn-hangzhou.alipay.aliyun-inc.com/vsag%2Fthirdparty%2Froaringbitmap-3.0.1.tar.gz?OSSAccessKeyId=LTAILFuN8FJQZ4Eu&Expires=101716520277&Signature=QtNyVXkKV2F3vdeL%2F1NICPhV6qI%3D
    URL_HASH MD5=463db911f97d5da69393d4a3190f9201
    DOWNLOAD_NAME roaringbitmap-3.0.1.tar.gz
    PREFIX
        ${CMAKE_CURRENT_BINARY_DIR}/${name}
    TMP_DIR
        ${BUILD_INFO_DIR}
    STAMP_DIR
        ${BUILD_INFO_DIR}
    DOWNLOAD_DIR
        ${DOWNLOAD_DIR}
    SOURCE_DIR
        ${source_dir}
    CONFIGURE_COMMAND
    	cmake -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DCMAKE_INSTALL_PREFIX=${install_dir} -S. -Bbuild
    BUILD_COMMAND
	cmake --build build --target install --parallel ${NUM_BUILDING_JOBS}
    INSTALL_COMMAND
        ""
    BUILD_IN_SOURCE 1
    LOG_CONFIGURE TRUE
    LOG_BUILD TRUE
    LOG_INSTALL TRUE
    DOWNLOAD_NO_PROGRESS 1
)

include_directories (${install_dir}/include)
link_directories (${install_dir}/lib)
link_directories (${install_dir}/lib64)
