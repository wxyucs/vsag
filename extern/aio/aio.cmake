
set(name aio)
set(source_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/source)
set(install_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/install)
ExternalProject_Add(
    ${name}
    # https://pagure.io/libaio/archive/libaio-0.3.113/libaio-libaio-0.3.113.tar.gz
    URL http://aivolvo-dev.cn-hangzhou-alipay-b.oss-cdn.aliyun-inc.com/vsag/third-party/libaio-0.3.113.tar.gz
    URL_HASH MD5=4422d9f1655f358d74ff48af2a3b9f49
    DOWNLOAD_NAME libaio-0.3.113.tar.gz
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${name}
    TMP_DIR ${BUILD_INFO_DIR}
    STAMP_DIR ${BUILD_INFO_DIR}
    DOWNLOAD_DIR ${DOWNLOAD_DIR}
    SOURCE_DIR ${source_dir}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND
        ${common_configure_envs}
        make all -j${NUM_BUILDING_JOBS}
    INSTALL_COMMAND
        make DESTDIR=${install_dir} install
    BUILD_IN_SOURCE 1
    LOG_CONFIGURE TRUE
    LOG_BUILD TRUE
    LOG_INSTALL TRUE
    DOWNLOAD_NO_PROGRESS 1
)

include_directories(${install_dir}/usr/include)
link_directories (${install_dir}/usr/lib)
link_directories (${install_dir}/usr/lib64)
