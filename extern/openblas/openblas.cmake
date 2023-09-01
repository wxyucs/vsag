
set(name openblas)
set(source_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/source)
set(install_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/install)
ExternalProject_Add(
    ${name}
    URL http://tbase.cn-hangzhou.alipay.aliyun-inc.com/vsag%2Fthirdparty%2FOpenBLAS-v0.3.23.tar.gz
    URL_HASH MD5=115634b39007de71eb7e75cf7591dfb2
    DOWNLOAD_NAME OpenBLAS-v0.3.23.tar.gz
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${name}
    TMP_DIR ${BUILD_INFO_DIR}
    STAMP_DIR ${BUILD_INFO_DIR}
    DOWNLOAD_DIR ${DOWNLOAD_DIR}
    SOURCE_DIR ${source_dir}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND
	make
	-j${NUM_BUILDING_JOBS}
    INSTALL_COMMAND
	make
	PREFIX=${install_dir}
	install
    BUILD_IN_SOURCE 1
    LOG_CONFIGURE TRUE
    LOG_BUILD TRUE
    LOG_INSTALL TRUE
    DOWNLOAD_NO_PROGRESS 1
)
