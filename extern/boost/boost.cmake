# Copyright (c) 2022 Ant Group. All rights reserved.

set(name boost)
set(source_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/source)
set(install_dir ${CMAKE_CURRENT_BINARY_DIR}/${name}/install)
get_filename_component(compiler_path ${CMAKE_CXX_COMPILER} DIRECTORY)
ExternalProject_Add(
    ${name}
    URL http://aivolvo-dev.cn-hangzhou-alipay-b.oss-cdn.aliyun-inc.com/tbase/third-party/boost_1_67_0.tar.gz
    URL_HASH MD5=4850fceb3f2222ee011d4f3ea304d2cb
    DOWNLOAD_NAME boost_1_67_0.tar.gz
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/${name}
    TMP_DIR ${BUILD_INFO_DIR}
    STAMP_DIR ${BUILD_INFO_DIR}
    DOWNLOAD_DIR ${DOWNLOAD_DIR}
    SOURCE_DIR ${source_dir}
    CONFIGURE_COMMAND ""
    CONFIGURE_COMMAND
        ./bootstrap.sh
            --without-icu
            --without-libraries=python,test,stacktrace,mpi,log,graph,graph_parallel
            --prefix=${install_dir}
    BUILD_COMMAND
        ./b2 install
            -d0
            -j${NUM_BUILDING_JOBS}
            --prefix=${install_dir}
            --disable-icu
            include=${install_dir}/include
            linkflags=-L${install_dir}/lib
            "cxxflags=-fPIC ${extra_cpp_flags}"
            runtime-link=static
            link=static
            variant=release
    BUILD_IN_SOURCE 1
    INSTALL_COMMAND ""
    LOG_CONFIGURE TRUE
    LOG_BUILD TRUE
    LOG_INSTALL TRUE
    DOWNLOAD_NO_PROGRESS 1
)

ExternalProject_Add_Step(${name} setup-compiler
    DEPENDEES configure
    DEPENDERS build
    COMMAND
        echo "using gcc : : ${CMAKE_CXX_COMPILER} $<SEMICOLON>"
            > ${source_dir}/tools/build/src/user-config.jam
    WORKING_DIRECTORY ${source_dir}
)

ExternalProject_Add_Step(${name} clean
    EXCLUDE_FROM_MAIN TRUE
    ALWAYS TRUE
    DEPENDEES configure
    COMMAND ./b2 clean
    COMMAND rm -f ${BUILD_INFO_DIR}/${name}-build
    WORKING_DIRECTORY ${source_dir}
)

ExternalProject_Add_StepTargets(${name} clean)

