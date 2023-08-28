export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
VSAG_HOME="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

mkdir -p ${VSAG_HOME}/third-party/build
pushd ${VSAG_HOME}/third-party/build
cmake \
    -D DOWNLOAD_DIR=${VSAG_HOME}/third-party/downloads \
    -D CMAKE_INSTALL_PREFIX=${VSAG_HOME}/third-party/install \
    -D CMAKE_CXX_COMPILER_LAUNCHER="" \
  ${VSAG_HOME}/third-party
make
popd

yum install -y libaio-devel yum-utils gperftools
yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
yum install -y intel-mkl
