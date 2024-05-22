#!/bin/bash

set -e

SCRIPTS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="${SCRIPTS_DIR}/../../"

COVERAGE_DIR="${ROOT_DIR}/testresult/coverage"
if [ -d "${COVERAGE_DIR}" ]; then
    rm -rf "${COVERAGE_DIR:?}"/*
else
    mkdir -p "${COVERAGE_DIR}"
fi

# 通过lcov生成coverage info文件
lcov --rc branch_coverage=1 \
     --rc geninfo_unexecuted_blocks=1 \
     --parallel 8 \
     --include "*/vsag/include/*" \
     --include "*/vsag/src/*" \
     --include "*/vsag/extern/hnswlib/hnswlib/hnswalg_static.h" \
     --include "*/vsag/extern/hnswlib/hnswlib/hnswalg.h" \
     --exclude "*/vsag/include/vsag/expected.hpp*" \
     --exclude "*_test.cpp" \
     --capture \
     --ignore-errors mismatch,mismatch \
     --ignore-errors unused,unused \
     --directory . \
     --output-file  "${COVERAGE_DIR}/coverage_ut.info"

# 合并覆盖率文件生成最终文件coverage.info
pushd "${COVERAGE_DIR}"
coverages=$(ls coverage_*.info)
if [ ! "$coverages" ];then
    echo "no coverage file"
    exit 0
fi
lcov_command="lcov"
for coverage in $coverages; do
    echo "$coverage"
    lcov_command="$lcov_command -a $coverage"
done
$lcov_command -o coverage.info --rc branch_coverage=1
popd


# lcov_cobertura工具对coverage.info进行处理，生成cobertura.xml
wget "http://tbase.cn-hangzhou.alipay.aliyun-inc.com/vsag%2Fthirdparty%2Flcov_cobertura.py?OSSAccessKeyId=LTAILFuN8FJQZ4Eu&Expires=101708684711&Signature=N%2BNkWsU1jKi0snNMMR3YXXQtLmg%3D" -O lcov_cobertura.py
python lcov_cobertura.py  $ROOT_DIR/testresult/coverage/coverage.info \
       --output $ROOT_DIR/testresult/coverage/cobertura.xml \
       --demangle
ls -al $ROOT_DIR/testresult/coverage
