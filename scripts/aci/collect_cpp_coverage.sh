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
lcov -q --rc lcov_branch_coverage=1 \
     --include '*/vsag/include/*' \
     --include '*/vsag/src/*' \
     --capture \
     --directory . \
     --output-file  "$ROOT_DIR/coverage_ut.info"

lcov --remove "${ROOT_DIR}"/coverage_ut.info '*/examples/*' '*/tests/*' \
     -o "${COVERAGE_DIR}/coverage_ut.info" \
     --rc lcov_branch_coverage=1

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
$lcov_command -o coverage.info --rc lcov_branch_coverage=1
popd


# lcov_cobertura工具对coverage.info进行处理，生成cobertura.xml
wget http://aivolvo-dev.cn-hangzhou-alipay-b.oss-cdn.aliyun-inc.com/citools/lcov_cobertura.py -O lcov_cobertura.py
python lcov_cobertura.py  $ROOT_DIR/testresult/coverage/coverage.info \
       --output $ROOT_DIR/testresult/coverage/cobertura.xml \
       --demangle
ls -al $ROOT_DIR/testresult/coverage
