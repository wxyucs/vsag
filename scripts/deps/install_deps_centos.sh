yum install -y gfortran libaio libaio-devel python3-devel
# 安装lcov
wget http://aivolvo-dev.cn-hangzhou-alipay-b.oss-cdn.aliyun-inc.com/tbase/tools/lcov-1.15-1.noarch.rpm -O /tmp/lcov-1.15-1.noarch.rpm
yum -y -q install /tmp/lcov-1.15-1.noarch.rpm && rm -f /tmp/lcov-1.15-1.noarch.rpm
