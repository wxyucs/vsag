yum install -y gfortran libaio libaio-devel python3-devel perl-CPAN perl-core
# 安装lcov
wget "http://tbase.cn-hangzhou.alipay.aliyun-inc.com/vsag%2Fthirdparty%2Flcov-2.0-1.noarch.rpm?OSSAccessKeyId=LTAILFuN8FJQZ4Eu&Expires=101708682593&Signature=tfK2UkyY6slmAvCV%2BxfzMEG5V3Y%3D" -O /tmp/lcov-2.0-1.noarch.rpm
yum -y -q install /tmp/lcov-2.0-1.noarch.rpm && rm -f /tmp/lcov-2.0-1.noarch.rpm

# replace JSON:PP with JSON:XS to improve coverage performance
PERL_MM_USE_DEFAULT=1 sudo perl -MCPAN -e 'CPAN::HandleConfig->edit("urllist", "unshift", "https://mirrors.aliyun.com/CPAN/"); mkmyconfig'
PERL_MM_USE_DEFAULT=1 sudo cpan install Test::More Canary::Stability JSON:XS

sed -i 's/lcov_json_module = auto/lcov_json_module = JSON::XS/g' /etc/lcovrc

# install intel mkl
yum update -y && yum install -y ca-certificates
yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
yum update -y && yum install -y intel-mkl-64bit-2020.0-088

# install openmp
yum install -b current -y libomp11-devel libomp11
