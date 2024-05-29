# install python3-devel
yum install -y python3-devel

# install openmp
yum install -b current -y libomp11-devel libomp11

# install intel mkl
yum install -y ca-certificates
yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
yum install -y intel-mkl-64bit-2020.0-088
