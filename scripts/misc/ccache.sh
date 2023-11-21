#!/bin/bash

#yum install -y ob-ccache -b test
echo "cache_dir = $WORKSPACE/cache
compiler_check = size
compression = false
compression_level = 0
depend_mode =false
direct_mode = true
hard_link = false
hash_dir = true
keeep_comments_cpp = false
log_file = $WORKSPACE/ccache.log
max_files = 0
max_size = 100.0G
oss_access_key_id = ${OSS_ACCESS_KEY_ID}
oss_access_key_secret = ${OSS_ACCESS_KEY_SECRET}
oss_bucket = tbase
pch_external_checksum = false
read_only  = false
read_only_backend = false
read_only_direct = false
run_second_cpp = true
sloppiness = include_file_ctime,include_file_mtime,locale
temporary_dir = $WORKSPACE/ccache/tmp
backend = oss://${OSS_HOST}
region = test/vsag/cache/ccache
keep_local_storage = false
keep_local_manifest = false" > /usr/local/etc/ccache.conf
