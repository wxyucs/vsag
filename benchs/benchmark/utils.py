import oss2
import os
import logging
import h5py
import datetime
from oss2.credentials import EnvironmentVariableCredentialsProvider


OSS_ACCESS_KEY_ID = os.environ.get('OSS_ACCESS_KEY_ID')
OSS_ACCESS_KEY_SECRET = os.environ.get('OSS_ACCESS_KEY_SECRET')
OSS_ENDPOINT = os.environ.get('OSS_ENDPOINT')
OSS_BUCKET = os.environ.get('OSS_BUCKET')
OSS_SOURCE_DIR = os.environ.get('OSS_SOURCE_DIR')

if not OSS_ACCESS_KEY_ID:
    logging.info(f"OSS_ACCESS_KEY_ID is not set")
if not OSS_ACCESS_KEY_SECRET:
    logging.info(f"OSS_ACCESS_KEY_SECRET is not set")
if not OSS_ENDPOINT:
    logging.info(f"OSS_ENDPOINT is not set")
if not OSS_BUCKET:
    logging.info(f"OSS_BUCKET is not set")
if not OSS_SOURCE_DIR:
    logging.info(f"OSS_SOURCE_DIR is not set")
if None in [OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_ENDPOINT, OSS_BUCKET, OSS_SOURCE_DIR]:
    exit(-1)

auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)

target_dir = '/tmp/dataset'

def read_dataset(dataset):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    source_file = os.path.join(OSS_SOURCE_DIR, dataset)
    target_file = os.path.join(target_dir, dataset)

    if not os.path.exists(target_file):
        bucket.get_object_to_file(source_file, target_file)
    return h5py.File(target_file, 'r')
