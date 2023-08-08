#!/usr/bin/env python3

import os
import h5py
import logging
import numpy as np
from pprint import pprint
from datetime import datetime
import oss2
from oss2.credentials import EnvironmentVariableCredentialsProvider
import vsag

logging.basicConfig(encoding='utf-8',
                    level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s',
                    handlers=[logging.FileHandler('/tmp/bench-face5m.log'),
                              logging.StreamHandler()])
logging.info(f'{__file__} running at {datetime.now()}')

OSS_ACCESS_KEY_ID = os.environ.get('OSS_ACCESS_KEY_ID')
OSS_ACCESS_KEY_SECRET = os.environ.get('OSS_ACCESS_KEY_SECRET')
OSS_ENDPOINT = os.environ.get('OSS_ENDPOINT')
OSS_BUCKET = os.environ.get('OSS_BUCKET')
OSS_OBJECT = os.environ.get('OSS_OBJECT')

if not OSS_ACCESS_KEY_ID:
    logging.info(f"OSS_ACCESS_KEY_ID is not set")
if not OSS_ACCESS_KEY_SECRET:
    logging.info(f"OSS_ACCESS_KEY_SECRET is not set")
if not OSS_ENDPOINT:
    logging.info(f"OSS_ENDPOINT is not set")
if not OSS_BUCKET:
    logging.info(f"OSS_BUCKET is not set")
if not OSS_OBJECT:
    logging.info(f"OSS_OBJECT is not set")
if None in [OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_ENDPOINT, OSS_BUCKET, OSS_OBJECT]:
    exit(-1)

auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
bucket = oss2.Bucket(auth, OSS_ENDPOINT, OSS_BUCKET)
bucket.get_object_to_file(OSS_OBJECT, '/tmp/face5m.h5')

with h5py.File("/tmp/face5m.h5", 'r') as file:
    base = np.array(file['base'])
    index = vsag.Index(base.shape[1], base.shape[0], "ip", "int8")
    for i, item in enumerate(base):
        index.addPoint(item, i)
    correct = 0
    for i, item in enumerate(base):
        labels, distances = index.searchTopK(item, 1)
        if labels[0] == i:
            correct += 1
    logging.info(f"recall: {correct/len(base)}")

