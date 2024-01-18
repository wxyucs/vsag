import oss2
import os
import h5py
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import datetime
from oss2.credentials import EnvironmentVariableCredentialsProvider
import ast

OSS_ACCESS_KEY_ID = os.environ.get('OSS_ACCESS_KEY_ID')
OSS_ACCESS_KEY_SECRET = os.environ.get('OSS_ACCESS_KEY_SECRET')
OSS_ENDPOINT = os.environ.get('OSS_ENDPOINT')
OSS_BUCKET = os.environ.get('OSS_BUCKET')
OSS_SOURCE_DIR = os.environ.get('OSS_SOURCE_DIR')

_auth = oss2.ProviderAuth(EnvironmentVariableCredentialsProvider())
_bucket = oss2.Bucket(_auth, OSS_ENDPOINT, OSS_BUCKET)
target_dir = '/tmp/dataset'
def download_and_open_dataset(dataset_name, logging=None):
    if None in [OSS_ACCESS_KEY_ID, OSS_ACCESS_KEY_SECRET, OSS_ENDPOINT, OSS_BUCKET, OSS_SOURCE_DIR]:
        if logging is not None:
            logging.error("missing oss env")
        exit(-1)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    source_file = os.path.join(OSS_SOURCE_DIR, dataset_name)
    target_file = os.path.join(target_dir, dataset_name)

    if not os.path.exists(target_file):
        _bucket.get_object_to_file(source_file, target_file)
    return h5py.File(target_file, 'r')


def read_dataset(dataset_name, logging=None):
    with download_and_open_dataset(dataset_name, logging) as file:
        train = np.array(file["train"])
        test = np.array(file["test"])
        neighbors = np.array(file["neighbors"])
        distances = np.array(file["distances"])
    return train, test, neighbors, distances


def create_dataset(base, query, topk, dataset_name, logging=None):
    nbrs = NearestNeighbors(n_neighbors=topk, metric="euclidean", algorithm='brute').fit(base)
    D, I = nbrs.kneighbors(query)
    with h5py.File(os.path.join(target_dir, dataset_name), "w") as f:
        f.create_dataset("train", data=base)
        f.create_dataset("test", data=query)
        f.create_dataset("neighbors", data=I)
        f.create_dataset("distances", data=D)


def csv_to_dataset(filename, base_column, query_column, topk, dataset_name, logging=None):
    df = pd.read_csv(filename)
    df[base_column] = df[base_column].apply(lambda x: np.array([float(i) for i in x.split(",")], dtype=np.float32))
    base = np.array(df[base_column].tolist())
    if base_column == query_column:
        query = base[:300000]
    else:
        df[query_column] = df[query_column].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
        query = np.array(df[query_column].tolist())
    create_dataset(base, query, topk, dataset_name, logging)


if __name__ == '__main__':
    nb = 100000
    nq = 10000
    dim = 128
    topk = 100
    base = np.random.rand(nb, dim)
    query = np.random.rand(nq, dim)

    create_dataset(base, query, topk,"random-100k-128-euclidean.hdf5")
    # csv_to_dataset("/tmp/dataset/merged.csv", "feature", "feature", 100, "redvector-600k-1024-euclidean.hdf5")






