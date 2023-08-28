import os
import h5py
import logging
import numpy as np
import time
from datetime import datetime
import vsag
from .utils import read_dataset

def build_hnsw(datas, dist_type, data_type, M, ef_construct):
    index = vsag.HNSWIndex(datas.shape[1], datas.shape[0], dist_type, data_type, M, ef_construct)
    for i, item in enumerate(datas):
        index.addPoint(item, i)
    return index

def build_vamana(datas, dist_type, data_type, M, ef_construct):
    index = vsag.HNSWIndex(datas.shape[1], datas.shape[0], dist_type, data_type)
    index.build(datas.flatten(), M, ef_construct)
    return index


def build_index(index_name, datas, dist_type, data_type, M, ef_construct):
    if index_name == "vamana":
        return build_vamana(datas, dist_type, data_type, M, ef_construct)
    elif index_name == "hnsw":
        return build_hnsw(datas, dist_type, data_type, M, ef_construct)



def run():

    logging.basicConfig(encoding='utf-8',
                        level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler('/tmp/bench-security.log'),
                                  logging.StreamHandler()])
    logging.info(f'{__file__} running at {datetime.now()}')


    with read_dataset("security-1m.h5", logging) as file:
        base = np.array(file['vector_256'])
        for index_name in ["hnsw", "vamana"]:
            for M in [16,]:
                for ef_search in [300,]:
                    logging.info(f"begin testing, parameters# index_name:{index_name}  M:{M}  ef_construct:{200}  ef_search:{ef_search}")
                    t1 = time.time()
                    index = build_index(index_name, base, "l2", "float32", M, 200)
                    t2 = time.time()
                    index.setEfRuntime(ef_search)
                    correct = 0
                    for i, item in enumerate(base):
                        labels, distances = index.searchTopK(item, 1)
                        if labels[0] == i:
                            correct += 1
                    t3 = time.time()
                    logging.info(f"building time: {t2 - t1}, searching time: {t3 - t2}")
                    logging.info(f"recall: {correct/len(base)}")