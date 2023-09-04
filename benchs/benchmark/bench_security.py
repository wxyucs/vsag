import os
import h5py
import logging
import numpy as np
import time
from datetime import datetime
import pyvsag
from .utils import read_dataset

def build_hnsw(datas, dist_type, data_type, M, ef_construct):
    index = pyvsag.HNSWIndex(datas.shape[1], datas.shape[0], dist_type, data_type, M, ef_construct)
    for i, item in enumerate(datas):
        index.addPoint(item, i)
    return index

def build_vamana(datas, dist_type, data_type, M, ef_construct):
    index = pyvsag.VamanaIndex(datas.shape[1], datas.shape[0], dist_type, data_type)
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


    k = 20

    with read_dataset("security-1m.h5", logging) as file:
        for result_key, key in zip(['ids_256', 'ids_512'], ['vector_256', 'vector_512']):
            base = np.array(file[key])
            ids = np.array(file[result_key])
            data_len = base.shape[0]
            for index_name in ["vamana"]:
                for M in [32]:
                    for ef_search in [300]:
                        for ef_construct in [200, 500]:
                            logging.info(f"begin testing, parameters# index_name:{index_name}  M:{M}  ef_construct:{ef_construct}  ef_search:{ef_search}")
                            t1 = time.time()
                            index = build_index(index_name, base, "l2", "float32", M, ef_construct)
                            t2 = time.time()
                            index.setEfRuntime(ef_search)
                            correct = 0
                            correct_1 = 0
                            for i, item in enumerate(base):
                                labels, distances = index.searchTopK(item, k)
                                ids_results = [ids[l] for l in labels]
                                if ids[i] in ids_results:
                                    correct += 1
                                if ids[i] in ids_results[-1:]:
                                    correct_1 += 1
                            t3 = time.time()
                            logging.info(f"building time: {(t2 - t1) * 1000 / data_len}, searching time: {(t3 - t2) * 1000 / data_len}")
                            logging.info(f"recall: {correct/len(base)}, recall_1@1:{correct_1/len(base)}")