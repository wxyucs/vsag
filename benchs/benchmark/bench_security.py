import os
import h5py
import logging
import numpy as np
import json
import time
from datetime import datetime
import pyvsag
from .utils import read_dataset


from itertools import product

def cartesian_product(*args):
    return list(product(*args))


def build_index(index_name, datas, ids, index_parameters):
    index = pyvsag.Index(index_name, json.dumps(index_parameters))
    index.build(datas, ids, datas.shape[0], datas.shape[1])
    return index
    



def run():

    logging.basicConfig(encoding='utf-8',
                        level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler('/tmp/bench-security.log'),
                                  logging.StreamHandler()])
    logging.info(f'{__file__} running at {datetime.now()}')


    k = 20
    for dataset in ["security-1m.h5"]:
        logging.info(f"dataset:{dataset}")
        with read_dataset(dataset, logging) as file:
            for result_key, key in zip(['ids_512'], ['vector_512']):
                base = np.array(file[key])[:12000]
                ids = np.array(file[result_key])[:12000]
                data_len = base.shape[0]
                for index_name in ["diskann"]:
                    for M, ef_search, ef_construct, beam_search, chunks_num, io_limit in [
                            (32, 300, 500, 4, 32, 500),
                            (32, 300, 500, 4, 64, 500),
                            (32, 300, 500, 4, 128, 500),
                        ]:

                        t1 = time.time()
                        index = build_index(index_name, base, ids, {
                                "dtype": "float32",
                                "metric_type": "l2",
                                "dim": base.shape[1],
                                "max_elements": base.shape[0],
                                "L": ef_construct,
                                "R": M,
                                "p_val": 0.5,
                                "disk_pq_dims": chunks_num 
                        })
                        t2 = time.time()
                        correct = 0
                        correct_1 = 0
                        io_sum = 0
                        for i, item in enumerate(base[:30000]):
                            labels, distances = index.searchTopK(item, k, json.dumps({
                                    "data_num": 1,
                                    "ef_search": ef_search, 
                                    "beam_search": beam_search, 
                                    "io_limit": io_limit
                                })
                            )
                            ids_results = [ids[l] for l in labels]
                            io_sum += distances[0]
                            if ids[i] in ids_results:
                                correct += 1
                            if ids[i] in ids_results[-1:]:
                                correct_1 += 1
                            t3 = time.time()
                            if (i % 10000 == 0 and i != 0) or i == data_len - 1:
                                logging.info(f"begin testing, parameters# index_name:{index_name}  M:{M}  ef_construct:{ef_construct}  ef_search:{ef_search}, beam_search: {beam_search}, chunks_num: {chunks_num}, io_limit: {io_limit}")
                                logging.info(f"datasize:{i}")
                                logging.info(f"avg io count:{io_sum / (i + 1)}")
                                logging.info(f"building time: {(t2 - t1) * 1000 / data_len}, searching time: {(t3 - t2) * 1000 / (i + 1)}")
                                logging.info(f"recall: {correct/(i + 1)}, recall_1@1:{correct_1/(i + 1)}")
