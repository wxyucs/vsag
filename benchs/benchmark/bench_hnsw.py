import os
from urllib.request import urlretrieve
import numpy as np
import time
import h5py
import vsag
from .utils import read_dataset
import datetime
import logging

def measure_all(dataset, X_train, vector_count, ef_construction, M, ef_values, num_queries, k):

    X_test = np.array(dataset['test'])
    results = np.array(dataset['neighbors'])
    print("\nRunning measure_all...")

    hnsw = vsag.Index(X_train.shape[1], X_train.shape[0], "l2", "float32", M, ef_construction, ef_values[0])

    for i in range(X_train.shape[0]):
        hnsw.addPoint(X_train[i], i)


    for ef_runtime in ef_values:
        print("\nRunning queries with ef_runtime =", ef_runtime, "...")
        hnsw.setEfRuntime(ef_runtime)
        correct = 0
        bf_total_time = 0
        bf_min_latency = 10000
        bf_max_latency = 0
        hnsw_total_time = 0
        hnsw_min_latency = 10000
        hnsw_max_latency = 0
        for i, target_vector, ground_truth in zip(range(num_queries), X_test[:num_queries], results[:num_queries]):
            start = time.time()
            result, distances = hnsw.searchTopK(target_vector, k)
            query_time = (time.time() - start)
            if query_time > hnsw_max_latency:
                hnsw_max_latency = query_time
            if query_time < hnsw_min_latency:
                hnsw_min_latency = query_time
            hnsw_total_time += query_time
            #
            # flat_query_params = create_flat_vector_query_param(flat_field_name, target_vector)
            # start = time.time()
            # flat_result = redis_client.ft(flat_index_name).search(flat_query, flat_query_params)
            # query_time = (time.time() - start)
            # if query_time > bf_max_latency:
            #     bf_max_latency = query_time
            # if query_time < bf_min_latency:
            #     bf_min_latency = query_time
            # bf_total_time += query_time
            # flat_result_ids = []
            # for i, doc in enumerate(flat_result.docs):
            #     flat_result_ids.append(doc.doc_id)
            # print(ground_truth, result)
            correct += len(np.intersect1d(result, ground_truth))

        # Measure recall
        recall = float(correct)/(k*num_queries)
        print("Average recall is:", recall)

        # print("BF query per seconds: ", num_queries/bf_total_time)
        # print("BF average lantency per query: ", bf_total_time/num_queries, "seconds")
        # print("BF min lantency for query: ", bf_min_latency, "seconds")
        # print("BF max lantency for query: ", bf_max_latency, "seconds")

        print("HNSW query per seconds: ", num_queries/hnsw_total_time)
        print("HNSW average lantency per query: ", hnsw_total_time/num_queries, "seconds")
        print("HNSW min lantency for query: ", hnsw_min_latency, "seconds")
        print("HNSW max lantency for query: ", hnsw_max_latency, "seconds")

def run_benchmark(dataset_name, ef_construction, M, ef_values, k=1):

    print("\nRunning benchmark for:", dataset_name)
    dataset = read_dataset(dataset_name)
    X_train = np.array(dataset['train'])
    distance = dataset.attrs['distance']
    dimension = int(dataset.attrs['dimension']) if 'dimension' in dataset.attrs else len(X_train[0])
    print('got a train set of size (%d * %d)' % (X_train.shape[0], dimension))
    print('metric is: %s' % distance)

    for vector_count in [800000, 900000, 1000000]:
        print("benchmark for vector count:", vector_count)
        measure_all(dataset, X_train, vector_count, ef_construction = ef_construction, M=M, ef_values=ef_values, num_queries=5, k=20)

def run():

    logging.basicConfig(encoding='utf-8',
                        level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler('/tmp/bench-hnsw.log'),
                                  logging.StreamHandler()])
    logging.info(f'{__file__} running at {datetime.now()}')
    DATASETS = ['sift-1m-128-euclidean.hdf5']
    # for every dataset the params are: (ef_construction, M, ef_runtime).
    dataset_params = {'glove-25-angular': (100, 16, [50, 100, 200]),
                      'glove-50-angular': (150, 24, [100, 200, 300]),
                      'glove-100-angular': (250, 36, [150, 300, 500]),
                      'glove-200-angular': (350, 48, [200, 350, 600]),
                      'mnist-784-euclidean': (150, 32, [100, 200, 350]),
                      'sift-1m-128-euclidean.hdf5': (200, 32, [150, 300, 500])}
    k = 10
    for d_name in DATASETS:
        run_benchmark(d_name, *(dataset_params[d_name]), k)
