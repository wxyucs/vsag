import pyvsag
import numpy as np
import pickle
import sys
import json


def float32_diskann_test():
    dim = 128
    num_elements = 10000

    # Generating sample data
    ids = range(num_elements)
    data = np.float32(np.random.random((num_elements, dim)))

    # Declaring index
    index_params = json.dumps({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": dim,
        "diskann": {
            "R": 16,
            "L": 100,
            "p_val": 0.5,
            "disk_pq_dims": 32
        }
    })
    index = pyvsag.Index("diskann", index_params)

    index.build(vectors=data,
                ids=ids,
                num_elements=num_elements,
                dim=dim)

    search_params = json.dumps({
        "diskann": {
            "ef_search": 100,
            "beam_search": 4,
            "io_limit": 200
        }
    })
    correct = 0
    for _id, vector in zip(ids, data):
        _ids, dists = index.knn_search(vector=vector, k=11, parameters=search_params)
        if _id in _ids:
            correct += 1
    print("float32 recall:", correct / len(ids))

if __name__ == '__main__':
    float32_diskann_test()
