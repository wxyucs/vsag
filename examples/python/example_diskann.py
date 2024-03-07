import pyvsag
import numpy as np
import pickle
import sys
import json


def cal_recall(index, ids, data, k, search_params):
    correct = 0
    for _id, vector in zip(ids, data):
        _ids, dists = index.knn_search(vector=vector, k=k, parameters=search_params)
        if _id in _ids:
            correct += 1
    return correct / len(ids)


def float32_diskann_test():
    dim = 128
    num_elements = 1000

    # Generating sample data
    ids = range(num_elements)
    data = np.float32(np.random.random((num_elements, dim)))

    # Declaring index
    index_params = json.dumps({
        "dtype": "float32",
        "metric_type": "l2",
        "dim": dim,
        "diskann": {
            "max_degree": 32,
            "ef_construction": 100,
            "pq_sample_rate": 0.5,
            "pq_dims": 32,
            "use_pq_search": True
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
    print("[build] float32 recall:", cal_recall(index, ids, data, 11, search_params))

    root_dir = "/tmp/"
    file_sizes = index.save(root_dir)

    index = pyvsag.Index("diskann", index_params)
    index.load(root_dir, file_sizes, True)
    print("[memory] float32 recall:", cal_recall(index, ids, data, 11, search_params))

    index = pyvsag.Index("diskann", index_params)
    index.load(root_dir, file_sizes, False)
    print("[disk] float32 recall:", cal_recall(index, ids, data, 11, search_params))


if __name__ == '__main__':
    float32_diskann_test()
