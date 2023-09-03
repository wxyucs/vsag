import pyvsag
import numpy as np
import pickle
import sys

sys.path.append("/tbase-project/cluster/vsag/build/")

"""
Example of search
"""


def float32_hnsw_test():
    dim = 256
    num_elements = 50000

    # Generating sample data
    data = np.float32(np.random.random((num_elements, dim)))

    # Declaring index
    p = pyvsag.HNSWIndex(dim, num_elements, "l2", "float32", 64, 200, 10)

    for i, item in enumerate(data):
        p.addPoint(item, i)

    correct = 0
    for i, item in enumerate(data):
        labels, distances = p.searchTopK(item, 1)
        if labels[0] == i:
            correct += 1
    print("float32 recall:", correct / len(data))

def int8_hnsw_test():
    dim = 128
    num_elements = 10000

    # Generating sample data
    data = np.int8(np.random.randint(-128, 127, size=(num_elements, dim)))

    # Declaring index
    p = pyvsag.HNSWIndex(dim, num_elements, "ip", "int8")

    for i, item in enumerate(data):
        p.addPoint(item, i)

    correct = 0
    for i, item in enumerate(data):
        labels, distances = p.searchTopK(item, 1)
        if labels[0] == i:
            correct += 1
    print("int8 recall:", correct / len(data))

if __name__ == '__main__':
    # int8_hnsw_test()
    float32_hnsw_test()