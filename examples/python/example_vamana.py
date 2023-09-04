import pyvsag
import numpy as np
import pickle


"""
Example of search
"""


def float32_vemana_test():
    dim = 128
    num_elements = 10000

    # Generating sample data
    data = np.float32(np.random.random((num_elements, dim)))

    # Declaring index
    p = pyvsag.VamanaIndex(dim, num_elements, "l2", "float32")

    build_data = data.flatten()

    p.build(build_data, 32, 200)

    correct = 0
    for i, item in enumerate(data):
        labels, distances = p.searchTopK(item, 1)
        if labels[0] == i:
            correct += 1
    print("float32 recall:", correct / len(data))

if __name__ == '__main__':
    float32_vemana_test()