import vsag
import numpy as np
import csv
from scipy.spatial.distance import cdist
import math
def data_load(data_size, dimension):
    vectors = np.random.rand(data_size, dimension)
    norms = np.linalg.norm(vectors, axis=1)
    normalized_vectors = vectors / norms.reshape(-1, 1)
    return normalized_vectors

def get_data_set():
    size = 0
    vector = []
    result = []
    with open("/tbase-project/data2.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        next(reader)
        for i in reader:
            numbers = i[1].replace('"', "").split(",")
            if len(numbers) != 256: continue
            vec = [int(j) / 512.0 for j in numbers]
            size += 1
            result.append(i[0])
            vector.append(vec)
            vec = np.array(vec)
            if size >= 5000000: break
    return result, vector

def get_query_set():
    global size
    vector = []
    result = []
    with open("/tbase-project/query2.csv", "r", encoding="utf8") as f:
        reader = csv.reader(f)
        next(reader)
        size = 0
        for i in reader:
            result.append(i[1])
            numbers = i[4].replace('"', "").split(",")
            vector.append([int(j) for j in numbers])
            size += 1
            if size >= 100000: break
    return result, vector


def test_kmeans():
    clusters = 400
    data_labels, data_vectors = get_data_set()
    print("data size", len(data_labels))
    centroids = vsag.kmeans(data_vectors, clusters, "ip")
    distances = cdist(data_vectors, centroids)
    min_idx = np.argmin(distances, axis=1)
    assert(len(data_labels) == len(min_idx))
    data_index = {label: idx for label, idx in zip(data_labels, min_idx)}

    results, query_vectors = get_query_set()

    ranks = []
    indexs = []
    group_num = 1
    for r, v in zip(results, query_vectors):
        v = np.array(v, dtype=np.float32)
        v /= 512
        if r not in data_index:
            print(r)
            continue
        index = int(data_index[r.strip()])
        index //= group_num
        inner_products = np.dot(centroids, v)
        sort_idx = np.argsort(inner_products)[::-1]
        sort_idx = np.floor_divide(sort_idx, group_num)
        uni = set()
        unique_idx = list()
        for i in sort_idx:
            if i not in uni:
                uni.add(i)
                unique_idx.append(i)
        sort_idx = np.array(unique_idx)
        rank = np.where(sort_idx == index)[0][0]
        ranks.append(rank)
        indexs.append(index)
    ranks.sort()

    print("P90: %d" % ranks[math.floor(len(ranks) * 0.90)])
    print("P99: %d" % ranks[math.floor(len(ranks) * 0.99)])
    print("P999: %d" % ranks[math.floor(len(ranks) * 0.999)])
    print("P9995: %d" % ranks[math.floor(len(ranks) * 0.9995)])
    print("P9997: %d" % ranks[math.floor(len(ranks) * 0.9997)])
    print("P9999: %d" % ranks[math.floor(len(ranks) * 0.9999)])





if __name__ == "__main__":
    test_kmeans()


