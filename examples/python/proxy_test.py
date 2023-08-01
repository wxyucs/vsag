import vsag
import numpy as np
from scipy.spatial.distance import cdist

def data_load(data_size, dimension):
    vectors = np.random.rand(data_size, dimension)
    norms = np.linalg.norm(vectors, axis=1)
    normalized_vectors = vectors / norms.reshape(-1, 1)
    return normalized_vectors


def test_kmeans():
    data_size = 10000
    dimension = 256
    clusters = 16
    vectors = data_load(data_size, dimension)
    centroids = vsag.kmeans(vectors, clusters)
    distances = cdist(vectors, centroids)
    min_idx = np.argmin(distances, axis=1)
    counts = np.zeros(centroids.shape[0], dtype=np.int)
    for idx in min_idx:
        counts[idx] += 1
    print(counts)


