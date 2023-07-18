#include "utils.h"
#include "vsag/vsag.h"

using namespace std;

float*
generate(int num_elements, int dim) {
    mt19937 rng;
    rng.seed(47);
    uniform_real_distribution<> distrib_real;
    float* data = new float[dim * num_elements];
    for (int i = 0; i < dim * num_elements; i++) {
        data[i] = distrib_real(rng);
    }

    return data;
}

int
main() {
    int dim = 128;
    int max_elements = 1000000;
    int M = 16;

    int ef_construction = 200;

    // Initing index
    vsag::HNSW hnsw(dim, max_elements, M, ef_construction);

    float* data = generate(max_elements, dim);

    // Add data to index
    int duration_add;
    {
        time_recorder tr(duration_add);
        for (int i = 0; i < max_elements; i++) {
            hnsw.addPoint(data + i * dim, i);
            print_process(i, max_elements, "adding ... ");
        }
        cout << endl;
    }
    cout << "add cost: " << duration_add << "ms" << endl;

    // Query the elements for themselves and measure recall
    float correct = 0;
    int duration_search;
    {
        time_recorder tr(duration_search);
        for (int i = 0; i < max_elements; i++) {
            priority_queue<pair<float, hnswlib::labeltype>> result =
                hnsw.searchTopK(data + i * dim, 1);
            hnswlib::labeltype label = result.top().second;
            if (label == i)
                correct++;
            print_process(i, max_elements, "searching ... ");
        }
        cout << endl;
    }
    cout << "search cost: " << duration_search << "ms" << endl;

    float recall = correct / max_elements;
    cout << "Recall: " << recall << endl;
    float qps = 1.0f * max_elements / duration_search * 1000;
    cout << "qps: " << qps << endl;

    return 0;
}
