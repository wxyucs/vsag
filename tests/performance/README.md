# test performance tool

usage:
```
Usage: ./build/tests/test_performance <dataset_file_path> <index_name> <build_param> <search_param>
```

example running commands:
```
./build/tests/test_performance \
    '/data/random-100k-128-euclidean.hdf5' \
	'hnsw' \
	'{"dim": 128, "dtype": "float32", "metric_type": "l2", "hnsw": {"max_degree": 12, "ef_construction": 100}, "diskann": {"max_degree": 12, "ef_construction": 100, "pq_dims": 64, "pq_sample_rate": 0.1}}' \
	'{"hnsw":{"ef_search":100},"diskann":{"ef_search":100,"beam_search":4,"io_limit":200,"use_reorder":true}}'
```

example output:
```
{
    "build_parameters": "{\"dim\": 128, \"dtype\": \"float32\", \"metric_type\": \"l2\", \"hnsw\": {\"max_degree\": 12, \"ef_construction\": 100}, \"diskann\": {\"max_degree\": 12, \"ef_construction\": 100, \"pq_dims\": 64, \"pq_sample_rate\": 0.1}}",
    "build_time_in_second": 43.442704397,
    "correct": 3984,
    "dataset": "/data/random-100k-128-euclidean.hdf5",
    "index_name": "hnsw",
    "qps": 2507.0343699079094,
    "recall": 0.3984000086784363,
    "search_parameters": "{\"hnsw\":{\"ef_search\":100},\"diskann\":{\"ef_search\":100,\"beam_search\":4,\"io_limit\":200,\"use_reorder\":true}}",
    "search_time_in_second": 3.988776588,
    "tps": 230.18824768861685
}
```

json escape website: https://www.sojson.com/yasuo.html
