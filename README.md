# VSAG (Vector Search for AntGroup)

## Usage
```bash
# compile this project with debug mode
$ make debug

# compile this project with release mode
$ make release

# compile and run unittests
$ make test

# compile and run benchmarks
$ make benchmark

# format cpp changes
$ make format

# clean all compilation results
$ make clean

# extract dataset from base and query csv files
$ python3 benchs/csv_extract.py --base-csv /path/to/base.csv \
	--base-id-name id --base-vector-name feature \
	--query-csv /path/to/query.csv \
	--query-id-name id --query-vector-name feature \
	--output /path/to/output.h5 --overwrite
```

## Roadmap
- ~~实现包含 HNSW 构建和搜索的功能（v0.1）~~
  - ~~设计向量接口（c++）~~
  - ~~集成 HNSW 库~~
  - ~~编写 example 代码~~
  - ~~编写 bench 代码~~
- ~~支持 INT8 向量类型和距离算法（v0.2）~~
  - ~~实现类型扩展~~
  - ~~编写 HNSW 的集成测试~~
  - ~~编写 INT8 类型的 bench 代码~~
- ~~提供 k-means 聚类算法（v0.3）~~
  - ~~集成 k-means 聚类算法~~
  - ~~扩展 k-means 聚类算法支持 INT8 类型~~
  - ~~编写 k-means 的集成测试~~
- 提供 python 接口的 HNSW 和 k-means（v0.4）
  - 使用 pybind11 导出 python 接口
  - 编写召回率/QPS的 benchmark 代码
- 编写基于 HDF5 的 python 接口 benchmark 工具（v0.5）
  - 编写基于 HDF5 的 benchmark 接口
  - 实现 benchmark 过程和报告输出
  - [optional] 改进性能
- 迁移分区实验到项目中

