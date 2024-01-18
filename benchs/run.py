#!/usr/bin/env python3
import yaml
from benchmark import bench_index


if __name__ == '__main__':
    with open('benchs/run.yaml', 'r') as f:
        config = yaml.safe_load(f)

    bench_index.run(config)