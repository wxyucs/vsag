#!/usr/bin/env python3
"""Validate full GIST-960, run sequential SQ8 ADC curves, and plot raw results.

Requires h5py, numpy, matplotlib. Build with VSAG_ENABLE_TOOLS=ON make release.
All large files stay in --artifacts. No data is downloaded by this runner.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess

import h5py
import numpy as np


def prepare(dataset, output):
    digest = hashlib.file_digest(dataset.open('rb'), 'sha256').hexdigest()
    with h5py.File(dataset, 'r') as h5:
        assert h5['train'].shape == (1_000_000, 960)
        assert h5['test'].shape == (1_000, 960)
        assert h5['neighbors'].shape[0] == 1_000
        assert h5['neighbors'].shape[1] >= 100
        neighbors = h5['neighbors'][:]
        assert neighbors.min() >= 0 and neighbors.max() < 1_000_000
        assert all(len(set(row[:10])) == 10 for row in neighbors)
        metadata = dict(source='https://ann-benchmarks.com/gist-960-euclidean.hdf5',
                        bytes=dataset.stat().st_size, sha256=digest,
                        shapes={key: list(h5[key].shape) for key in h5})
        for key in ('train', 'test'):
            with (output / f'{key}.f32').open('wb') as stream:
                for start in range(0, len(h5[key]), 10_000):
                    block = h5[key][start:start + 10_000]
                    assert np.isfinite(block).all()
                    np.asarray(block, dtype='<f4').tofile(stream)
        np.asarray(neighbors[:, :10], dtype='<i8').tofile(output / 'neighbors.i64')
    (output / 'dataset.json').write_text(json.dumps(metadata, indent=2) + '\n')


def summarize(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    summary = []
    curves = {}
    for quantizer in ('sq8', 'sq8_per_vector'):
        with (output / f'{quantizer}-raw.csv').open() as stream:
            raw = list(csv.DictReader(stream))
        rows = []
        for ef in sorted({int(row['ef_search']) for row in raw}):
            repeats = [row for row in raw if int(row['ef_search']) == ef]
            assert len(repeats) == 3
            qps = [float(row['qps']) for row in repeats]
            recall = [float(row['recall']) for row in repeats]
            row = dict(quantizer=quantizer, ef_search=ef,
                       recall=statistics.median(recall),
                       qps=statistics.median(qps), qps_min=min(qps), qps_max=max(qps),
                       failures=sum(int(r['failed']) for r in repeats))
            rows.append(row)
            summary.append(row)
        curves[quantizer] = rows
        plt.errorbar([r['recall'] for r in rows], [r['qps'] for r in rows],
                     yerr=[[r['qps'] - r['qps_min'] for r in rows],
                           [r['qps_max'] - r['qps'] for r in rows]],
                     marker='o', label=quantizer, capsize=3)
    plt.xlabel('Recall@10')
    plt.ylabel('Queries / second (one search thread)')
    plt.title('GIST-960: HGraph, FP32-query ADC, no reranking')
    plt.grid(alpha=.3)
    plt.legend()
    plt.tight_layout()
    for extension in ('png', 'svg'):
        plt.savefig(output / f'recall-qps.{extension}', dpi=180)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    with (output / 'summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    matched = []
    for target in (.90, .95, .99):
        values = {}
        for quantizer, rows in curves.items():
            points = sorted(rows, key=lambda r: r['recall'])
            value = None
            for row in points:
                if row['recall'] == target:
                    value = max(value or 0, row['qps'])
            if value is None:
                for a, b in zip(points, points[1:]):
                    if a['recall'] < target < b['recall']:
                        value = a['qps'] + (b['qps'] - a['qps']) * (
                            target - a['recall']) / (b['recall'] - a['recall'])
                        break
            values[quantizer] = value
        matched.append(dict(recall=target, method='linear interpolation, no extrapolation',
                            qps=values))
    (output / 'matched-recall.json').write_text(json.dumps(matched, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--binary', type=Path,
                        default=Path('build-release/tools/benchmarks/sq8_per_vector_benchmark'))
    parser.add_argument('--cpu', type=int, default=0)
    parser.add_argument('--ef', default='10,20,40,80,120,200,400,800')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--prepared', action='store_true')
    parser.add_argument('--search-only', action='store_true',
                        help='Reuse both indexes and append new ef_search points')
    args = parser.parse_args()
    output = args.artifacts.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.plot_only:
        summarize(output)
        return
    if not args.prepared and not args.search_only:
        prepare(args.dataset, output)
    if args.prepare_only:
        return
    binary = args.binary.resolve()
    metadata = dict(source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(),
        dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True)),
        binary_sha256=hashlib.file_digest(binary.open('rb'), 'sha256').hexdigest(),
        library_sha256=hashlib.file_digest(
            (binary.parents[2] / 'src/libvsag.so').open('rb'), 'sha256').hexdigest(),
        machine=platform.machine(), cpu=args.cpu, affinity=sorted(os.sched_getaffinity(0)),
        hardware=subprocess.check_output(['lscpu'], text=True),
        compiler=subprocess.check_output(['g++', '--version'], text=True),
        ef_search=args.ef, warmup_passes=1, repetitions=3,
        graph_level_seed=2021, train_sample_count=1_000_000, graph_type="nsw", build_threads=16, max_degree=32, ef_construction=200,
        query_threads=1, reranking=False, preprocessing='none',
        baseline_training='unchanged deterministic stride sample, up to 100000 vectors',
        comparison='end-to-end separately constructed graphs; parallel build is not deterministic')
    suffix = '-extension' if args.search_only else ''
    (output / f'environment{suffix}.json').write_text(json.dumps(metadata, indent=2) + '\n')
    for quantizer in ('sq8', 'sq8_per_vector'):
        command = [str(binary), str(output), quantizer, str(args.cpu), args.ef]
        if args.search_only:
            command.append('search')
        (output / f'{quantizer}-invocation{suffix}.json').write_text(json.dumps(command) + '\n')
        with (output / f'{quantizer}-run{suffix}.log').open('w') as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
    summarize(output)


if __name__ == '__main__':
    main()
