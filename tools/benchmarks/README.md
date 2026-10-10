# Per-vector SQ8 comparison

Build `sq8_per_vector_benchmark` with `VSAG_ENABLE_TOOLS=ON make release COMPILE_JOBS=32`.
Install `h5py`, `numpy`, and `matplotlib` in an isolated Python environment. Obtain the
public [GIST-960 Euclidean HDF5 dataset](https://ann-benchmarks.com/gist-960-euclidean.hdf5)
and run:

```sh
python tools/benchmarks/sq8_per_vector.py \
  --dataset /path/to/gist-960-euclidean.hdf5 \
  --artifacts /path/to/results --cpu 0
```

Choose an available CPU and ensure no competing workload runs during measurement. The runner
requires all 1,000,000 database vectors and 1,000 official queries, without preprocessing.
It validates HDF5 shapes and ground-truth indices and records the source URL, byte count and
SHA-256. It writes about 4 GB of extracted inputs plus the two serialized indexes.

The comparison uses HGraph L2 with `max_degree=32`, `ef_construction=200`, 16 build threads,
`use_reorder=false`, and the built-in graph level seed 2021. Baseline `sq8` training is unchanged:
a deterministic stride sample of up to 100,000 vectors. Parallel construction may vary with
scheduling, and each quantizer constructs its own graph. This measures the full index behavior,
not distances on a shared graph.

Search runs on one pinned thread. At each common `ef_search`, one full-query warmup precedes
three measured passes. The monotonic search wall clock excludes loading, construction,
serialization, result validation, recall computation and plotting. QPS counts successful queries;
recall retains the full query denominator. Failed/short queries remain in diagnostics. Raw CSV,
median and min/max QPS, matched-recall interpolation (no extrapolation), build time, index memory,
serialized bytes, build/search JSON and PNG/SVG curves are written to the artifact directory.
The observed maximum recall is limited to the measured sweep; it is not an exact quantizer ceiling.

See the [English](../../docs/docs/en/src/quantization/sq.md) and
[Chinese](../../docs/docs/zh/src/quantization/sq.md) quantization documentation, and the
[evaluation measurement conventions](../../docs/docs/en/src/resources/eval.md).

If more search effort is needed, reuse the saved indexes and append new common points with
`--search-only --ef 1600,3200` and the same other arguments. Do not repeat existing `ef_search`
values in an extension. `--plot-only` regenerates summaries and plots from raw CSV.
