#  Copyright 2024-present the vsag project
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Stage-1 spike tools.

Three minimal tools the LLM can drive:

* ``dataset.info``  inspect an HDF5 ann-benchmarks dataset
* ``index.build``   build a pyvsag index in-memory from a dataset's training set
* ``index.search``  run KNN on the test set, compute recall@k against ground truth

All tools are pure Python with stable JSON-friendly inputs / outputs.
They MUST NOT depend on the LLM client; the spike driver imports and calls
them, and the tests exercise them stand-alone.

The dataset format is the ann-benchmarks HDF5 layout:
``train`` (N, dim), ``test`` (Q, dim), ``neighbors`` (Q, K_gt), ``distances`` (Q, K_gt).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# Cache built indexes by handle id so subsequent search calls can reuse them
# within a single spike run. The LLM-driven loop will only see opaque ids.
_INDEX_REGISTRY: Dict[str, Any] = {}
_NEXT_HANDLE: List[int] = [0]


def _new_handle(prefix: str) -> str:
    _NEXT_HANDLE[0] += 1
    return f"{prefix}-{_NEXT_HANDLE[0]}"


# ---------------------------------------------------------------------------
# dataset.info
# ---------------------------------------------------------------------------


def dataset_info(path: str) -> Dict[str, Any]:
    """Inspect an HDF5 ann-benchmarks dataset.

    Args:
        path: filesystem path to the .hdf5 file.

    Returns:
        Dict with keys: ``path``, ``train_shape``, ``test_shape``,
        ``neighbors_shape``, ``dim``, ``train_n``, ``test_n``, ``dtype``,
        ``distance``, ``size_bytes``.
    """
    import os

    import h5py

    if not os.path.isfile(path):
        return {"error": f"file not found: {path}"}

    out: Dict[str, Any] = {"path": path, "size_bytes": os.path.getsize(path)}
    with h5py.File(path, "r") as f:
        train = f["train"]
        test = f["test"]
        neighbors = f.get("neighbors")
        out["train_shape"] = list(train.shape)
        out["test_shape"] = list(test.shape)
        out["neighbors_shape"] = list(neighbors.shape) if neighbors is not None else None
        out["dim"] = int(train.shape[1])
        out["train_n"] = int(train.shape[0])
        out["test_n"] = int(test.shape[0])
        out["dtype"] = str(train.dtype)
        # ann-benchmarks encodes the metric in the filename (sift-128-euclidean,
        # glove-100-angular). Be conservative and report what the file says
        # only if explicit; otherwise infer from filename suffix.
        attr_distance = f.attrs.get("distance")
        if attr_distance is not None:
            out["distance"] = (
                attr_distance.decode() if isinstance(attr_distance, bytes) else str(attr_distance)
            )
        else:
            stem = os.path.basename(path).removesuffix(".hdf5")
            tail = stem.rsplit("-", 1)[-1] if "-" in stem else ""
            out["distance"] = tail or "unknown"
    return out


# ---------------------------------------------------------------------------
# index.build
# ---------------------------------------------------------------------------


_ALGO_PRESETS: Dict[str, Dict[str, Any]] = {
    # Conservative defaults known to give >=90% recall on SIFT-1M; refined later
    # via benchs/ in stage 4. Spike does not need optimal params, only working
    # ones.
    "hgraph": {
        "base_quantization_type": "sq8",
        "max_degree": 26,
        "ef_construction": 100,
        "alpha": 1.2,
    },
    "hnsw": {
        "max_degree": 16,
        "ef_construction": 100,
    },
}


def index_build(
    dataset_path: str,
    algorithm: str = "hgraph",
    metric: Optional[str] = None,
    index_param: Optional[Dict[str, Any]] = None,
    num_elements: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a pyvsag index from a dataset's training partition.

    Args:
        dataset_path: HDF5 file (ann-benchmarks layout).
        algorithm: ``hgraph`` (default) or ``hnsw``.
        metric: ``l2`` or ``ip``; inferred from filename if omitted.
        index_param: override / merge into the per-algorithm preset.
        num_elements: cap training set size (useful for the spike to keep
            wall-clock low; default = all rows).

    Returns:
        Dict with ``handle`` (opaque id), ``algorithm``, ``metric``, ``dim``,
        ``num_elements``, ``build_seconds``, ``params`` (effective JSON).
    """
    import h5py
    import numpy as np
    import pyvsag

    if algorithm not in _ALGO_PRESETS:
        return {"error": f"unsupported algorithm: {algorithm}; have {list(_ALGO_PRESETS)}"}

    info = dataset_info(dataset_path)
    if "error" in info:
        return info

    if metric is None:
        # ann-benchmarks "euclidean" -> l2, "angular" -> ip
        metric = {"euclidean": "l2", "angular": "ip"}.get(info["distance"], "l2")

    dim = info["dim"]
    n_total = info["train_n"]
    n = min(num_elements or n_total, n_total)

    effective_param = dict(_ALGO_PRESETS[algorithm])
    if index_param:
        effective_param.update(index_param)

    params_obj = {
        "dtype": "float32",
        "metric_type": metric,
        "dim": dim,
        "index_param": effective_param,
    }
    params_json = json.dumps(params_obj)

    with h5py.File(dataset_path, "r") as f:
        # Slice from disk keeps RAM footprint sane for SIFT-1M (~512MB f32).
        vectors = np.asarray(f["train"][:n], dtype=np.float32)
    ids = list(range(n))

    t0 = time.perf_counter()
    index = pyvsag.Index(algorithm, params_json)
    index.build(vectors=vectors, ids=ids, num_elements=n, dim=dim)
    build_seconds = time.perf_counter() - t0

    handle = _new_handle(f"{algorithm}-idx")
    _INDEX_REGISTRY[handle] = {
        "index": index,
        "algorithm": algorithm,
        "metric": metric,
        "dim": dim,
        "num_elements": n,
        "params": params_obj,
        "dataset_path": dataset_path,
    }
    return {
        "handle": handle,
        "algorithm": algorithm,
        "metric": metric,
        "dim": dim,
        "num_elements": n,
        "build_seconds": round(build_seconds, 3),
        "params": params_obj,
    }


# ---------------------------------------------------------------------------
# index.search
# ---------------------------------------------------------------------------


def index_search(
    handle: str,
    topk: int = 10,
    ef_search: int = 100,
    num_queries: Optional[int] = None,
) -> Dict[str, Any]:
    """Run KNN search on the index's source-dataset test partition.

    Computes recall@topk against the dataset's ground-truth neighbors. Results
    are aggregated as a single mean recall, plus per-query latency stats.

    Args:
        handle: opaque id from a prior ``index.build`` call.
        topk: K for KNN.
        ef_search: search-time recall/latency tradeoff knob (hgraph/hnsw).
        num_queries: cap query set size; default = all test rows.

    Returns:
        Dict with ``handle``, ``topk``, ``num_queries``, ``recall_at_k``,
        ``mean_latency_ms``, ``p99_latency_ms``, ``qps``.
    """
    import h5py
    import numpy as np

    if handle not in _INDEX_REGISTRY:
        return {"error": f"unknown handle: {handle}"}

    rec = _INDEX_REGISTRY[handle]
    index = rec["index"]
    algorithm = rec["algorithm"]

    with h5py.File(rec["dataset_path"], "r") as f:
        test = np.asarray(f["test"], dtype=np.float32)
        gt = np.asarray(f["neighbors"], dtype=np.int64)

    q_total = test.shape[0]
    q = min(num_queries or q_total, q_total)
    test = test[:q]
    gt = gt[:q]

    # vsag accepts per-algorithm search params under a top-level key matching
    # the index name (see examples/python/103_index_hgraph.py).
    search_params = json.dumps({algorithm: {"ef_search": ef_search}})

    hits = 0
    total_gt = 0
    latencies_ms: List[float] = []
    t0 = time.perf_counter()
    for i in range(q):
        qs = time.perf_counter()
        result_ids, _ = index.knn_search(vector=test[i], k=topk, parameters=search_params)
        latencies_ms.append((time.perf_counter() - qs) * 1000.0)
        gt_topk = set(int(x) for x in gt[i, :topk])
        got = set(int(x) for x in result_ids)
        hits += len(gt_topk & got)
        total_gt += len(gt_topk)
    wall = time.perf_counter() - t0

    latencies_ms.sort()
    p99 = latencies_ms[max(0, int(len(latencies_ms) * 0.99) - 1)] if latencies_ms else 0.0
    return {
        "handle": handle,
        "topk": topk,
        "num_queries": q,
        "recall_at_k": round(hits / total_gt, 4) if total_gt else 0.0,
        "mean_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 3),
        "p99_latency_ms": round(p99, 3),
        "qps": round(q / wall, 1) if wall > 0 else 0.0,
        "ef_search": ef_search,
    }


# ---------------------------------------------------------------------------
# Tool registry: exposed to the LLM via OpenAI-style function specs.
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Any = field(repr=False)


TOOLS: List[ToolSpec] = [
    ToolSpec(
        name="dataset_info",
        description=(
            "Inspect an HDF5 vector dataset (ann-benchmarks layout). "
            "Returns shape, dim, dtype, distance metric, and file size."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Filesystem path to .hdf5 file"}
            },
            "required": ["path"],
        },
        fn=dataset_info,
    ),
    ToolSpec(
        name="index_build",
        description=(
            "Build an in-memory VSAG index from the training partition of an "
            "HDF5 dataset. Returns an opaque handle for later search calls."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset_path": {"type": "string"},
                "algorithm": {
                    "type": "string",
                    "enum": ["hgraph", "hnsw"],
                    "default": "hgraph",
                },
                "metric": {
                    "type": "string",
                    "enum": ["l2", "ip"],
                    "description": "Distance metric; inferred from filename if omitted.",
                },
                "index_param": {
                    "type": "object",
                    "description": "Optional override of algorithm-specific build params.",
                },
                "num_elements": {
                    "type": "integer",
                    "description": "Cap on training rows; default uses all.",
                },
            },
            "required": ["dataset_path"],
        },
        fn=index_build,
    ),
    ToolSpec(
        name="index_search",
        description=(
            "Run KNN search against an index built earlier and report recall@k "
            "against the dataset ground truth, plus latency / qps."
        ),
        parameters={
            "type": "object",
            "properties": {
                "handle": {"type": "string", "description": "From index_build"},
                "topk": {"type": "integer", "default": 10},
                "ef_search": {"type": "integer", "default": 100},
                "num_queries": {
                    "type": "integer",
                    "description": "Cap on test queries; default uses all.",
                },
            },
            "required": ["handle"],
        },
        fn=index_search,
    ),
]


def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a tool call by name. Used by the LLM driver."""
    for spec in TOOLS:
        if spec.name == name:
            return spec.fn(**arguments)
    return {"error": f"unknown tool: {name}"}
