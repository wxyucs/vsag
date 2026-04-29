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

"""Index lifecycle + search tools.

These tools all flow through :class:`IndexRegistry` so handles outlive a
single tool call (and even a single agent session, since the registry
persists to disk under ``$VSAG_CODE_HOME``).

Tools:

* ``index_build``   build a pyvsag index from an HDF5 dataset (mutate)
* ``index_search``  KNN search + recall@k against ground truth   (read)
* ``index_stats``   metadata + memory-ish summary for a handle   (read)
* ``index_list``    enumerate registered handles                 (read)
* ``index_remove``  drop an index from disk + memory (destructive)
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .errors import ToolError
from .registry import (
    TIER_DESTRUCTIVE,
    TIER_MUTATE,
    TIER_READ,
    ToolSpec,
    register,
)
from .registry_index import IndexEntry, get_registry


# Conservative defaults known to give >=90% recall on SIFT-1M; refined later
# via benchs/. Spike numbers (RESULTS.md) confirm 0.95+ on hgraph defaults.
_ALGO_PRESETS: Dict[str, Dict[str, Any]] = {
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


# ---------------------------------------------------------------------------
# index_build
# ---------------------------------------------------------------------------


def index_build(
    dataset_path: str,
    algorithm: str = "hgraph",
    metric: Optional[str] = None,
    index_param: Optional[Dict[str, Any]] = None,
    num_elements: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a pyvsag index from a dataset's training partition.

    The built index is registered with the process registry; the returned
    ``handle`` is what subsequent ``index_*`` tools accept.
    """
    import h5py
    import numpy as np
    import pyvsag  # type: ignore

    # Validate via shared paths (algorithm + dataset before we read GB of data).
    if algorithm not in _ALGO_PRESETS:
        raise ToolError(
            "unsupported",
            f"unsupported algorithm: {algorithm}",
            f"Use one of: {sorted(_ALGO_PRESETS)}.",
        )

    # Lean on dataset_info to validate path + extract metadata.
    from .dataset_tools import dataset_info  # late import: avoid circular
    info = dataset_info(dataset_path)

    if metric is None:
        metric = {"euclidean": "l2", "angular": "ip"}.get(info["distance"], "l2")
    if metric not in ("l2", "ip"):
        raise ToolError(
            "invalid_argument",
            f"unknown metric: {metric}",
            "Valid metrics are 'l2' or 'ip'.",
        )

    dim = info["dim"]
    n_total = info["train_n"]
    requested = num_elements if num_elements is not None else n_total
    if requested <= 0:
        raise ToolError(
            "invalid_argument",
            f"num_elements must be positive, got {requested}",
            "Omit num_elements to use the entire training set.",
        )
    n = min(requested, n_total)

    effective = dict(_ALGO_PRESETS[algorithm])
    if index_param:
        effective.update(index_param)

    params_obj: Dict[str, Any] = {
        "dtype": "float32",
        "metric_type": metric,
        "dim": dim,
        algorithm: effective,
    }
    params_json = json.dumps(params_obj)

    with h5py.File(dataset_path, "r") as f:
        vectors = np.asarray(f["train"][:n], dtype=np.float32)
    ids = list(range(n))

    t0 = time.perf_counter()
    index = pyvsag.Index(algorithm, params_json)
    index.build(vectors=vectors, ids=ids, num_elements=n, dim=dim)
    build_seconds = time.perf_counter() - t0

    registry = get_registry()
    handle = registry.new_handle(algorithm)
    entry = IndexEntry(
        handle=handle,
        algorithm=algorithm,
        metric=metric,
        dim=int(dim),
        num_elements=int(n),
        params=params_obj,
        dataset_path=dataset_path,
        created_at=time.time(),
        last_used=time.time(),
        live=index,
    )
    registry.register(entry)

    return {
        "handle": handle,
        "algorithm": algorithm,
        "metric": metric,
        "dim": int(dim),
        "num_elements": int(n),
        "build_seconds": round(build_seconds, 3),
        "params": params_obj,
    }


# ---------------------------------------------------------------------------
# index_search
# ---------------------------------------------------------------------------


def _percentile(sorted_values: List[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, min(len(sorted_values) - 1, int(round(p * len(sorted_values))) - 1))
    return float(sorted_values[idx])


def index_search(
    handle: str,
    topk: int = 10,
    ef_search: int = 100,
    num_queries: Optional[int] = None,
) -> Dict[str, Any]:
    """KNN search; reports recall@topk + latency / QPS over the test set."""
    import h5py
    import numpy as np

    if topk <= 0 or topk > 1000:
        raise ToolError(
            "invalid_argument",
            f"topk must be in 1..1000, got {topk}",
            "Use topk between 1 and 100 for typical recall studies.",
        )

    registry = get_registry()
    entry = registry.get(handle)
    if entry is None:
        raise ToolError(
            "not_found",
            f"unknown handle: {handle}",
            "Use index_list to see available handles, or index_build to make one.",
        )
    if entry.dataset_path is None:
        raise ToolError(
            "invalid_argument",
            f"handle {handle!r} has no associated dataset_path",
            "Search is only supported for indexes built via index_build.",
        )

    with h5py.File(entry.dataset_path, "r") as f:
        test = np.asarray(f["test"], dtype=np.float32)
        gt = np.asarray(f["neighbors"], dtype=np.int64)

    q_total = test.shape[0]
    q_req = num_queries if num_queries is not None else q_total
    if q_req <= 0:
        raise ToolError(
            "invalid_argument",
            f"num_queries must be positive, got {q_req}",
            "Omit num_queries to evaluate on the full test partition.",
        )
    q = min(q_req, q_total)
    test = test[:q]
    gt = gt[:q]

    search_params = json.dumps({entry.algorithm: {"ef_search": ef_search}})

    hits = 0
    total_gt = 0
    latencies_ms: List[float] = []
    t0 = time.perf_counter()
    for i in range(q):
        qs = time.perf_counter()
        result_ids, _ = entry.live.knn_search(
            vector=test[i], k=topk, parameters=search_params
        )
        latencies_ms.append((time.perf_counter() - qs) * 1000.0)
        gt_topk = {int(x) for x in gt[i, :topk]}
        got = {int(x) for x in result_ids}
        hits += len(gt_topk & got)
        total_gt += len(gt_topk)
    wall = time.perf_counter() - t0

    latencies_ms.sort()
    return {
        "handle": handle,
        "topk": topk,
        "num_queries": q,
        "recall_at_k": round(hits / total_gt, 4) if total_gt else 0.0,
        "mean_latency_ms": round(sum(latencies_ms) / len(latencies_ms), 3),
        "p50_latency_ms": round(_percentile(latencies_ms, 0.50), 3),
        "p95_latency_ms": round(_percentile(latencies_ms, 0.95), 3),
        "p99_latency_ms": round(_percentile(latencies_ms, 0.99), 3),
        "qps": round(q / wall, 1) if wall > 0 else 0.0,
        "ef_search": ef_search,
    }


# ---------------------------------------------------------------------------
# index_stats / index_list / index_remove
# ---------------------------------------------------------------------------


def index_stats(handle: str) -> Dict[str, Any]:
    """Return metadata + a coarse memory estimate for a registered index."""
    entry = get_registry().get(handle)
    if entry is None:
        raise ToolError(
            "not_found",
            f"unknown handle: {handle}",
            "Use index_list to see available handles.",
        )
    # Coarse estimate: f32 vectors + per-vector graph overhead. Real numbers
    # come from index.analyze in a later stage.
    vec_bytes = entry.num_elements * entry.dim * 4
    graph_overhead = entry.num_elements * 64
    return {
        "handle": entry.handle,
        "algorithm": entry.algorithm,
        "metric": entry.metric,
        "dim": entry.dim,
        "num_elements": entry.num_elements,
        "params": entry.params,
        "dataset_path": entry.dataset_path,
        "created_at": entry.created_at,
        "last_used": entry.last_used,
        "estimated_memory_bytes": int(vec_bytes + graph_overhead),
        "live": entry.live is not None,
    }


def index_list() -> Dict[str, Any]:
    """Enumerate registered indexes (in-memory and on-disk)."""
    items: List[Dict[str, Any]] = []
    for entry in get_registry().list():
        items.append(
            {
                "handle": entry.handle,
                "algorithm": entry.algorithm,
                "dim": entry.dim,
                "num_elements": entry.num_elements,
                "metric": entry.metric,
                "live": entry.live is not None,
                "last_used": entry.last_used,
            }
        )
    return {"count": len(items), "indexes": items}


def index_remove(handle: str) -> Dict[str, Any]:
    """Drop an index from memory AND its on-disk payload (destructive)."""
    ok = get_registry().remove(handle)
    if not ok:
        raise ToolError(
            "not_found",
            f"unknown handle: {handle}",
            "Use index_list to see available handles.",
        )
    return {"handle": handle, "removed": True}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


register(
    ToolSpec(
        name="index_build",
        description=(
            "Build an in-memory VSAG index from the training partition of an "
            "HDF5 dataset. Returns an opaque handle for later index_search / "
            "index_stats calls. Defaults to algorithm='hgraph' which is the "
            "current best general-purpose graph index."
        ),
        parameters={
            "type": "object",
            "properties": {
                "dataset_path": {"type": "string"},
                "algorithm": {
                    "type": "string",
                    "enum": list(_ALGO_PRESETS.keys()),
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
                    "minimum": 1,
                    "description": "Cap on training rows; default uses all.",
                },
            },
            "required": ["dataset_path"],
        },
        fn=index_build,
        tier=TIER_MUTATE,
    )
)


register(
    ToolSpec(
        name="index_search",
        description=(
            "Run KNN search on an index built earlier and report recall@topk "
            "against the dataset ground truth, plus mean / p50 / p95 / p99 "
            "latency and QPS. The index handle must come from index_build."
        ),
        parameters={
            "type": "object",
            "properties": {
                "handle": {"type": "string"},
                "topk": {"type": "integer", "default": 10, "minimum": 1, "maximum": 1000},
                "ef_search": {"type": "integer", "default": 100, "minimum": 1},
                "num_queries": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Cap on test queries; default uses all.",
                },
            },
            "required": ["handle"],
        },
        fn=index_search,
        tier=TIER_READ,
    )
)


register(
    ToolSpec(
        name="index_stats",
        description=(
            "Return metadata (algorithm, dim, num_elements, params, dataset, "
            "create / last-used timestamps) plus a coarse memory estimate "
            "for a registered index handle."
        ),
        parameters={
            "type": "object",
            "properties": {"handle": {"type": "string"}},
            "required": ["handle"],
        },
        fn=index_stats,
        tier=TIER_READ,
    )
)


register(
    ToolSpec(
        name="index_list",
        description=(
            "Enumerate every index handle the agent currently knows about, "
            "including ones that were evicted from memory but persist on "
            "disk and can be reloaded on demand."
        ),
        parameters={"type": "object", "properties": {}, "required": []},
        fn=index_list,
        tier=TIER_READ,
    )
)


register(
    ToolSpec(
        name="index_remove",
        description=(
            "Permanently delete an index handle and its on-disk payload. "
            "This is irreversible; the agent must confirm with the user "
            "before calling it on a non-empty index."
        ),
        parameters={
            "type": "object",
            "properties": {"handle": {"type": "string"}},
            "required": ["handle"],
        },
        fn=index_remove,
        tier=TIER_DESTRUCTIVE,
    )
)
