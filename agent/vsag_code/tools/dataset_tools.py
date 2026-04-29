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

"""Dataset-inspection tools.

Currently:

* ``dataset_info``  inspect an ann-benchmarks HDF5 file.
* ``dataset_list``  list candidate datasets under ``$VSAG_DATASETS``.
* ``dataset_peek``  return a small head-slice of vectors for sanity checks.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .errors import ToolError
from .registry import TIER_READ, ToolSpec, register


def _datasets_root() -> Path:
    return Path(os.environ.get("VSAG_DATASETS", "/data/datasets"))


def dataset_list(prefix: Optional[str] = None) -> Dict[str, Any]:
    """List ``*.hdf5`` datasets under ``$VSAG_DATASETS`` (default ``/data/datasets``)."""
    root = _datasets_root()
    if not root.is_dir():
        return {"root": str(root), "datasets": []}
    items: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.hdf5")):
        name = path.name
        if prefix and not name.startswith(prefix):
            continue
        items.append({"name": name, "path": str(path), "size_bytes": path.stat().st_size})
    return {"root": str(root), "datasets": items}


def dataset_info(path: str) -> Dict[str, Any]:
    """Inspect an HDF5 ann-benchmarks dataset.

    Returns a dict with ``train_n``, ``test_n``, ``dim``, ``dtype``,
    ``distance``, plus shape tuples and the file size in bytes.
    """
    import h5py  # noqa: WPS433  (deferred import keeps cold start cheap)

    p = Path(path)
    if not p.is_file():
        raise ToolError(
            "not_found",
            f"file not found: {path}",
            "Run dataset_list to see available files under $VSAG_DATASETS.",
        )

    out: Dict[str, Any] = {"path": str(p), "size_bytes": p.stat().st_size}
    with h5py.File(p, "r") as f:
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
        attr = f.attrs.get("distance")
        if attr is not None:
            out["distance"] = attr.decode() if isinstance(attr, bytes) else str(attr)
        else:
            stem = p.stem
            tail = stem.rsplit("-", 1)[-1] if "-" in stem else ""
            out["distance"] = tail or "unknown"
    return out


def dataset_peek(path: str, n: int = 3, partition: str = "train") -> Dict[str, Any]:
    """Return the first ``n`` rows of ``train``/``test`` for sanity-check use.

    The vectors are returned as plain Python lists (rounded to 4 decimals) so
    they survive JSON serialization without exploding the LLM context.
    """
    import h5py
    import numpy as np

    if partition not in ("train", "test"):
        raise ToolError(
            "invalid_argument",
            f"partition must be 'train' or 'test', got {partition!r}",
            "Pass partition='train' (default) or partition='test'.",
        )
    if n <= 0 or n > 16:
        raise ToolError(
            "invalid_argument",
            f"n must be in 1..16, got {n}",
            "Peek is for spot checks; use index_search for real evaluations.",
        )
    p = Path(path)
    if not p.is_file():
        raise ToolError("not_found", f"file not found: {path}", "")
    with h5py.File(p, "r") as f:
        rows = np.asarray(f[partition][: int(n)], dtype=np.float32)
    return {
        "path": str(p),
        "partition": partition,
        "n": int(rows.shape[0]),
        "dim": int(rows.shape[1]),
        "rows": [[round(float(x), 4) for x in row] for row in rows],
    }


# ---------------------------------------------------------------------------
# ToolSpec registration
# ---------------------------------------------------------------------------


register(
    ToolSpec(
        name="dataset_list",
        description=(
            "List HDF5 vector datasets available under $VSAG_DATASETS "
            "(default /data/datasets). Optional 'prefix' filter."
        ),
        parameters={
            "type": "object",
            "properties": {
                "prefix": {
                    "type": "string",
                    "description": "Filter to dataset names starting with this prefix.",
                },
            },
            "required": [],
        },
        fn=dataset_list,
        tier=TIER_READ,
    )
)


register(
    ToolSpec(
        name="dataset_info",
        description=(
            "Inspect an ann-benchmarks HDF5 vector dataset. Returns train_n, "
            "test_n, dim, dtype, distance metric, and file size."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Filesystem path to the .hdf5 file.",
                },
            },
            "required": ["path"],
        },
        fn=dataset_info,
        tier=TIER_READ,
    )
)


register(
    ToolSpec(
        name="dataset_peek",
        description=(
            "Return the first n rows of the 'train' or 'test' partition for "
            "spot-checking. n is capped at 16. Use index_search for real "
            "evaluation; this is purely diagnostic."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "n": {"type": "integer", "default": 3, "minimum": 1, "maximum": 16},
                "partition": {
                    "type": "string",
                    "enum": ["train", "test"],
                    "default": "train",
                },
            },
            "required": ["path"],
        },
        fn=dataset_peek,
        tier=TIER_READ,
    )
)
