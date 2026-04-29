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

"""Index handle registry with TTL + on-disk serialization.

The LLM only ever sees opaque handles (strings); the registry maps each
handle to:

* the live :class:`pyvsag.Index` instance (in-memory)
* the build parameters and metric/dim metadata
* a TTL (last-touched timestamp) used to evict idle entries
* a path on disk where the index is mirrored via ``Index.save`` /
  ``Index.load`` for cross-session resume

Eviction policy is simple LRU bounded by ``max_in_memory`` (default 4).
When an evicted entry is reaccessed, ``touch`` deserializes from disk.

The on-disk layout under ``$VSAG_CODE_HOME/indexes/<handle>/``:

    meta.json     # algorithm, metric, dim, num_elements, params, ts
    payload/      # whatever ``pyvsag.Index.save`` writes
    file_sizes.json  # optional, for hnsw/diskann that need it on load
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _vsag_code_home() -> Path:
    """Return ``$VSAG_CODE_HOME`` (default ``/workspace/.vsag-code``)."""
    home = Path(os.environ.get("VSAG_CODE_HOME", "/workspace/.vsag-code"))
    home.mkdir(parents=True, exist_ok=True)
    return home


def _registry_dir() -> Path:
    d = _vsag_code_home() / "indexes"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class IndexEntry:
    handle: str
    algorithm: str
    metric: str
    dim: int
    num_elements: int
    params: Dict[str, Any]
    dataset_path: Optional[str]
    created_at: float
    last_used: float
    # Live pyvsag handle; None when evicted, reload on demand.
    live: Any = field(default=None, repr=False)
    file_sizes: Optional[Dict[str, int]] = None

    def to_meta(self) -> Dict[str, Any]:
        return {
            "handle": self.handle,
            "algorithm": self.algorithm,
            "metric": self.metric,
            "dim": self.dim,
            "num_elements": self.num_elements,
            "params": self.params,
            "dataset_path": self.dataset_path,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "file_sizes": self.file_sizes,
        }


class IndexRegistry:
    """Process-local index registry with disk-backed persistence.

    Thread-safe enough for the single-threaded agent loop; concurrent
    eviction + access is serialized through a single :class:`Lock`. The
    registry is intentionally NOT a long-lived global; one instance is
    created per agent session.
    """

    def __init__(self, max_in_memory: int = 4, ttl_seconds: float = 3600.0) -> None:
        self.max_in_memory = max_in_memory
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
        self._entries: Dict[str, IndexEntry] = {}
        self._next_id = 0
        self._reload_known_handles()

    # -- handle lifecycle -------------------------------------------------

    def _reload_known_handles(self) -> None:
        """Discover persisted handles from previous sessions (lazy live load)."""
        for sub in sorted(_registry_dir().iterdir() if _registry_dir().exists() else []):
            meta_path = sub / "meta.json"
            if not meta_path.is_file():
                continue
            try:
                meta = json.loads(meta_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            handle = meta.get("handle") or sub.name
            if handle in self._entries:
                continue
            self._entries[handle] = IndexEntry(
                handle=handle,
                algorithm=meta["algorithm"],
                metric=meta["metric"],
                dim=int(meta["dim"]),
                num_elements=int(meta["num_elements"]),
                params=meta["params"],
                dataset_path=meta.get("dataset_path"),
                created_at=float(meta.get("created_at", time.time())),
                last_used=float(meta.get("last_used", time.time())),
                live=None,
                file_sizes=meta.get("file_sizes"),
            )

    def new_handle(self, algorithm: str) -> str:
        with self._lock:
            self._next_id += 1
            return f"{algorithm}-idx-{self._next_id}"

    def register(self, entry: IndexEntry) -> None:
        """Persist a new entry to disk and pin it in memory."""
        with self._lock:
            self._entries[entry.handle] = entry
            self._persist_meta(entry)
            self._save_payload(entry)
            self._maybe_evict()

    def get(self, handle: str) -> Optional[IndexEntry]:
        """Return a live entry, reloading from disk if it was evicted."""
        with self._lock:
            entry = self._entries.get(handle)
            if entry is None:
                return None
            entry.last_used = time.time()
            if entry.live is None:
                self._load_payload(entry)
            self._persist_meta(entry)
            return entry

    def remove(self, handle: str) -> bool:
        with self._lock:
            entry = self._entries.pop(handle, None)
            if entry is None:
                return False
            shutil.rmtree(_registry_dir() / handle, ignore_errors=True)
            return True

    def list(self) -> List[IndexEntry]:
        with self._lock:
            return sorted(self._entries.values(), key=lambda e: -e.last_used)

    # -- eviction ---------------------------------------------------------

    def _maybe_evict(self) -> None:
        live = [e for e in self._entries.values() if e.live is not None]
        now = time.time()
        # Evict entries past TTL first
        for e in live:
            if now - e.last_used > self.ttl_seconds:
                e.live = None
        live = [e for e in self._entries.values() if e.live is not None]
        # Then LRU-evict any over the cap
        if len(live) > self.max_in_memory:
            victims = sorted(live, key=lambda e: e.last_used)[
                : len(live) - self.max_in_memory
            ]
            for e in victims:
                e.live = None

    # -- disk i/o ---------------------------------------------------------

    def _entry_dir(self, entry: IndexEntry) -> Path:
        d = _registry_dir() / entry.handle
        (d / "payload").mkdir(parents=True, exist_ok=True)
        return d

    def _persist_meta(self, entry: IndexEntry) -> None:
        d = self._entry_dir(entry)
        (d / "meta.json").write_text(json.dumps(entry.to_meta(), indent=2))

    def _save_payload(self, entry: IndexEntry) -> None:
        if entry.live is None:
            return
        d = self._entry_dir(entry)
        target = str(d / "payload" / "index")
        try:
            file_sizes = entry.live.save(target)
        except Exception:  # pylint: disable=broad-except
            # Some algorithms may not support save in the pinned pyvsag
            # build; that's OK \u2014 the entry stays in-memory only.
            return
        if isinstance(file_sizes, dict):
            entry.file_sizes = {str(k): int(v) for k, v in file_sizes.items()}
            (d / "file_sizes.json").write_text(json.dumps(entry.file_sizes))

    def _load_payload(self, entry: IndexEntry) -> None:
        try:
            import pyvsag  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "pyvsag is not importable; reload from disk requires the "
                "C++ extension on PYTHONPATH"
            ) from exc

        params_json = json.dumps(entry.params)
        index = pyvsag.Index(entry.algorithm, params_json)
        d = self._entry_dir(entry)
        target = str(d / "payload" / "index")
        if entry.file_sizes is not None:
            index.load(target, entry.file_sizes, True)
        else:
            index.load(target)
        entry.live = index


# Module-level singleton: one process == one registry.
_REGISTRY: Optional[IndexRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> IndexRegistry:
    """Return the process-wide :class:`IndexRegistry` (lazily created)."""
    global _REGISTRY  # pylint: disable=global-statement
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = IndexRegistry()
        return _REGISTRY


def reset_registry_for_tests() -> None:
    """Test hook: drop the singleton so a fresh ``$VSAG_CODE_HOME`` is picked up."""
    global _REGISTRY  # pylint: disable=global-statement
    with _REGISTRY_LOCK:
        _REGISTRY = None
