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

"""Tests for :class:`IndexRegistry` save/load + LRU eviction.

We mock ``pyvsag`` with a stand-in :class:`_FakeIndex` so the suite runs
on macOS / CI without a built VSAG binary. The save / load contract we
exercise matches the real pyvsag API: ``save(prefix) -> dict[str, int]``
and ``load(prefix, file_sizes, owns)``.
"""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import pytest


class _FakeIndex:
    """Minimal stand-in for :class:`pyvsag.Index`.

    ``save`` writes a single payload file (so the registry's ``payload/``
    layout is exercised) and returns a ``file_sizes`` dict; ``load``
    reads that file back and stores its bytes for assertions.
    """

    def __init__(self, algorithm, params_json):
        self.algorithm = algorithm
        self.params = json.loads(params_json)
        self.payload_bytes = b""
        self.loaded_from = None

    def build(self, **kwargs):
        self.payload_bytes = b"BUILT-" + self.algorithm.encode()

    def save(self, prefix: str):
        path = Path(prefix + ".bin")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.payload_bytes)
        return {"main": len(self.payload_bytes)}

    def load(self, prefix: str, file_sizes=None, owns=False):
        path = Path(prefix + ".bin")
        self.payload_bytes = path.read_bytes()
        self.loaded_from = prefix
        self.file_sizes = file_sizes


@pytest.fixture
def fake_pyvsag(monkeypatch):
    mod = types.ModuleType("pyvsag")
    mod.Index = _FakeIndex  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyvsag", mod)
    return mod


def _make_entry(handle: str = "hgraph-idx-1"):
    from vsag_code.tools.registry_index import IndexEntry

    live = _FakeIndex("hgraph", json.dumps({"hgraph": {"max_degree": 26}}))
    live.build()
    return IndexEntry(
        handle=handle,
        algorithm="hgraph",
        metric="l2",
        dim=128,
        num_elements=10,
        params={"hgraph": {"max_degree": 26}, "metric_type": "l2", "dim": 128},
        dataset_path=None,
        created_at=time.time(),
        last_used=time.time(),
        live=live,
    )


def test_register_then_get_returns_same_live(vsag_code_home, fake_pyvsag):
    from vsag_code.tools.registry_index import IndexRegistry

    reg = IndexRegistry(max_in_memory=4)
    entry = _make_entry()
    reg.register(entry)

    got = reg.get(entry.handle)
    assert got is not None
    assert got.live is entry.live
    assert got.dim == 128


def test_eviction_persists_and_reloads(vsag_code_home, fake_pyvsag):
    from vsag_code.tools.registry_index import IndexRegistry

    reg = IndexRegistry(max_in_memory=1)
    a = _make_entry("hgraph-idx-1")
    b = _make_entry("hgraph-idx-2")
    reg.register(a)
    reg.register(b)
    # `a` should have been evicted (cap=1).
    assert reg._entries["hgraph-idx-1"].live is None  # type: ignore[attr-defined]
    assert reg._entries["hgraph-idx-2"].live is not None  # type: ignore[attr-defined]

    revived = reg.get("hgraph-idx-1")
    assert revived is not None
    assert revived.live is not None
    assert revived.live.payload_bytes.startswith(b"BUILT-")


def test_remove_deletes_disk_payload(vsag_code_home, fake_pyvsag):
    from vsag_code.tools.registry_index import IndexRegistry

    reg = IndexRegistry()
    e = _make_entry()
    reg.register(e)

    on_disk = vsag_code_home / "indexes" / e.handle
    assert on_disk.is_dir()
    assert (on_disk / "meta.json").is_file()

    assert reg.remove(e.handle) is True
    assert not on_disk.exists()
    assert reg.remove(e.handle) is False


def test_reload_known_handles_from_fresh_registry(vsag_code_home, fake_pyvsag):
    from vsag_code.tools.registry_index import IndexRegistry

    reg1 = IndexRegistry()
    reg1.register(_make_entry("hgraph-idx-7"))

    reg2 = IndexRegistry()
    listed = [e.handle for e in reg2.list()]
    assert "hgraph-idx-7" in listed
