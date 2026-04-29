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

"""High-level retrieval helper used by the ``docs_search`` tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .store import DEFAULT_STORE_DIR, DocStore


_CACHE: Dict[str, DocStore] = {}


def _open_cached(directory: str) -> DocStore:
    store = _CACHE.get(directory)
    if store is None:
        store = DocStore.open(Path(directory))
        _CACHE[directory] = store
    return store


def retrieve(
    query: str,
    k: int = 5,
    *,
    store_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return up to ``k`` (score, chunk) hits as JSON-serializable dicts."""
    directory = store_dir or DEFAULT_STORE_DIR
    store = _open_cached(directory)
    hits = store.search(query, k=k)
    out: List[Dict[str, Any]] = []
    for score, chunk in hits:
        out.append(
            {
                "score": round(score, 4),
                "source": chunk.source,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "title": chunk.title,
                "text": chunk.text,
            }
        )
    return out


__all__ = ["retrieve"]
