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

"""On-disk RAG store: chunk metadata as JSON, embeddings as ``.npy``.

The store is intentionally trivial: a single directory with a
``manifest.json`` (one chunk per entry) and a parallel ``embeddings.npy``
matrix of shape ``(n_chunks, dim)``. Loading is O(file size); search is
a numpy mat-vec. Good enough for ~10k chunks on docs / examples.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

DEFAULT_STORE_DIR = "/workspace/.vsag-code/rag"
DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class DocChunk:
    """One retrievable chunk: source path, line span, plain text."""

    source: str
    start_line: int
    end_line: int
    text: str
    title: str = ""


class DocStore:
    """Persistent corpus + embedding matrix.

    The store survives across processes; ``ingest_repo`` writes it once,
    and the ``docs_search`` tool re-opens it on every call. The embedder
    is loaded lazily so import-time cost is zero when no RAG query runs.
    """

    def __init__(
        self,
        directory: Path,
        *,
        embed_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        self.directory = Path(directory)
        self.embed_model = embed_model
        self._chunks: List[DocChunk] = []
        self._matrix: Optional[np.ndarray] = None
        self._encoder = None  # lazy

    # -- lifecycle --------------------------------------------------------

    @classmethod
    def open(cls, directory: Path) -> "DocStore":
        store = cls(directory)
        manifest = store.directory / "manifest.json"
        embeds = store.directory / "embeddings.npy"
        if not manifest.is_file() or not embeds.is_file():
            raise FileNotFoundError(
                f"RAG store not found at {directory!s}; run `ingest_repo` first."
            )
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        store.embed_model = meta.get("embed_model", DEFAULT_EMBED_MODEL)
        store._chunks = [DocChunk(**c) for c in meta["chunks"]]
        store._matrix = np.load(embeds)
        if store._matrix.shape[0] != len(store._chunks):
            raise RuntimeError(
                f"corrupt store: {store._matrix.shape[0]} embeddings vs "
                f"{len(store._chunks)} chunks"
            )
        return store

    def save(self) -> None:
        if self._matrix is None:
            raise RuntimeError("nothing to save: matrix is None")
        self.directory.mkdir(parents=True, exist_ok=True)
        manifest = {
            "embed_model": self.embed_model,
            "chunks": [asdict(c) for c in self._chunks],
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(self.directory / "embeddings.npy", self._matrix)

    # -- ingest -----------------------------------------------------------

    def populate(self, chunks: Iterable[DocChunk]) -> None:
        self._chunks = list(chunks)
        if not self._chunks:
            self._matrix = np.zeros((0, 384), dtype=np.float32)
            return
        encoder = self._get_encoder()
        texts = [c.text for c in self._chunks]
        emb = encoder.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._matrix = emb.astype(np.float32, copy=False)

    # -- search -----------------------------------------------------------

    def search(self, query: str, k: int = 5) -> List[tuple]:
        """Return ``[(score, chunk), ...]`` sorted by descending score."""
        if self._matrix is None or len(self._chunks) == 0:
            return []
        encoder = self._get_encoder()
        q = encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        scores = self._matrix @ q[0]
        k = max(1, min(k, len(self._chunks)))
        top = np.argpartition(-scores, k - 1)[:k]
        order = top[np.argsort(-scores[top])]
        return [(float(scores[i]), self._chunks[int(i)]) for i in order]

    # -- internals --------------------------------------------------------

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for RAG; install with "
                "`pip install vsag-code[rag]`"
            ) from exc
        self._encoder = SentenceTransformer(self.embed_model)
        return self._encoder

    @property
    def chunks(self) -> List[DocChunk]:
        return list(self._chunks)


__all__ = ["DocStore", "DocChunk", "DEFAULT_STORE_DIR", "DEFAULT_EMBED_MODEL"]
