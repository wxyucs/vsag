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

"""Repo ingestion: walk the VSAG checkout, chunk text, embed, persist.

Only a curated set of source globs is ingested by default (proposal
\u00a76.4): public headers, examples, benchs README, top-level design docs,
the proposal, and AGENTS.md. Implementation files under ``src/`` are
intentionally excluded -- the agent operates VSAG, it does not modify
it, and indexing all of ``src/`` would mostly add noise.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from .store import DocChunk, DocStore


# Default source globs, relative to the repo root. Tunable via
# ``ingest_repo(sources=[...])`` for tests or alternative layouts.
DEFAULT_SOURCES: Sequence[str] = (
    "include/vsag/*.h",
    "examples/cpp/*.cpp",
    "examples/python/*.py",
    "benchs/README.md",
    "docs/**/*.md",
    "VSAG_CODE_PROPOSAL.md",
    "AGENTS.md",
    "README.md",
    "DEVELOPMENT.md",
    "CONTRIBUTING.md",
)


_CHUNK_LINES = 60
_CHUNK_OVERLAP = 10


def _iter_files(repo_root: Path, sources: Sequence[str]) -> Iterable[Path]:
    seen: set = set()
    for pattern in sources:
        for path in repo_root.glob(pattern):
            if not path.is_file():
                continue
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            yield path


def _split_lines(text: str) -> List[str]:
    return text.splitlines()


def _chunk_file(path: Path, repo_root: Path) -> List[DocChunk]:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    lines = _split_lines(text)
    if not lines:
        return []
    rel = str(path.relative_to(repo_root))
    title = path.stem
    chunks: List[DocChunk] = []
    step = max(1, _CHUNK_LINES - _CHUNK_OVERLAP)
    for start in range(0, len(lines), step):
        end = min(len(lines), start + _CHUNK_LINES)
        body = "\n".join(lines[start:end]).strip()
        if not body:
            continue
        # Embed source path + a small header so the embedder sees context.
        head = f"# {rel} (lines {start + 1}-{end})\n"
        chunks.append(
            DocChunk(
                source=rel,
                start_line=start + 1,
                end_line=end,
                text=head + body,
                title=title,
            )
        )
        if end == len(lines):
            break
    return chunks


def ingest_repo(
    repo_root: Path,
    store_dir: Path,
    *,
    sources: Sequence[str] = DEFAULT_SOURCES,
    embed_model: str | None = None,
) -> DocStore:
    """Walk ``repo_root``, chunk the curated sources, embed, persist."""
    repo_root = Path(repo_root).resolve()
    store_dir = Path(store_dir).resolve()
    if not repo_root.is_dir():
        raise FileNotFoundError(f"repo not found: {repo_root}")
    chunks: List[DocChunk] = []
    for path in _iter_files(repo_root, sources):
        chunks.extend(_chunk_file(path, repo_root))
    store = DocStore(
        store_dir,
        embed_model=embed_model or "sentence-transformers/all-MiniLM-L6-v2",
    )
    store.populate(chunks)
    store.save()
    return store


__all__ = ["ingest_repo", "DEFAULT_SOURCES"]
