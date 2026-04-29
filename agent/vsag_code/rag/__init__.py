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

"""Local RAG over VSAG documentation, examples, and public headers.

The agent can call ``docs_search(query, k=5)`` (registered as a
``read``-tier tool) to retrieve grounded snippets when a user asks
conceptual questions or for usage examples. Embeddings are computed
locally with ``sentence-transformers`` (``all-MiniLM-L6-v2``); the
sentence-transformers dependency is optional and the tool degrades
gracefully when it is missing.
"""

from .ingest import ingest_repo, DEFAULT_SOURCES
from .store import DocStore, DocChunk
from .retrieve import retrieve

__all__ = [
    "ingest_repo",
    "DEFAULT_SOURCES",
    "DocStore",
    "DocChunk",
    "retrieve",
]
