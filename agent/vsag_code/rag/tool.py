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

"""``docs_search`` tool: registered into the global tool registry.

This module is imported lazily so the package still works without the
``[rag]`` extras installed. The tool body returns a structured-error
envelope when the store is missing or sentence-transformers is not on
PYTHONPATH; the agent's prompt rules then steer it to mention the
limitation rather than retry.
"""

from __future__ import annotations

import os
from typing import Any, Dict

from ..tools.errors import error_payload
from ..tools.registry import TIER_READ, ToolSpec, register
from .retrieve import retrieve
from .store import DEFAULT_STORE_DIR


_DOCS_SEARCH_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Free-form natural-language question.",
        },
        "k": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "default": 5,
            "description": "Number of chunks to return.",
        },
    },
    "required": ["query"],
}


def docs_search(query: str, k: int = 5) -> Dict[str, Any]:
    """Return up to ``k`` retrieved doc chunks for ``query``.

    The store path is resolved from ``$VSAG_CODE_RAG_DIR`` if set, else
    falls back to ``$VSAG_CODE_HOME/rag`` (default
    ``/workspace/.vsag-code/rag``).
    """
    directory = os.environ.get("VSAG_CODE_RAG_DIR") or DEFAULT_STORE_DIR
    try:
        hits = retrieve(query, k=k, store_dir=directory)
    except FileNotFoundError as exc:
        return error_payload(
            "not_found",
            str(exc),
            "Run `python -m vsag_code.rag.cli ingest --repo <path>` first.",
        )
    except RuntimeError as exc:
        return error_payload(
            "unsupported",
            str(exc),
            "Install RAG extras: `pip install vsag-code[rag]`.",
        )
    return {"query": query, "k": k, "hits": hits, "store_dir": directory}


def register_docs_search() -> ToolSpec:
    return register(
        ToolSpec(
            name="docs_search",
            description=(
                "Retrieve VSAG documentation, header, and example snippets that "
                "best match a free-form question. Use this for conceptual "
                "questions and usage examples; do not use it as a substitute "
                "for dataset_info or index_stats."
            ),
            parameters=_DOCS_SEARCH_SCHEMA,
            fn=docs_search,
            tier=TIER_READ,
        )
    )


__all__ = ["docs_search", "register_docs_search"]
