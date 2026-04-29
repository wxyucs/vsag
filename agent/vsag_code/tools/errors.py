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

"""Structured-error contract shared by every tool.

The tool layer never raises into the agent loop; instead, every tool
returns either a normal success dict or:

    {"error": {"code": str, "message": str, "suggestion": str}}

This shape is stable and small enough to fit in a tool-call response
without burning context, while still giving the LLM enough signal to
self-correct (the ``suggestion`` field). Error codes are kept in a
small enum-like set so tests and prompts can match on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

# Stable error codes. The list is intentionally short; new codes should
# be added only when the LLM needs to distinguish a recoverable case
# from an unrecoverable one.
ERROR_CODES = (
    "not_found",          # path / handle / preset does not exist
    "invalid_argument",   # arg fails JSON-schema or range check
    "unsupported",        # algorithm / metric / dtype not in the matrix
    "permission_denied",  # destructive op without confirm
    "internal",           # uncaught exception; surfaced with str(exc)
)


@dataclass(frozen=True)
class ToolError(Exception):
    """Raised internally inside a tool. Wrapped to a dict by ``error_payload``."""

    code: str
    message: str
    suggestion: str = ""

    def __post_init__(self) -> None:
        # Keep ``Exception``'s ``args`` tuple consistent for test ergonomics.
        Exception.__init__(self, f"{self.code}: {self.message}")
        if self.code not in ERROR_CODES:
            raise ValueError(f"unknown error code: {self.code}")


def error_payload(
    code: str, message: str, suggestion: Optional[str] = None
) -> Dict[str, Any]:
    """Return the canonical error dict for a tool result."""
    if code not in ERROR_CODES:
        raise ValueError(f"unknown error code: {code}")
    return {
        "error": {
            "code": code,
            "message": message,
            "suggestion": suggestion or "",
        }
    }
