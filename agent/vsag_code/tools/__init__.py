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

"""Tool layer: the curated set of operations the LLM is allowed to invoke.

Each tool is a pure Python function with a JSON-friendly signature and a
matching :class:`ToolSpec` declaring its name, description, JSON-schema
parameters, and permission tier. The agent loop is responsible for
permission gating; this module is purely declarative and side-effect-free
beyond the operation it advertises.

Tool tiers (see proposal \u00a73.5):

* ``read``         no observable mutation; runs without confirmation
* ``mutate``       creates / modifies in-process or on-disk state
* ``destructive``  removes data; requires explicit confirm

All tool functions return a JSON-serializable dict. Errors are reported
as ``{"error": {"code": str, "message": str, "suggestion": str}}`` rather
than raised; this matches the structured-error contract from the
proposal and lets the LLM self-recover.
"""

from __future__ import annotations

from .errors import ToolError, error_payload
from .registry import (
    TOOLS,
    ToolSpec,
    call_tool,
    chat_tool_specs,
    get_tool,
    validate_arguments,
)

__all__ = [
    "TOOLS",
    "ToolError",
    "ToolSpec",
    "call_tool",
    "chat_tool_specs",
    "error_payload",
    "get_tool",
    "validate_arguments",
]
