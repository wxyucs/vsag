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

"""Tool registry: ``ToolSpec`` declarations, validation, dispatch.

This module owns the LLM-facing tool surface. It does *not* implement
any tool body; the bodies live in :mod:`vsag_code.tools.dataset_tools`,
:mod:`vsag_code.tools.index_tools`, etc., and are imported here to be
registered. This split keeps the registry a single audit-able list.

The validator is a tiny stdlib-only JSON-Schema subset covering exactly
what tool args need: ``type``, ``enum``, ``required``, ``default``, plus
nested ``properties``. Anything richer can be added as needed; the goal
is to refuse obvious bad calls (wrong type / unknown arg / missing
required) before the tool body runs, without pulling in jsonschema as a
dependency.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

from .errors import ToolError, error_payload

# Permission tiers (proposal \u00a73.5).
TIER_READ = "read"
TIER_MUTATE = "mutate"
TIER_DESTRUCTIVE = "destructive"
TIERS = (TIER_READ, TIER_MUTATE, TIER_DESTRUCTIVE)


# ---------------------------------------------------------------------------
# ToolSpec
# ---------------------------------------------------------------------------


@dataclass
class ToolSpec:
    """Declarative description of one tool.

    Attributes:
        name: stable identifier the LLM uses; ``snake_case``.
        description: one-paragraph natural-language hint shown to the LLM.
        parameters: JSON-Schema (subset) for the function arguments.
        fn: callable invoked by ``call_tool``; signature must match the
            schema's ``properties`` keys.
        tier: permission tier (read / mutate / destructive).
    """

    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Callable[..., Dict[str, Any]] = field(repr=False)
    tier: str = TIER_READ

    def __post_init__(self) -> None:
        if self.tier not in TIERS:
            raise ValueError(f"invalid tier {self.tier!r}; must be one of {TIERS}")
        if not self.name.replace("_", "").isalnum():
            raise ValueError(f"tool name must be snake_case alnum: {self.name!r}")
        if self.parameters.get("type") != "object":
            raise ValueError(f"tool {self.name!r} parameters must be a JSON object schema")


# ---------------------------------------------------------------------------
# Schema validation (stdlib subset)
# ---------------------------------------------------------------------------


_JSON_TYPE_MAP: Dict[str, Tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),  # bool is also int; we explicitly reject bool below
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list, tuple),
    "null": (type(None),),
}


def _validate_value(name: str, value: Any, schema: Dict[str, Any]) -> Any:
    """Validate ``value`` against ``schema`` and return its coerced form."""
    declared = schema.get("type")
    if declared is None:
        return value
    if declared == "integer" and isinstance(value, bool):
        raise ToolError(
            "invalid_argument",
            f"{name!r}: expected integer, got bool",
            "Pass an integer literal, e.g. 1000.",
        )
    expected = _JSON_TYPE_MAP.get(declared)
    if expected is None:
        return value
    if not isinstance(value, expected):
        raise ToolError(
            "invalid_argument",
            f"{name!r}: expected {declared}, got {type(value).__name__}",
            f"Use a {declared} value.",
        )
    if "enum" in schema and value not in schema["enum"]:
        raise ToolError(
            "invalid_argument",
            f"{name!r}: value {value!r} not in {schema['enum']}",
            f"Choose one of: {', '.join(map(str, schema['enum']))}.",
        )
    if declared == "integer":
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolError(
                "invalid_argument",
                f"{name!r}: {value} < minimum {schema['minimum']}",
                f"Use a value >= {schema['minimum']}.",
            )
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolError(
                "invalid_argument",
                f"{name!r}: {value} > maximum {schema['maximum']}",
                f"Use a value <= {schema['maximum']}.",
            )
    return value


def validate_arguments(
    spec: ToolSpec, arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate + coerce ``arguments`` against ``spec.parameters``.

    Returns a new dict containing only declared properties, with defaults
    filled in for omitted optional arguments.

    Raises :class:`ToolError` on any mismatch.
    """
    schema = spec.parameters
    props: Dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    out: Dict[str, Any] = {}

    unknown = set(arguments) - set(props)
    if unknown:
        raise ToolError(
            "invalid_argument",
            f"unknown argument(s) for {spec.name!r}: {sorted(unknown)}",
            f"Allowed: {sorted(props.keys())}.",
        )

    for key, sub in props.items():
        if key in arguments:
            out[key] = _validate_value(key, arguments[key], sub)
        elif key in required:
            raise ToolError(
                "invalid_argument",
                f"missing required argument {key!r} for {spec.name!r}",
                f"Provide {key} ({sub.get('type', 'any')}).",
            )
        elif "default" in sub:
            out[key] = sub["default"]
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


_REGISTRY: Dict[str, ToolSpec] = {}


def register(spec: ToolSpec) -> ToolSpec:
    """Register ``spec``; returns it so callers can chain."""
    if spec.name in _REGISTRY:
        raise ValueError(f"duplicate tool: {spec.name!r}")
    _REGISTRY[spec.name] = spec
    return spec


def get_tool(name: str) -> ToolSpec:
    if name not in _REGISTRY:
        raise ToolError(
            "not_found",
            f"unknown tool: {name!r}",
            f"Available: {sorted(_REGISTRY)}.",
        )
    return _REGISTRY[name]


def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Validate + dispatch a tool call. Returns a JSON-serializable dict.

    Errors (including ``ToolError``) are translated to the structured
    error envelope; the agent loop never sees an exception from here.
    """
    try:
        spec = get_tool(name)
        cleaned = validate_arguments(spec, arguments or {})
        result = spec.fn(**cleaned)
    except ToolError as exc:
        return error_payload(exc.code, exc.message, exc.suggestion)
    except Exception as exc:  # pylint: disable=broad-except
        return error_payload(
            "internal",
            f"{type(exc).__name__}: {exc}",
            "This is likely a bug; the trace will be visible in the agent log.",
        )
    if not isinstance(result, dict):
        return error_payload(
            "internal",
            f"tool {name!r} returned non-dict: {type(result).__name__}",
            "Tools must return a JSON-serializable dict.",
        )
    return result


def chat_tool_specs() -> List[Dict[str, Any]]:
    """Render the registry as OpenAI-style function specs for chat APIs."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in _REGISTRY.values()
    ]


@property  # type: ignore[misc]
def TOOLS_property():  # pragma: no cover - kept for back-compat
    return list(_REGISTRY.values())


def _all_tools() -> List[ToolSpec]:
    return list(_REGISTRY.values())


# Eagerly import the per-domain tool modules so their decorators / direct
# ``register()`` calls populate the registry. The import order matters
# only for tests that snapshot ``TOOLS`` before any tool body runs.
from . import dataset_tools as _dataset_tools  # noqa: E402,F401
from . import index_tools as _index_tools  # noqa: E402,F401

# RAG tool is optional: register only if the module imports cleanly.
# Failures here are silent on purpose -- a fresh checkout without the
# ``[rag]`` extras installed must still import the rest of the registry.
try:
    from ..rag.tool import register_docs_search as _register_docs_search  # noqa: E402

    _register_docs_search()
except Exception:  # pylint: disable=broad-except
    pass

# Public attribute kept for compatibility with the spike layout.
TOOLS: List[ToolSpec] = _all_tools()


# Sanity: every registered tool's function signature should accept the
# schema's properties as keyword args. Catches drift between the schema
# and the body at import time rather than at the first LLM call.
def _audit_signatures() -> None:
    for spec in _REGISTRY.values():
        sig = inspect.signature(spec.fn)
        accepted = set(sig.parameters.keys())
        accepts_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_kwargs:
            continue
        declared = set(spec.parameters.get("properties", {}).keys())
        missing = declared - accepted
        if missing:
            raise RuntimeError(
                f"tool {spec.name!r}: function does not accept declared "
                f"properties {sorted(missing)}"
            )


_audit_signatures()
