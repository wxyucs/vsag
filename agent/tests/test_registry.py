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

"""Tests for the registry validator + dispatch + error envelope."""

from __future__ import annotations

import pytest

from vsag_code.tools import (
    ToolError,
    ToolSpec,
    call_tool,
    chat_tool_specs,
    error_payload,
    validate_arguments,
)
from vsag_code.tools.registry import TIER_READ, register


def _spec_factory():
    """Build a fresh tool spec without permanently registering it.

    The registry rejects duplicates, so each unit test that needs a spec
    constructs one with a unique name and validates against it directly
    via ``validate_arguments`` -- no ``register`` call required.
    """
    return ToolSpec(
        name="probe_tool",
        description="probe",
        parameters={
            "type": "object",
            "properties": {
                "k": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
                "mode": {"type": "string", "enum": ["fast", "slow"]},
                "name": {"type": "string"},
            },
            "required": ["name"],
        },
        fn=lambda **kw: {"echoed": kw},
    )


def test_validate_arguments_happy_path_fills_defaults():
    spec = _spec_factory()
    out = validate_arguments(spec, {"name": "alice"})
    assert out == {"name": "alice", "k": 3}


def test_validate_arguments_missing_required():
    spec = _spec_factory()
    with pytest.raises(ToolError) as exc:
        validate_arguments(spec, {})
    assert exc.value.code == "invalid_argument"
    assert "name" in exc.value.message


def test_validate_arguments_rejects_unknown_argument():
    spec = _spec_factory()
    with pytest.raises(ToolError) as exc:
        validate_arguments(spec, {"name": "x", "extra": 1})
    assert exc.value.code == "invalid_argument"
    assert "extra" in exc.value.message


def test_validate_arguments_type_mismatch():
    spec = _spec_factory()
    with pytest.raises(ToolError) as exc:
        validate_arguments(spec, {"name": 1})
    assert exc.value.code == "invalid_argument"
    assert "string" in exc.value.message


def test_validate_arguments_bool_is_not_integer():
    spec = _spec_factory()
    with pytest.raises(ToolError):
        validate_arguments(spec, {"name": "x", "k": True})


def test_validate_arguments_enum_check():
    spec = _spec_factory()
    with pytest.raises(ToolError) as exc:
        validate_arguments(spec, {"name": "x", "mode": "loud"})
    assert exc.value.code == "invalid_argument"
    assert "loud" in exc.value.message


def test_validate_arguments_minimum_maximum():
    spec = _spec_factory()
    with pytest.raises(ToolError):
        validate_arguments(spec, {"name": "x", "k": 0})
    with pytest.raises(ToolError):
        validate_arguments(spec, {"name": "x", "k": 99})


def test_error_payload_shape_is_stable():
    payload = error_payload("not_found", "missing", "try X")
    assert payload == {
        "error": {
            "code": "not_found",
            "message": "missing",
            "suggestion": "try X",
        }
    }


def test_error_payload_rejects_unknown_code():
    with pytest.raises(ValueError):
        error_payload("not_a_real_code", "boom")


def test_call_tool_unknown_returns_error_envelope():
    res = call_tool("absolutely_not_registered_tool", {})
    assert "error" in res
    assert res["error"]["code"] == "not_found"


def test_call_tool_validates_before_dispatch():
    res = call_tool("dataset_info", {})
    assert "error" in res
    assert res["error"]["code"] == "invalid_argument"


def test_chat_tool_specs_renders_function_shape():
    specs = chat_tool_specs()
    assert all(s["type"] == "function" for s in specs)
    names = {s["function"]["name"] for s in specs}
    # Core tools must be present.
    for required in ("dataset_info", "index_build", "index_search", "index_remove"):
        assert required in names
    # Each entry must have a JSON-Schema-shaped parameters dict.
    for s in specs:
        params = s["function"]["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
