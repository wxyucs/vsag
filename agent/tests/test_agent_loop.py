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

"""Tests for :class:`AgentSession`: dispatch, permission gate, max_steps.

Uses a fake :func:`chat_completions` so we never hit the network.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from vsag_code.agent import AgentSession, Event
from vsag_code.llm import ProviderConfig
from vsag_code.tools import call_tool
from vsag_code.tools.registry import TIER_DESTRUCTIVE, ToolSpec, register


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _provider() -> ProviderConfig:
    return ProviderConfig(
        name="fake",
        base_url="http://example.invalid",
        model="fake-model",
        auth_header="Bearer fake",
        extra_headers={},
    )


class _ScriptedLLM:
    """Replay a fixed list of assistant messages, ignoring the input."""

    def __init__(self, scripted_messages: List[Dict[str, Any]]):
        self.scripted = list(scripted_messages)
        self.calls = 0

    def __call__(self, cfg, messages, tools, **kwargs):
        self.calls += 1
        if not self.scripted:
            raise AssertionError("LLM ran out of scripted responses")
        msg = self.scripted.pop(0)
        return {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]}


@pytest.fixture
def patch_chat(monkeypatch):
    """Install a scripted LLM into both ``llm.client`` and ``agent.loop``."""

    def _install(scripted):
        fake = _ScriptedLLM(scripted)
        # The loop imports ``chat_completions`` at module scope, so patch
        # the binding inside ``vsag_code.agent.loop``, not just the source.
        import vsag_code.agent.loop as loop_mod
        import vsag_code.llm as llm_mod

        monkeypatch.setattr(llm_mod, "chat_completions", fake)
        monkeypatch.setattr(loop_mod, "chat_completions", fake)
        return fake

    return _install


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


def test_session_returns_final_when_no_tool_calls(patch_chat):
    patch_chat([{"role": "assistant", "content": "all set"}])
    session = AgentSession(provider=_provider(), system_prompt="sys")
    result = session.send("hello")
    assert result["ok"] is True
    assert result["final"] == "all set"


def test_session_dispatches_tool_calls_then_finishes(patch_chat):
    scripted = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "dataset_list",
                        "arguments": json.dumps({}),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "done"},
    ]
    patch_chat(scripted)
    session = AgentSession(provider=_provider(), system_prompt="sys")

    events: List[Event] = []
    session.on_event = events.append
    result = session.send("list datasets please")

    assert result["ok"] is True
    assert result["final"] == "done"
    kinds = [e.kind for e in events]
    assert "tool_call" in kinds
    assert "tool_result" in kinds
    assert "final" in kinds


def test_session_blocks_destructive_without_confirm(patch_chat, vsag_code_home):
    scripted = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_x",
                    "type": "function",
                    "function": {
                        "name": "index_remove",
                        "arguments": json.dumps({"handle": "ghost"}),
                    },
                }
            ],
        },
        {"role": "assistant", "content": "stopped"},
    ]
    patch_chat(scripted)
    session = AgentSession(
        provider=_provider(),
        system_prompt="sys",
        confirm=lambda _n, _a: False,
    )
    result = session.send("delete ghost")
    assert result["ok"] is True
    # Tool result should be a permission_denied error envelope.
    tool_steps = [s for s in result["trace"] if s.get("tool") == "index_remove"]
    assert tool_steps
    payload = tool_steps[0]["result"]
    assert payload["error"]["code"] == "permission_denied"


def test_session_max_steps_terminates(patch_chat):
    looping = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "loop",
                "type": "function",
                "function": {
                    "name": "dataset_list",
                    "arguments": json.dumps({}),
                },
            }
        ],
    }
    patch_chat([looping] * 10)
    session = AgentSession(provider=_provider(), system_prompt="sys", max_steps=3)
    result = session.send("loop")
    assert result["ok"] is False
    assert "max_steps" in result["error"]
