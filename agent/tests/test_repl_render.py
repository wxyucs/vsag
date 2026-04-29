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

"""Tests for :mod:`vsag_code.tui.repl`: event rendering only.

The actual prompt loop touches stdin / prompt_toolkit and is not unit
tested -- it's exercised via the end-to-end smoke run in the pod.
"""

from __future__ import annotations

import io

from vsag_code.agent import Event
from vsag_code.tui.repl import render_event


def _render(event: Event, color: bool = False) -> str:
    buf = io.StringIO()
    render_event(event, color=color, stream=buf)
    return buf.getvalue()


def test_render_step_event():
    out = _render(Event("step", {"step": 1, "elapsed_s": 0.42, "model": "m"}))
    assert "step 1" in out
    assert "0.42" in out


def test_render_tool_call_event():
    out = _render(Event("tool_call", {"step": 1, "name": "dataset_list", "args": {}}))
    assert "dataset_list" in out


def test_render_tool_result_ok():
    out = _render(
        Event("tool_result", {"step": 1, "name": "x", "result": {"handle": "h-1"}})
    )
    assert "ok" in out
    assert "handle=h-1" in out


def test_render_tool_result_error_includes_suggestion():
    err = {
        "error": {
            "code": "not_found",
            "message": "missing",
            "suggestion": "try X",
        }
    }
    out = _render(Event("tool_result", {"step": 1, "name": "x", "result": err}))
    assert "not_found" in out
    assert "missing" in out
    assert "try X" in out


def test_render_final_includes_separator():
    out = _render(Event("final", {"content": "hello"}))
    assert "hello" in out


def test_render_search_summary_uses_recall_at_k():
    res = {
        "topk": 10,
        "recall_at_k": 0.95,
        "qps": 1234.5,
        "p95_latency_ms": 1.2,
    }
    out = _render(Event("tool_result", {"step": 1, "name": "index_search", "result": res}))
    assert "recall@10=0.9500" in out
    assert "qps=1234.5" in out
