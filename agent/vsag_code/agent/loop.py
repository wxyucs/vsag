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

"""Tool-calling agent loop.

Single multi-turn loop: append user/system messages, call the LLM,
inspect tool calls, dispatch to the tool registry with permission
gating, append tool results, repeat until the model emits a plain
assistant message (no tool_calls) or we hit ``max_steps``.

The loop emits :class:`Event` objects via the ``on_event`` callback so
the TUI can stream tokens / progress without owning the loop. For a
non-interactive run the default callback just appends to a list and
returns the same trace shape that the spike used (so RESULTS.md
parsing keeps working).
"""

from __future__ import annotations

import json
import time
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..llm import ProviderConfig, chat_completions, chat_completions_url
from ..tools import (
    TOOLS,
    call_tool,
    chat_tool_specs,
    error_payload,
    get_tool,
)
from ..tools.registry import TIER_DESTRUCTIVE


# ---------------------------------------------------------------------------
# Events: streamed to the UI / logger
# ---------------------------------------------------------------------------


@dataclass
class Event:
    kind: str  # "step" | "tool_call" | "tool_result" | "final" | "error"
    data: Dict[str, Any]


EventHandler = Callable[[Event], None]


def _noop(_: Event) -> None:
    return


# ---------------------------------------------------------------------------
# Permission gate
# ---------------------------------------------------------------------------


ConfirmFn = Callable[[str, Dict[str, Any]], bool]


def _auto_allow(_name: str, _args: Dict[str, Any]) -> bool:
    return True


def _auto_deny(_name: str, _args: Dict[str, Any]) -> bool:
    return False


# ---------------------------------------------------------------------------
# AgentSession
# ---------------------------------------------------------------------------


@dataclass
class AgentSession:
    """Mutable conversation state for one logical chat session.

    A session bundles the system prompt, the running message history,
    the tool spec list, plus the LLM provider config and event sink.
    The TUI keeps one of these alive across user inputs; one-shot CLI
    runs construct, ``send``, and discard.
    """

    provider: ProviderConfig
    system_prompt: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_specs: List[Dict[str, Any]] = field(default_factory=list)
    on_event: EventHandler = field(default_factory=lambda: _noop)
    confirm: ConfirmFn = field(default_factory=lambda: _auto_allow)
    max_steps: int = 16
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tool_specs:
            self.tool_specs = chat_tool_specs()
        if not self.messages:
            self.messages.append({"role": "system", "content": self.system_prompt})

    # -- public API -------------------------------------------------------

    def send(self, user_text: str) -> Dict[str, Any]:
        """Append a user turn and run until the model stops calling tools."""
        self.messages.append({"role": "user", "content": user_text})
        return self._run_until_quiescent()

    def reset(self, system_prompt: Optional[str] = None) -> None:
        if system_prompt is not None:
            self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.trace = []

    # -- core loop --------------------------------------------------------

    def _run_until_quiescent(self) -> Dict[str, Any]:
        for step in range(self.max_steps):
            t0 = time.perf_counter()
            try:
                resp = chat_completions(
                    self.provider, self.messages, self.tool_specs
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                err = f"HTTP {exc.code}: {body[:500]}"
                self._emit(Event("error", {"step": step + 1, "error": err}))
                return {"ok": False, "error": err, "trace": self.trace}
            except urllib.error.URLError as exc:
                err = f"network: {exc}"
                self._emit(Event("error", {"step": step + 1, "error": err}))
                return {"ok": False, "error": err, "trace": self.trace}
            elapsed = time.perf_counter() - t0
            choice = resp["choices"][0]["message"]
            self.messages.append(choice)
            self.trace.append(
                {"step": step + 1, "elapsed_s": round(elapsed, 2), "message": choice}
            )
            self._emit(
                Event(
                    "step",
                    {
                        "step": step + 1,
                        "elapsed_s": round(elapsed, 2),
                        "model": self.provider.model,
                        "url": chat_completions_url(self.provider),
                    },
                )
            )

            tool_calls = choice.get("tool_calls") or []
            if not tool_calls:
                content = choice.get("content") or ""
                self._emit(Event("final", {"content": content}))
                return {"ok": True, "final": content, "trace": self.trace}

            self._dispatch_tool_calls(step + 1, tool_calls)
        msg = f"exceeded max_steps={self.max_steps} without final answer"
        self._emit(Event("error", {"error": msg}))
        return {"ok": False, "error": msg, "trace": self.trace}

    def _dispatch_tool_calls(
        self, step: int, tool_calls: List[Dict[str, Any]]
    ) -> None:
        for call in tool_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                args = {}
                self._emit(
                    Event(
                        "tool_call",
                        {"step": step, "name": name, "args": {}, "warning": str(exc)},
                    )
                )
            else:
                self._emit(
                    Event("tool_call", {"step": step, "name": name, "args": args})
                )

            result = self._gated_call(name, args)
            self._emit(
                Event("tool_result", {"step": step, "name": name, "result": result})
            )
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": json.dumps(result, default=str),
                }
            )
            self.trace.append(
                {"step": step, "tool": name, "args": args, "result": result}
            )

    def _gated_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        # Look up the tier first so we can deny destructive calls before
        # the validator even runs (cheaper + clearer error).
        try:
            spec = get_tool(name)
        except Exception:  # pylint: disable=broad-except
            return error_payload(
                "not_found",
                f"unknown tool: {name!r}",
                f"Available tools: {[t.name for t in TOOLS]}.",
            )
        if spec.tier == TIER_DESTRUCTIVE and not self.confirm(name, args):
            return error_payload(
                "permission_denied",
                f"destructive tool {name!r} not confirmed by user",
                "Ask the user to explicitly authorize this deletion.",
            )
        return call_tool(name, args)

    # -- internals --------------------------------------------------------

    def _emit(self, event: Event) -> None:
        try:
            self.on_event(event)
        except Exception:  # pylint: disable=broad-except
            # Event sinks must not break the loop.
            pass


# ---------------------------------------------------------------------------
# Convenience: one-shot run (used by spike-style scripts and CI tests)
# ---------------------------------------------------------------------------


def run_loop(
    provider: ProviderConfig,
    system_prompt: str,
    user_text: str,
    *,
    on_event: Optional[EventHandler] = None,
    confirm: Optional[ConfirmFn] = None,
    max_steps: int = 16,
) -> Dict[str, Any]:
    """Run a single non-interactive turn end-to-end."""
    session = AgentSession(
        provider=provider,
        system_prompt=system_prompt,
        on_event=on_event or _noop,
        confirm=confirm or _auto_allow,
        max_steps=max_steps,
    )
    return session.send(user_text)
