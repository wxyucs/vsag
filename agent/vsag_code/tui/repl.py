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

"""Prompt-toolkit REPL for the VSAG-Code agent.

A thin wrapper around :class:`vsag_code.agent.AgentSession`. Owns:

* the prompt loop + history file at ``$VSAG_CODE_HOME/history``;
* the slash-command vocabulary (``/help``, ``/provider``, ``/reset``,
  ``/trace``, ``/quit``);
* the streaming event renderer (tool calls, tool results, errors, the
  final assistant turn) using ANSI colors;
* the destructive-tool confirm prompt (``index_remove``).

The output style is intentionally minimal: monochrome fallback when
``stdout.isatty()`` is False (so ``--once GOAL >trace.txt`` produces a
clean log without ANSI escapes).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..agent import AgentSession, Event
from ..llm import ProviderConfig

try:  # pragma: no cover - import-time fallback when prompt_toolkit missing
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.shortcuts import confirm as pt_confirm
except ImportError:  # pragma: no cover
    PromptSession = None  # type: ignore[assignment]
    FileHistory = None  # type: ignore[assignment]
    pt_confirm = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------


_ANSI = {
    "reset": "\033[0m",
    "dim": "\033[2m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


def _supports_ansi(stream) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _paint(text: str, color: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{_ANSI[color]}{text}{_ANSI['reset']}"


# ---------------------------------------------------------------------------
# Event rendering
# ---------------------------------------------------------------------------


def render_event(event: Event, *, color: bool = True, stream=None) -> None:
    """Pretty-print one :class:`Event` to ``stream`` (default stdout)."""
    out = stream if stream is not None else sys.stdout
    enabled = color and _supports_ansi(out)
    kind = event.kind
    data = event.data
    if kind == "step":
        n = data.get("step")
        elapsed = data.get("elapsed_s")
        model = data.get("model", "?")
        line = _paint(f"[step {n}] {model} ({elapsed}s)", "dim", enabled=enabled)
        print(line, file=out)
    elif kind == "tool_call":
        name = data.get("name", "?")
        args = data.get("args", {})
        head = _paint(f"  -> {name}", "cyan", enabled=enabled)
        body = _paint(json.dumps(args, default=str), "dim", enabled=enabled)
        print(f"{head} {body}", file=out)
        if data.get("warning"):
            warn = _paint(f"     warn: {data['warning']}", "yellow", enabled=enabled)
            print(warn, file=out)
    elif kind == "tool_result":
        result = data.get("result", {})
        if isinstance(result, dict) and "error" in result:
            err = result["error"]
            head = _paint("  <- error", "red", enabled=enabled)
            print(
                f"{head} {err.get('code', '?')}: {err.get('message', '')}",
                file=out,
            )
            if err.get("suggestion"):
                tip = _paint(f"     hint: {err['suggestion']}", "yellow", enabled=enabled)
                print(tip, file=out)
        else:
            head = _paint("  <- ok", "green", enabled=enabled)
            summary = _summarize_result(result)
            print(f"{head} {summary}", file=out)
    elif kind == "final":
        content = data.get("content") or ""
        bar = _paint("=" * 60, "dim", enabled=enabled)
        print(bar, file=out)
        print(content, file=out)
        print(bar, file=out)
    elif kind == "error":
        msg = data.get("error", "?")
        line = _paint(f"!! {msg}", "red", enabled=enabled)
        print(line, file=out)
    out.flush()


def _summarize_result(result: Dict[str, Any]) -> str:
    """Compact one-line view; full payload is in the JSON trace."""
    if not isinstance(result, dict):
        return repr(result)
    keys = list(result.keys())
    if "handle" in result:
        return f"handle={result['handle']}"
    if "recall_at_k" in result and "qps" in result:
        return (
            f"recall@{result.get('topk', '?')}={result['recall_at_k']:.4f} "
            f"qps={result['qps']:.1f}"
        )
    if "datasets" in result:
        return f"datasets={len(result['datasets'])}"
    if "indexes" in result:
        return f"indexes={len(result['indexes'])}"
    if "hits" in result:
        return f"hits={len(result['hits'])}"
    short = {k: result[k] for k in keys[:3]}
    return json.dumps(short, default=str)[:200]


# ---------------------------------------------------------------------------
# REPL
# ---------------------------------------------------------------------------


_HELP = """\
Slash commands:
  /help            show this message
  /provider        show current LLM provider + model
  /reset           clear conversation, keep system prompt
  /trace [PATH]    dump current trace as JSON (default: stdout)
  /quit            exit

Anything else is sent to the agent. The agent will inspect datasets
before building or searching, persist index handles across turns, and
ask before deleting anything.
"""


ConfirmFn = Callable[[str, Dict[str, Any]], bool]


class Repl:
    """Stateful REPL bound to one :class:`AgentSession`."""

    def __init__(
        self,
        session: AgentSession,
        *,
        history_path: Optional[Path] = None,
        color: bool = True,
        auto_yes: bool = False,
        out=None,
    ) -> None:
        self.session = session
        self.color = color
        self.auto_yes = auto_yes
        self.out = out if out is not None else sys.stdout
        self._history_path = history_path
        self._pt_session: Optional[Any] = None
        # Wire renderer + confirm callback into the session.
        self.session.on_event = lambda ev: render_event(
            ev, color=self.color, stream=self.out
        )
        self.session.confirm = self._confirm_destructive

    # -- public API -------------------------------------------------------

    def run(self) -> int:
        """Blocking loop. Returns exit code (0 on clean quit)."""
        self._banner()
        self._init_prompt_session()
        while True:
            try:
                line = self._read_line()
            except (EOFError, KeyboardInterrupt):
                print(file=self.out)
                return 0
            if not line.strip():
                continue
            if line.startswith("/"):
                rc = self._handle_command(line.strip())
                if rc is not None:
                    return rc
                continue
            try:
                self.session.send(line)
            except Exception as exc:  # pylint: disable=broad-except
                err = _paint(f"!! send failed: {exc}", "red", enabled=self.color)
                print(err, file=self.out)

    # -- internals --------------------------------------------------------

    def _banner(self) -> None:
        cfg = self.session.provider
        head = _paint(
            f"VSAG-Code (provider={cfg.name}, model={cfg.model})",
            "bold",
            enabled=self.color,
        )
        sub = _paint(
            "Type /help for commands. Ctrl-D to quit.",
            "dim",
            enabled=self.color,
        )
        print(head, file=self.out)
        print(sub, file=self.out)

    def _init_prompt_session(self) -> None:
        if PromptSession is None:
            return
        history = None
        if self._history_path is not None:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(self._history_path))
        self._pt_session = PromptSession(history=history)

    def _read_line(self) -> str:
        if self._pt_session is not None:
            return self._pt_session.prompt("> ")
        # Fallback when prompt_toolkit is unavailable (e.g. CI without TTY).
        return input("> ")

    def _handle_command(self, line: str) -> Optional[int]:
        parts = line.split(maxsplit=1)
        cmd = parts[0]
        rest = parts[1].strip() if len(parts) == 2 else ""
        if cmd in ("/quit", "/exit", "/q"):
            return 0
        if cmd == "/help":
            print(_HELP, file=self.out)
            return None
        if cmd == "/provider":
            cfg = self.session.provider
            print(f"provider={cfg.name} model={cfg.model}", file=self.out)
            return None
        if cmd == "/reset":
            self.session.reset()
            print("session reset.", file=self.out)
            return None
        if cmd == "/trace":
            self._dump_trace(rest or None)
            return None
        print(f"unknown command: {cmd} (try /help)", file=self.out)
        return None

    def _dump_trace(self, path: Optional[str]) -> None:
        payload = json.dumps(self.session.trace, default=str, indent=2)
        if path:
            Path(path).write_text(payload, encoding="utf-8")
            print(f"wrote trace -> {path}", file=self.out)
        else:
            print(payload, file=self.out)

    def _confirm_destructive(self, name: str, args: Dict[str, Any]) -> bool:
        if self.auto_yes:
            return True
        prompt = f"\n[confirm] destructive tool {name!r} args={args}; allow? [y/N] "
        if pt_confirm is not None:
            try:
                return bool(pt_confirm(prompt))
            except (EOFError, KeyboardInterrupt):
                return False
        try:
            return (input(prompt).strip().lower() in ("y", "yes"))
        except (EOFError, KeyboardInterrupt):
            return False


# ---------------------------------------------------------------------------
# Headless helper used by ``--once GOAL``
# ---------------------------------------------------------------------------


def run_once(
    provider: ProviderConfig,
    system_prompt: str,
    goal: str,
    *,
    color: bool = True,
    auto_yes: bool = False,
    max_steps: int = 16,
    out=None,
) -> Dict[str, Any]:
    """Run a single goal headlessly while still streaming events."""
    out = out if out is not None else sys.stdout
    session = AgentSession(
        provider=provider,
        system_prompt=system_prompt,
        max_steps=max_steps,
    )
    session.on_event = lambda ev: render_event(ev, color=color, stream=out)
    if auto_yes:
        session.confirm = lambda _n, _a: True
    else:
        session.confirm = lambda _n, _a: False
    return session.send(goal)


__all__ = ["Repl", "render_event", "run_once"]
