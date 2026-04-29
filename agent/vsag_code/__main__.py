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

"""``vsag-code`` CLI entry point.

Two run modes:

* interactive: ``vsag-code --provider deepseek`` opens a REPL.
* one-shot: ``vsag-code --provider deepseek --once "goal text"``
  runs a single agent turn, prints the streamed events + final, and
  optionally writes the JSON trace to ``--out trace.json``.

The CLI itself is intentionally thin: argument parsing, provider
resolution, prompt selection, and a hand-off to either the REPL or
``run_once``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from .agent import AgentSession, SYSTEM_PROMPT_EN, SYSTEM_PROMPT_ZH
from .llm import resolve_provider
from .tui.repl import Repl, run_once


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vsag-code",
        description="Conversational TUI agent for the VSAG vector index library.",
    )
    p.add_argument(
        "--provider",
        default=os.environ.get("VSAG_CODE_PROVIDER", "deepseek"),
        choices=["openai", "deepseek", "copilot", "anthropic", "ollama"],
        help="LLM provider (default: deepseek).",
    )
    p.add_argument(
        "--model",
        default=os.environ.get("VSAG_CODE_MODEL"),
        help="Override the provider's default model id.",
    )
    p.add_argument(
        "--lang",
        default=os.environ.get("VSAG_CODE_LANG", "en"),
        choices=["en", "zh"],
        help="System-prompt locale (default: en).",
    )
    p.add_argument(
        "--once",
        metavar="GOAL",
        default=None,
        help="Run one goal headlessly and exit.",
    )
    p.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Write the JSON trace to PATH (one-shot mode).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Auto-allow destructive tools (skip confirm prompt).",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=16,
        help="Maximum agent loop iterations per turn (default: 16).",
    )
    p.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in output.",
    )
    p.add_argument(
        "--history",
        metavar="PATH",
        default=None,
        help="Override REPL history file path.",
    )
    return p


def _system_prompt(lang: str) -> str:
    return SYSTEM_PROMPT_ZH if lang == "zh" else SYSTEM_PROMPT_EN


def _history_path(override: Optional[str]) -> Path:
    if override:
        return Path(override).expanduser()
    home = Path(os.environ.get("VSAG_CODE_HOME", "/workspace/.vsag-code"))
    return home / "history"


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    color = not args.no_color
    sysprompt = _system_prompt(args.lang)

    try:
        cfg = resolve_provider(args.provider, args.model)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.once is not None:
        result = run_once(
            cfg,
            sysprompt,
            args.once,
            color=color,
            auto_yes=args.yes,
            max_steps=args.max_steps,
        )
        if args.out:
            Path(args.out).write_text(
                json.dumps(result, default=str, indent=2),
                encoding="utf-8",
            )
            print(f"wrote trace -> {args.out}", file=sys.stderr)
        return 0 if result.get("ok") else 1

    session = AgentSession(
        provider=cfg,
        system_prompt=sysprompt,
        max_steps=args.max_steps,
    )
    repl = Repl(
        session,
        history_path=_history_path(args.history),
        color=color,
        auto_yes=args.yes,
    )
    return repl.run()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
