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

"""Stage-1 spike: LLM-driven tool-calling loop.

Single script that wires :mod:`tools` to an LLM via OpenAI-style function
calling. Goal: validate end-to-end that a chat model can chain
``dataset_info`` -> ``index_build`` -> ``index_search`` from a natural
language goal, and that ``recall@k`` comes out plausible.

Providers (selected via ``--provider``):

* ``openai``    OpenAI / DeepSeek / any OpenAI-compatible endpoint.
* ``deepseek``  alias for ``--provider openai`` with DeepSeek defaults.
* ``copilot``   GitHub Copilot via OAuth device flow; access token cached
                under ``$VSAG_CODE_HOME`` (default ``/workspace/.vsag-code``).

Env vars consumed:
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL,
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL,
    COPILOT_MODEL, VSAG_CODE_HOME,
    HTTP_PROXY/HTTPS_PROXY (honored automatically by ``urllib``).

Stdlib only on purpose: spike must not bring an SDK dependency footprint
that would later have to be undone for the staged TUI client. The driver
is intentionally chatty on stdout so its trace can be pasted into
``RESULTS.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ``tools`` lives next to this file; allow ``python spike.py`` from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tools  # noqa: E402  (intentional sys.path tweak above)


# ---------------------------------------------------------------------------
# Provider configuration
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    model: str
    auth_header: str  # e.g. "Bearer sk-..." or "token ghu_..."
    extra_headers: Dict[str, str]


def _resolve_openai(provider: str) -> ProviderConfig:
    """Build ``ProviderConfig`` for OpenAI / DeepSeek / generic compat."""
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError(
            f"missing API key for provider={provider}; set "
            f"{'DEEPSEEK_API_KEY' if provider == 'deepseek' else 'OPENAI_API_KEY'}"
        )
    return ProviderConfig(
        name=provider,
        base_url=base.rstrip("/"),
        model=model,
        auth_header=f"Bearer {api_key}",
        extra_headers={},
    )


# ---------------------------------------------------------------------------
# GitHub Copilot device flow
# ---------------------------------------------------------------------------


# Public Copilot client_id used by the official VS Code extension. Copilot
# does not currently issue per-app client_ids for third-party tools; using
# this id is the documented community pattern for editor integrations and
# requires the user to have an active Copilot subscription on their
# GitHub account (see issue tracker references in proposal §5.2).
_COPILOT_CLIENT_ID = "Iv1.b507a08c87ecfe98"
_COPILOT_DEVICE_URL = "https://github.com/login/device/code"
_COPILOT_TOKEN_URL = "https://github.com/login/oauth/access_token"
_COPILOT_API_TOKEN_URL = "https://api.github.com/copilot_internal/v2/token"
_COPILOT_CHAT_URL = "https://api.githubcopilot.com/chat/completions"
_COPILOT_EDITOR_HEADERS = {
    "Editor-Version": "vscode/1.95.0",
    "Editor-Plugin-Version": "copilot-chat/0.22.0",
    "Copilot-Integration-Id": "vscode-chat",
    "User-Agent": "GitHubCopilotChat/0.22.0",
}


def _vsag_code_home() -> Path:
    home = Path(os.environ.get("VSAG_CODE_HOME", "/workspace/.vsag-code"))
    home.mkdir(parents=True, exist_ok=True)
    return home


def _http_post_form(url: str, form: Dict[str, str], headers: Dict[str, str]) -> Dict[str, Any]:
    body = "&".join(f"{k}={urllib.request.quote(v)}" for k, v in form.items()).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _http_get_json(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _copilot_oauth_login() -> str:
    """Run device-flow once, return the long-lived GitHub OAuth token."""
    print("\n[copilot] starting device flow login...", flush=True)
    init = _http_post_form(
        _COPILOT_DEVICE_URL,
        {"client_id": _COPILOT_CLIENT_ID, "scope": "read:user"},
        {"Accept": "application/json"},
    )
    print(f"[copilot] visit {init['verification_uri']}", flush=True)
    print(f"[copilot] enter code: {init['user_code']}", flush=True)
    interval = int(init.get("interval", 5))
    deadline = time.time() + int(init.get("expires_in", 900))
    while time.time() < deadline:
        time.sleep(interval)
        try:
            res = _http_post_form(
                _COPILOT_TOKEN_URL,
                {
                    "client_id": _COPILOT_CLIENT_ID,
                    "device_code": init["device_code"],
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                {"Accept": "application/json"},
            )
        except urllib.error.URLError as exc:  # transient network blip
            print(f"[copilot] poll error: {exc}; retrying", flush=True)
            continue
        if "access_token" in res:
            return res["access_token"]
        if res.get("error") == "authorization_pending":
            continue
        if res.get("error") == "slow_down":
            interval += 5
            continue
        raise RuntimeError(f"device flow failed: {res}")
    raise RuntimeError("device flow timed out")


def _copilot_load_or_login() -> str:
    cache = _vsag_code_home() / "copilot-token.json"
    if cache.is_file():
        try:
            return json.loads(cache.read_text())["github_token"]
        except (KeyError, json.JSONDecodeError):
            pass
    token = _copilot_oauth_login()
    cache.write_text(json.dumps({"github_token": token}))
    cache.chmod(0o600)
    return token


def _copilot_exchange_api_token(github_token: str) -> Tuple[str, int]:
    """Exchange the GitHub OAuth token for a short-lived Copilot API token."""
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/json",
        **_COPILOT_EDITOR_HEADERS,
    }
    res = _http_get_json(_COPILOT_API_TOKEN_URL, headers)
    return res["token"], int(res.get("expires_at", time.time() + 1500))


def _resolve_copilot() -> ProviderConfig:
    github_token = _copilot_load_or_login()
    api_token, _ = _copilot_exchange_api_token(github_token)
    model = os.environ.get("COPILOT_MODEL", "gpt-4o")
    return ProviderConfig(
        name="copilot",
        base_url=_COPILOT_CHAT_URL.rsplit("/chat/", 1)[0],
        model=model,
        auth_header=f"Bearer {api_token}",
        extra_headers=_COPILOT_EDITOR_HEADERS,
    )


def resolve_provider(provider: str) -> ProviderConfig:
    if provider in ("openai", "deepseek"):
        return _resolve_openai(provider)
    if provider == "copilot":
        return _resolve_copilot()
    raise RuntimeError(f"unknown provider: {provider}")


# ---------------------------------------------------------------------------
# OpenAI-style chat-completions request
# ---------------------------------------------------------------------------


def _chat_completions_url(cfg: ProviderConfig) -> str:
    if cfg.name == "copilot":
        return _COPILOT_CHAT_URL
    return f"{cfg.base_url}/chat/completions"


def chat_once(cfg: ProviderConfig, messages: List[Dict[str, Any]],
              tool_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
    payload = {
        "model": cfg.model,
        "messages": messages,
        "tools": tool_specs,
        "tool_choice": "auto",
        "temperature": 0.0,
    }
    headers = {
        "Authorization": cfg.auth_header,
        "Content-Type": "application/json",
        "Accept": "application/json",
        **cfg.extra_headers,
    }
    req = urllib.request.Request(
        _chat_completions_url(cfg),
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Tool spec serialization
# ---------------------------------------------------------------------------


def _tool_specs_for_chat() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            },
        }
        for spec in tools.TOOLS
    ]


# ---------------------------------------------------------------------------
# Driver loop
# ---------------------------------------------------------------------------


SYSTEM_PROMPT = """You are VSAG-Code, a non-coding agent that operates the
VSAG vector-index library by calling tools. Available tools:

- dataset_info(path)       inspect an HDF5 ann-benchmarks dataset
- index_build(dataset_path, algorithm, ...) build an in-memory pyvsag index
- index_search(handle, topk, ef_search, num_queries) recall@k + latency

Rules:
1. Always start by calling dataset_info to confirm the dataset before you
   build anything; never guess dim/metric/n.
2. Build with the dataset's training partition. Use num_elements only when
   the user explicitly asks to subsample.
3. After build, always run index_search and report recall@k together with
   latency. Answer succinctly with a single short paragraph + the
   numeric metrics.
4. Do not invent tool calls or arguments not listed above. If you do not
   know, say so.
"""


def run_loop(cfg: ProviderConfig, goal: str, max_steps: int = 8) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": goal},
    ]
    tool_specs = _tool_specs_for_chat()
    trace: List[Dict[str, Any]] = []
    for step in range(max_steps):
        print(f"\n=== step {step + 1} (provider={cfg.name}, model={cfg.model}) ===", flush=True)
        t0 = time.perf_counter()
        try:
            resp = chat_once(cfg, messages, tool_specs)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")
            return {"ok": False, "error": f"HTTP {exc.code}: {body[:500]}", "trace": trace}
        except urllib.error.URLError as exc:
            return {"ok": False, "error": f"network: {exc}", "trace": trace}
        elapsed = time.perf_counter() - t0
        choice = resp["choices"][0]["message"]
        trace.append({"step": step + 1, "elapsed_s": round(elapsed, 2), "message": choice})
        messages.append(choice)
        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            content = choice.get("content") or ""
            print(f"[final, {elapsed:.1f}s]\n{content}", flush=True)
            return {"ok": True, "final": content, "trace": trace, "messages": messages}
        for call in tool_calls:
            name = call["function"]["name"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError as exc:
                args = {}
                print(f"[warn] could not parse args for {name}: {exc}", flush=True)
            print(f"-> tool_call {name}({json.dumps(args)[:200]})", flush=True)
            ts = time.perf_counter()
            result = tools.call_tool(name, args)
            print(f"<- result ({time.perf_counter() - ts:.1f}s): "
                  f"{json.dumps(result, default=str)[:300]}", flush=True)
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "name": name,
                "content": json.dumps(result, default=str),
            })
            trace.append({"step": step + 1, "tool": name, "args": args, "result": result})
    return {"ok": False, "error": f"exceeded {max_steps} steps", "trace": trace}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="VSAG-Code stage-1 spike")
    parser.add_argument("--provider", choices=["openai", "deepseek", "copilot"], required=True)
    parser.add_argument(
        "--goal",
        default=("Inspect /data/datasets/sift-128-euclidean.hdf5, build a hgraph "
                 "index on it, then report recall@10 and latency over 1000 queries."),
        help="Natural-language task description for the LLM.",
    )
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--out", type=str, default=None,
                        help="Write JSON trace to this path.")
    args = parser.parse_args(argv)

    cfg = resolve_provider(args.provider)
    print(f"[spike] provider={cfg.name} model={cfg.model} url={_chat_completions_url(cfg)}",
          flush=True)
    print(f"[spike] goal: {args.goal}", flush=True)

    result = run_loop(cfg, args.goal, max_steps=args.max_steps)
    print(f"\n[spike] ok={result['ok']}", flush=True)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Drop the live messages buffer (already redundant with trace).
        slim = {k: v for k, v in result.items() if k != "messages"}
        slim["provider"] = asdict(cfg) | {"auth_header": "<redacted>"}
        out_path.write_text(json.dumps(slim, indent=2, default=str))
        print(f"[spike] trace -> {out_path}", flush=True)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
