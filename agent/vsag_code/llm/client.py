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

"""Provider-agnostic chat client.

Resolves a :class:`ProviderConfig` from the chosen provider name +
environment, exposes a single ``chat_completions`` call that takes
OpenAI-style ``messages`` + ``tools`` and returns a parsed dict.

Supported providers (matching the spike + Stage-1 RESULTS):

* ``openai``    OpenAI proper (or any OpenAI-compatible endpoint via
                ``OPENAI_BASE_URL``: vLLM / Ollama / lm-studio).
* ``deepseek``  Alias for ``openai`` with ``api.deepseek.com/v1``.
* ``copilot``   GitHub Copilot via OAuth device flow; long-lived github
                token cached at ``$VSAG_CODE_HOME/copilot-token.json``,
                exchanged for a short-lived API token before each call.
* ``anthropic`` Adapter to Anthropic Messages API; tool-calls translated
                to/from OpenAI-style on the fly.
* ``ollama``    Local Ollama server; uses ``/api/chat`` translated to
                OpenAI shape. Mostly useful for offline development.

Env vars consulted (per provider):

    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
    COPILOT_MODEL, VSAG_CODE_HOME
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL
    OLLAMA_BASE_URL, OLLAMA_MODEL
    HTTP_PROXY/HTTPS_PROXY (used automatically by urllib)
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Provider config
# ---------------------------------------------------------------------------


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    model: str
    auth_header: str
    extra_headers: Dict[str, str]
    flavor: str = "openai"  # "openai" or "anthropic"


def _vsag_code_home() -> Path:
    home = Path(os.environ.get("VSAG_CODE_HOME", "/workspace/.vsag-code"))
    home.mkdir(parents=True, exist_ok=True)
    return home


def _resolve_openai(provider: str, model_override: Optional[str] = None) -> ProviderConfig:
    if provider == "deepseek":
        api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
        base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        model = model_override or os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    elif provider == "ollama":
        api_key = os.environ.get("OLLAMA_API_KEY", "ollama")
        base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model = model_override or os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    else:
        api_key = os.environ.get("OPENAI_API_KEY")
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        model = model_override or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
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
        flavor="openai",
    )


def _resolve_anthropic(model_override: Optional[str] = None) -> ProviderConfig:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("missing ANTHROPIC_API_KEY")
    return ProviderConfig(
        name="anthropic",
        base_url=os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/"),
        model=model_override or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        auth_header=f"x-api-key {api_key}",  # actual header set later
        extra_headers={"anthropic-version": "2023-06-01"},
        flavor="anthropic",
    )


# ---------------------------------------------------------------------------
# Copilot device flow
# ---------------------------------------------------------------------------


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


def _http_post_form(
    url: str, form: Dict[str, str], headers: Dict[str, str]
) -> Dict[str, Any]:
    body = "&".join(f"{k}={urllib.request.quote(v)}" for k, v in form.items()).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _http_get_json(url: str, headers: Dict[str, str]) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _copilot_oauth_login(printer=print) -> str:
    init = _http_post_form(
        _COPILOT_DEVICE_URL,
        {"client_id": _COPILOT_CLIENT_ID, "scope": "read:user"},
        {"Accept": "application/json"},
    )
    printer(f"[copilot] visit {init['verification_uri']}")
    printer(f"[copilot] enter code: {init['user_code']}")
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
        except urllib.error.URLError as exc:
            printer(f"[copilot] poll error: {exc}; retrying")
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


def _copilot_load_or_login(printer=print) -> str:
    cache = _vsag_code_home() / "copilot-token.json"
    if cache.is_file():
        try:
            return json.loads(cache.read_text())["github_token"]
        except (KeyError, json.JSONDecodeError):
            pass
    token = _copilot_oauth_login(printer=printer)
    cache.write_text(json.dumps({"github_token": token}))
    cache.chmod(0o600)
    return token


def _copilot_exchange_api_token(github_token: str) -> Tuple[str, int]:
    headers = {
        "Authorization": f"token {github_token}",
        "Accept": "application/json",
        **_COPILOT_EDITOR_HEADERS,
    }
    res = _http_get_json(_COPILOT_API_TOKEN_URL, headers)
    return res["token"], int(res.get("expires_at", time.time() + 1500))


def _resolve_copilot(model_override: Optional[str] = None, printer=print) -> ProviderConfig:
    github_token = _copilot_load_or_login(printer=printer)
    api_token, _ = _copilot_exchange_api_token(github_token)
    model = model_override or os.environ.get("COPILOT_MODEL", "gpt-4o")
    return ProviderConfig(
        name="copilot",
        base_url=_COPILOT_CHAT_URL.rsplit("/chat/", 1)[0],
        model=model,
        auth_header=f"Bearer {api_token}",
        extra_headers=_COPILOT_EDITOR_HEADERS,
        flavor="openai",
    )


# ---------------------------------------------------------------------------
# Public resolver
# ---------------------------------------------------------------------------


_PROVIDERS = ("openai", "deepseek", "copilot", "anthropic", "ollama")


def resolve_provider(
    provider: str, model: Optional[str] = None, printer=print
) -> ProviderConfig:
    if provider in ("openai", "deepseek", "ollama"):
        return _resolve_openai(provider, model)
    if provider == "copilot":
        return _resolve_copilot(model, printer=printer)
    if provider == "anthropic":
        return _resolve_anthropic(model)
    raise RuntimeError(f"unknown provider: {provider}; choose from {_PROVIDERS}")


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------


def chat_completions_url(cfg: ProviderConfig) -> str:
    if cfg.name == "copilot":
        return _COPILOT_CHAT_URL
    if cfg.flavor == "anthropic":
        return f"{cfg.base_url}/v1/messages"
    return f"{cfg.base_url}/chat/completions"


def chat_completions(
    cfg: ProviderConfig,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    *,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """Issue a single chat-completions call. Returns the raw provider dict.

    For ``flavor='openai'`` the response shape matches OpenAI's chat
    completion API. For ``flavor='anthropic'`` the response is
    translated into the OpenAI shape so the agent loop stays uniform.
    """
    if cfg.flavor == "anthropic":
        return _chat_completions_anthropic(cfg, messages, tools, temperature, timeout)
    return _chat_completions_openai(cfg, messages, tools, temperature, timeout)


def _chat_completions_openai(
    cfg: ProviderConfig,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    temperature: float,
    timeout: float,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    headers = {
        "Authorization": cfg.auth_header,
        "Content-Type": "application/json",
        "Accept": "application/json",
        **cfg.extra_headers,
    }
    req = urllib.request.Request(
        chat_completions_url(cfg),
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _chat_completions_anthropic(
    cfg: ProviderConfig,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    temperature: float,
    timeout: float,
) -> Dict[str, Any]:
    """Translate OpenAI-shaped messages -> Anthropic Messages API and back.

    Anthropic uses ``input_schema`` (not ``parameters``), separate
    ``tool_use`` blocks (not ``tool_calls``), and a top-level ``system``
    string instead of a system message. We translate both directions so
    the agent loop never sees Anthropic-specific shapes.
    """
    system_parts: List[str] = []
    a_msgs: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system_parts.append(m.get("content", "") or "")
            continue
        if role == "tool":
            a_msgs.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m["tool_call_id"],
                            "content": m.get("content", ""),
                        }
                    ],
                }
            )
            continue
        if role == "assistant" and m.get("tool_calls"):
            blocks: List[Dict[str, Any]] = []
            if m.get("content"):
                blocks.append({"type": "text", "text": m["content"]})
            for tc in m["tool_calls"]:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc["id"],
                        "name": tc["function"]["name"],
                        "input": json.loads(tc["function"].get("arguments") or "{}"),
                    }
                )
            a_msgs.append({"role": "assistant", "content": blocks})
            continue
        # Plain user / assistant text
        a_msgs.append({"role": role or "user", "content": m.get("content", "") or ""})

    a_tools = [
        {
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "input_schema": t["function"]["parameters"],
        }
        for t in tools
    ]

    payload: Dict[str, Any] = {
        "model": cfg.model,
        "system": "\n\n".join(p for p in system_parts if p),
        "messages": a_msgs,
        "max_tokens": 2048,
        "temperature": temperature,
    }
    if a_tools:
        payload["tools"] = a_tools

    headers = {
        # Anthropic uses x-api-key, not Bearer.
        "x-api-key": cfg.auth_header.removeprefix("x-api-key ").strip(),
        "Content-Type": "application/json",
        "Accept": "application/json",
        **cfg.extra_headers,
    }
    req = urllib.request.Request(
        chat_completions_url(cfg),
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = json.loads(resp.read().decode())

    # Translate Anthropic -> OpenAI choice shape.
    text_parts: List[str] = []
    tool_calls: List[Dict[str, Any]] = []
    for block in raw.get("content", []):
        if block.get("type") == "text":
            text_parts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
    msg: Dict[str, Any] = {"role": "assistant", "content": "".join(text_parts) or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {
        "id": raw.get("id"),
        "model": raw.get("model"),
        "choices": [{"index": 0, "message": msg, "finish_reason": raw.get("stop_reason")}],
    }
