"""LLM backend interface (§6.1).

The two LLM-assisted steps — instruction back-generation (§6) and LLM-as-judge
eval (§8) — go through this interface so a local model can replace the API.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMBackend(Protocol):
    def complete(self, system: str, user: str, **kw) -> str:
        """Return the model's text completion for a system+user prompt."""
        ...


class AnthropicBackend:
    """Default backend, using ANTHROPIC_API_KEY.

    The ``anthropic`` package is imported lazily so the verifier and data-tagging
    paths never require it. Install with ``pip install -e ".[llm]"``.
    """

    def __init__(self, model: str = "claude-haiku-4-5", max_tokens: int = 1024):
        self.model = model
        self.max_tokens = max_tokens
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - env-dependent
                raise ImportError(
                    "anthropic is required for AnthropicBackend: "
                    'pip install -e ".[llm]"'
                ) from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is not set")
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, user: str, **kw) -> str:
        client = self._get_client()
        resp = client.messages.create(
            model=kw.get("model", self.model),
            max_tokens=kw.get("max_tokens", self.max_tokens),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


class OllamaBackend:
    """Open-source local model via Ollama's native chat API (no key, offline).

    Talks to ``/api/chat`` with stdlib only — no extra dependency. Point ``model``
    at a pulled model (e.g. ``aya-expanse``, ``qwen2.5``) that handles Kyrgyz.
    """

    def __init__(self, model: str = "aya-expanse",
                 host: str = "http://localhost:11434", options: dict | None = None):
        self.model = model
        self.host = host.rstrip("/")
        self.options = options or {}

    def complete(self, system: str, user: str, **kw) -> str:
        import json
        import urllib.request

        body = {
            "model": kw.get("model", self.model),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {**self.options, **kw.get("options", {})},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=kw.get("timeout", 120)) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"]


class FakeBackend:
    """Deterministic backend for tests and offline dry-runs.

    Either return a fixed string, or a caller-supplied function of (system, user).
    """

    def __init__(self, reply="", fn=None):
        self._reply = reply
        self._fn = fn
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str, **kw) -> str:
        self.calls.append((system, user))
        if self._fn is not None:
            return self._fn(system, user)
        return self._reply
