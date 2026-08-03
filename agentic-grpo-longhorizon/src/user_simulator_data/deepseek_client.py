"""Minimal, secret-safe client for the official DeepSeek Chat Completions API."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-flash"
Transport = Callable[[str, Mapping[str, str], Mapping[str, Any], float], Mapping[str, Any]]


def _urllib_transport(
    url: str,
    headers: Mapping[str, str],
    payload: Mapping[str, Any],
    timeout: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=dict(headers),
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"DeepSeek HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"DeepSeek connection failed: {exc.reason}") from exc


@dataclass
class DeepSeekClient:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 180.0
    max_attempts: int = 3
    transport: Transport = _urllib_transport

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise RuntimeError("DEEPSEEK_API_KEY is empty")
        if self.model not in {"deepseek-v4-flash", "deepseek-v4-pro"}:
            raise ValueError(f"unsupported DeepSeek model: {self.model}")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")

    @classmethod
    def from_environment(
        cls,
        *,
        env_name: str = "DEEPSEEK_API_KEY",
        base_url: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        **kwargs: Any,
    ) -> "DeepSeekClient":
        api_key = os.environ.get(env_name, "").strip()
        if not api_key:
            raise RuntimeError(
                f"{env_name} is not set; export it in the runtime shell. "
                "Never place API keys in configs, test fixtures, or command history."
            )
        return cls(
            api_key=api_key,
            base_url=(base_url or os.environ.get("DEEPSEEK_BASE_URL") or DEFAULT_BASE_URL),
            model=model,
            **kwargs,
        )

    def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1600,
        thinking: bool = True,
        reasoning_effort: str = "high",
    ) -> dict[str, Any]:
        """Request one JSON object and discard provider chain-of-thought."""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "max_tokens": int(max_tokens),
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled" if thinking else "disabled"},
        }
        # DeepSeek documents that temperature/top_p are ignored in thinking mode.
        if thinking:
            payload["reasoning_effort"] = reasoning_effort
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        url = self.base_url.rstrip("/") + "/chat/completions"

        last_error: Optional[Exception] = None
        response: Optional[Mapping[str, Any]] = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = self.transport(url, headers, payload, self.timeout_seconds)
                break
            except RuntimeError as exc:
                last_error = exc
                if attempt == self.max_attempts:
                    raise
                time.sleep(min(2 ** (attempt - 1), 8))
        if response is None:
            raise RuntimeError(f"DeepSeek request failed: {last_error}")

        try:
            message = response["choices"][0]["message"]
            content = str(message["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("DeepSeek response is missing choices[0].message.content") from exc
        if not content.strip():
            raise RuntimeError("DeepSeek JSON mode returned empty content")

        # Do not persist reasoning_content.  The structured evidence fields in the
        # final JSON are sufficient for reproducible QA without retaining private CoT.
        return {
            "content": content,
            "model": response.get("model", self.model),
            "response_id": response.get("id"),
            "usage": dict(response.get("usage") or {}),
        }
