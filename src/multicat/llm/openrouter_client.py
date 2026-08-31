from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable

import requests as _requests


@dataclass(frozen=True)
class CallMetrics:
    """Per-request metrics, for cost/runtime reporting (e.g. methods section
    of a paper: total tokens consumed and average call duration).
    """

    model: str
    duration_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    succeeded: bool
    error: str | None = None


@dataclass(frozen=True)
class OpenRouterClient:
    api_key: str
    model: str
    base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    timeout: int = 300
    max_tokens: int = 65536
    on_call_metrics: Callable[[CallMetrics], None] | None = field(default=None)

    def chat_json(self, *, system_prompt: str, user_prompt: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
        }
        return self._post_payload(payload)

    def _post_payload(self, payload: dict) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local/catalysis-chapter1",
            "X-Title": "Catalysis Chapter 1",
        }
        started_at = time.monotonic()
        try:
            response = _requests.post(
                self.base_url, json=payload, headers=headers, timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
            result = parse_json_response(content)
        except Exception as exc:
            self._emit_metrics(started_at, None, succeeded=False, error=str(exc))
            raise
        self._emit_metrics(started_at, usage, succeeded=True, error=None)
        return result

    def _emit_metrics(
        self,
        started_at: float,
        usage: dict | None,
        *,
        succeeded: bool,
        error: str | None,
    ) -> None:
        if self.on_call_metrics is None:
            return
        usage = usage or {}
        metrics = CallMetrics(
            model=self.model,
            duration_seconds=time.monotonic() - started_at,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            succeeded=succeeded,
            error=error,
        )
        self.on_call_metrics(metrics)


def parse_json_response(text: str) -> dict:
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    if match:
        stripped = match.group(1).strip()
    return json.loads(stripped)
