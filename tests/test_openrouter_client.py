import json as _json

import pytest
import requests as _requests

from multicat.llm.openrouter_client import OpenRouterClient, parse_json_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_post(response: dict):
    """Return a fake requests.post returning one canned response.

    `response` has keys: status_code (int), json_body (dict | None), text (str | None).
    """
    calls: list = []

    def fake_post(url, *, json=None, headers=None, timeout=None):
        calls.append(json)
        status = response.get("status_code", 200)
        body = response.get("json_body")
        text = response.get("text", _json.dumps(body) if body else "")

        class FakeResp:
            status_code = status

            def raise_for_status(self_):
                if status >= 400:
                    raise _requests.exceptions.HTTPError(f"HTTP {status}")

            def json(self_):
                if body is None:
                    raise _requests.exceptions.JSONDecodeError("", "", 0)
                return body

            @property
            def text(self_):
                return text

        return FakeResp()

    return fake_post, calls


def _ok_body(content: str, usage: dict | None = None) -> dict:
    body = {"choices": [{"message": {"content": content}}]}
    if usage:
        body["usage"] = usage
    return body


# ---------------------------------------------------------------------------
# parse_json_response
# ---------------------------------------------------------------------------

def test_parse_json_response_accepts_plain_json():
    assert parse_json_response('{"decision": "keep", "reason": "fits"}') == {
        "decision": "keep",
        "reason": "fits",
    }


def test_parse_json_response_accepts_markdown_json_block():
    text = """Here is the result:\n\n```json\n{"decision": "skip", "reason": "review"}\n```\n"""
    assert parse_json_response(text) == {"decision": "skip", "reason": "review"}


# ---------------------------------------------------------------------------
# Successful call
# ---------------------------------------------------------------------------

def test_openrouter_client_returns_parsed_json_on_success(monkeypatch):
    fake_post, _ = _make_fake_post(
        {"status_code": 200, "json_body": _ok_body('{"decision": "keep", "reason": "ok"}')}
    )
    monkeypatch.setattr("multicat.llm.openrouter_client._requests.post", fake_post)
    client = OpenRouterClient(api_key="test", model="test-model")

    result = client.chat_json(system_prompt="system", user_prompt="user")

    assert result == {"decision": "keep", "reason": "ok"}


def test_openrouter_client_sends_max_tokens_to_avoid_truncated_json(monkeypatch):
    """Regression: extraction JSON was truncated when no max_tokens was set."""
    fake_post, calls = _make_fake_post(
        {"status_code": 200, "json_body": _ok_body('{"decision": "keep", "reason": "ok"}')}
    )
    monkeypatch.setattr("multicat.llm.openrouter_client._requests.post", fake_post)
    client = OpenRouterClient(api_key="test", model="test-model")
    client.chat_json(system_prompt="system", user_prompt="user")

    assert calls[0]["max_tokens"] == client.max_tokens
    assert calls[0]["max_tokens"] >= 8000


# ---------------------------------------------------------------------------
# Error handling (single attempt, no retries)
# ---------------------------------------------------------------------------

def test_openrouter_client_reports_non_json_api_response(monkeypatch):
    fake_post, _ = _make_fake_post({"status_code": 200, "json_body": None, "text": "<html>bad gateway</html>"})
    monkeypatch.setattr("multicat.llm.openrouter_client._requests.post", fake_post)
    client = OpenRouterClient(api_key="test", model="test-model")

    with pytest.raises(_requests.exceptions.JSONDecodeError):
        client.chat_json(system_prompt="system", user_prompt="user")


def test_openrouter_client_raises_on_malformed_json_content(monkeypatch):
    fake_post, _ = _make_fake_post(
        {"status_code": 200, "json_body": _ok_body('{"decision": unterminated')}
    )
    monkeypatch.setattr("multicat.llm.openrouter_client._requests.post", fake_post)
    client = OpenRouterClient(api_key="test", model="test-model")

    with pytest.raises(_json.JSONDecodeError):
        client.chat_json(system_prompt="system", user_prompt="user")


def test_openrouter_client_raises_on_http_error_status(monkeypatch):
    fake_post, _ = _make_fake_post({"status_code": 429, "json_body": None, "text": "rate limited"})
    monkeypatch.setattr("multicat.llm.openrouter_client._requests.post", fake_post)
    client = OpenRouterClient(api_key="test", model="test-model")

    with pytest.raises(_requests.exceptions.HTTPError):
        client.chat_json(system_prompt="system", user_prompt="user")


# ---------------------------------------------------------------------------
# Call metrics
# ---------------------------------------------------------------------------

def test_openrouter_client_emits_call_metrics_with_token_usage_on_success(monkeypatch):
    """Regression: every successful call must report token usage via on_call_metrics."""
    captured = []
    usage = {"prompt_tokens": 1200, "completion_tokens": 340, "total_tokens": 1540}
    fake_post, _ = _make_fake_post(
        {"status_code": 200, "json_body": _ok_body('{"decision": "keep", "reason": "ok"}', usage)}
    )
    monkeypatch.setattr("multicat.llm.openrouter_client._requests.post", fake_post)
    client = OpenRouterClient(api_key="test", model="test-model", on_call_metrics=captured.append)

    client.chat_json(system_prompt="system", user_prompt="user")

    assert len(captured) == 1
    m = captured[0]
    assert m.model == "test-model"
    assert m.prompt_tokens == 1200
    assert m.completion_tokens == 340
    assert m.total_tokens == 1540
    assert m.succeeded is True
    assert m.duration_seconds >= 0


def test_openrouter_client_emits_call_metrics_on_failure(monkeypatch):
    """Regression: failed calls must still emit metrics with succeeded=False."""
    captured = []
    fake_post, _ = _make_fake_post(
        {"status_code": 200, "json_body": _ok_body('{"decision": unterminated')}
    )
    monkeypatch.setattr("multicat.llm.openrouter_client._requests.post", fake_post)
    client = OpenRouterClient(api_key="test", model="test-model", on_call_metrics=captured.append)

    with pytest.raises(_json.JSONDecodeError):
        client.chat_json(system_prompt="system", user_prompt="user")

    assert len(captured) == 1
    m = captured[0]
    assert m.succeeded is False
    assert m.error is not None
