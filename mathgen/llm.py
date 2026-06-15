"""OpenCode Go Qwen client."""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"
QWEN_MAX_MODEL = "qwen3.7-max"
QWEN_PLUS_MODEL = "qwen3.7-plus"
QWEN_MODELS = {QWEN_MAX_MODEL, QWEN_PLUS_MODEL}
QWEN_MODEL = QWEN_MAX_MODEL


class LLMError(RuntimeError):
    """Raised when the OpenCode Go API request fails."""


class EmptyStreamError(LLMError):
    """Raised when a streaming response completes without text content."""


def _env_string(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    value = _env_string(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise LLMError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    value = _env_string(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise LLMError(f"{name} must be a number") from exc


def _env_flag(name: str, default: bool) -> bool:
    value = _env_string(name).lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


class OpenCodeGoClient:
    """Qwen 3.7 client for OpenCode Go's Anthropic-compatible endpoint."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        max_tokens: int | None = None,
        thinking_budget_tokens: int | None = None,
        thinking_enabled: bool | None = None,
        stream: bool | None = None,
        model: str | None = None,
        **_unused: Any,
    ) -> None:
        self.api_key = api_key or _env_string("OPENCODE_GO_API_KEY")
        self.base_url = (base_url or _env_string("OPENCODE_GO_BASE_URL", OPENCODE_GO_BASE_URL)).rstrip("/")
        self.model = self._configured_model(model)
        self.timeout_seconds = timeout_seconds or _env_float("OPENCODE_GO_TIMEOUT_SECONDS", 300.0)
        self.max_tokens = max_tokens or _env_int("OPENCODE_GO_MAX_TOKENS", 32768)
        self.thinking_budget_tokens = (
            thinking_budget_tokens
            if thinking_budget_tokens is not None
            else _env_int("OPENCODE_GO_THINKING_BUDGET_TOKENS", 16384)
        )
        self.thinking_display = _env_string("OPENCODE_GO_THINKING_DISPLAY", "omitted").lower()
        self.thinking_enabled = (
            thinking_enabled
            if thinking_enabled is not None
            else _env_flag("OPENCODE_GO_THINKING_ENABLED", True)
        )
        self.stream = stream if stream is not None else _env_flag("OPENCODE_GO_STREAM", True)
        self.retry_attempts = _env_int("OPENCODE_GO_RETRY_ATTEMPTS", 3)
        self.retry_delay_seconds = _env_float("OPENCODE_GO_RETRY_DELAY_SECONDS", 10.0)
        self.user_agent = _env_string("OPENCODE_GO_USER_AGENT", "gen-study/0.1 Qwen client")

    def _configured_model(self, model: str | None) -> str:
        configured_model = model or _env_string("OPENCODE_GO_MODEL", QWEN_MODEL)
        if configured_model not in QWEN_MODELS:
            allowed = ", ".join(sorted(QWEN_MODELS))
            raise LLMError(f"OPENCODE_GO_MODEL must be one of: {allowed}")
        return configured_model

    def chat(self, messages: list[dict[str, Any]]) -> str:
        if not self.api_key:
            raise LLMError("OPENCODE_GO_API_KEY is not configured")
        return self._chat_messages(messages)

    def _chat_messages(self, messages: list[dict[str, Any]]) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.4,
            "system": self._system_prompt(messages),
            "messages": self._messages(messages),
        }
        self._apply_thinking(payload)
        if self.stream:
            payload["stream"] = True
        data = json.dumps(payload).encode("utf-8")

        def make_request() -> urllib.request.Request:
            return urllib.request.Request(
                f"{self.base_url}/messages",
                data=data,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": _env_string("OPENCODE_GO_ANTHROPIC_VERSION", "2023-06-01"),
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": self.user_agent,
                },
                method="POST",
            )

        if self.stream:
            return self._open_stream_with_retries(make_request, self._read_messages_sse)

        response_body = self._open_with_retries(make_request)
        try:
            decoded: dict[str, Any] = json.loads(response_body)
            content = decoded["content"]
            if isinstance(content, list):
                return "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
            if isinstance(content, str):
                return content
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError("OpenCode Go response did not match messages format") from exc
        raise LLMError("OpenCode Go response did not include text content")

    def _open_stream_with_retries(self, make_request, reader) -> str:
        for attempt in range(self.retry_attempts + 1):
            try:
                with urllib.request.urlopen(make_request(), timeout=self.timeout_seconds) as response:
                    return reader(response)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                exc.close()
                if not self._is_retryable_http_error(exc.code, body) or attempt >= self.retry_attempts:
                    raise LLMError(f"OpenCode Go returned HTTP {exc.code}: {body}") from exc
                delay = self._retry_delay(body)
                print(
                    f"Retrying streaming OpenCode Go messages request after "
                    f"HTTP {exc.code} in {delay:g}s "
                    f"(attempt {attempt + 1}/{self.retry_attempts})",
                    flush=True,
                )
                time.sleep(delay)
            except urllib.error.URLError as exc:
                raise LLMError(f"OpenCode Go request failed: {exc.reason}") from exc
            except EmptyStreamError as exc:
                if attempt >= self.retry_attempts:
                    raise LLMError(str(exc)) from exc
                print(
                    f"Retrying streaming OpenCode Go messages request after "
                    f"empty text stream ({exc}) in {self.retry_delay_seconds:g}s "
                    f"(attempt {attempt + 1}/{self.retry_attempts})",
                    flush=True,
                )
                time.sleep(self.retry_delay_seconds)
            except (http.client.HTTPException, TimeoutError, OSError) as exc:
                raise LLMError(f"OpenCode Go connection failed: {exc}") from exc
        raise LLMError("OpenCode Go streaming request failed")

    def _open_with_retries(self, make_request) -> str:
        for attempt in range(self.retry_attempts + 1):
            try:
                with urllib.request.urlopen(make_request(), timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                exc.close()
                if not self._is_retryable_http_error(exc.code, body) or attempt >= self.retry_attempts:
                    raise LLMError(f"OpenCode Go returned HTTP {exc.code}: {body}") from exc
                delay = self._retry_delay(body)
                print(
                    f"Retrying OpenCode Go messages request after HTTP {exc.code} "
                    f"in {delay:g}s (attempt {attempt + 1}/{self.retry_attempts})",
                    flush=True,
                )
                time.sleep(delay)
            except urllib.error.URLError as exc:
                raise LLMError(f"OpenCode Go request failed: {exc.reason}") from exc
            except (http.client.HTTPException, TimeoutError, OSError) as exc:
                raise LLMError(f"OpenCode Go connection failed: {exc}") from exc
        raise LLMError("OpenCode Go request failed")

    def _read_messages_sse(self, response) -> str:
        text_parts = []
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "error":
                error = event.get("error", {})
                message = error.get("message") or json.dumps(error, ensure_ascii=False)
                raise LLMError(f"OpenCode Go streaming error: {message}")
            if event.get("type") == "content_block_start":
                block = event.get("content_block")
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            delta = event.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                text_parts.append(delta.get("text", ""))
        text = "".join(text_parts)
        if not text:
            raise EmptyStreamError("OpenCode Go streaming response did not include text content")
        return text

    def _system_prompt(self, messages: list[dict[str, Any]]) -> str:
        return "\n\n".join(
            message["content"]
            for message in messages
            if message.get("role") == "system"
            and isinstance(message.get("content"), str)
            and message.get("content")
        )

    def _messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            converted.append({"role": role, "content": message.get("content", "")})
        return converted

    def _apply_thinking(self, payload: dict[str, Any]) -> None:
        if not self.thinking_enabled:
            return
        if self.thinking_budget_tokens >= self.max_tokens:
            raise LLMError(
                "OPENCODE_GO_THINKING_BUDGET_TOKENS must be less than "
                "OPENCODE_GO_MAX_TOKENS"
            )
        thinking = {
            "type": "enabled",
            "budget_tokens": self.thinking_budget_tokens,
        }
        if self.thinking_display in {"omitted", "summarized"}:
            thinking["display"] = self.thinking_display
        payload["thinking"] = thinking

    def _is_retryable_http_error(self, code: int, body: str) -> bool:
        if code in {408, 409, 425, 429, 500, 502, 503, 504, 524}:
            return True
        return '"retryable":true' in body.replace(" ", "").lower()

    def _retry_delay(self, body: str) -> float:
        try:
            decoded = json.loads(body)
        except json.JSONDecodeError:
            return self.retry_delay_seconds
        retry_after = decoded.get("retry_after")
        if isinstance(retry_after, (int, float)) and retry_after >= 0:
            return min(float(retry_after), self.retry_delay_seconds)
        return self.retry_delay_seconds
