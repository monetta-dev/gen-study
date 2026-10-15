"""OpenRouter client for GPT-6.1 Sol and optional Qwen models."""

from __future__ import annotations

import http.client
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GPT_61_SOL_MODEL = "openai/gpt-6.1-sol"
DEFAULT_MODEL = GPT_61_SOL_MODEL
QWEN_MAX_MODEL = "qwen/qwen3.7-max"
QWEN_PLUS_MODEL = "qwen/qwen3.7-plus"
QWEN_MODELS = {QWEN_MAX_MODEL, QWEN_PLUS_MODEL}
QWEN_MODEL = QWEN_MAX_MODEL
SUPPORTED_MODELS = QWEN_MODELS | {GPT_61_SOL_MODEL}
MODEL_ALIASES = {
    "gpt-6.1-sol": GPT_61_SOL_MODEL,
    "qwen3.7-max": QWEN_MAX_MODEL,
    "qwen3.7-plus": QWEN_PLUS_MODEL,
}


class LLMError(RuntimeError):
    """Raised when the OpenRouter API request fails."""


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


class OpenRouterClient:
    """Model client for OpenRouter's OpenAI-compatible endpoint."""

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
        reasoning_effort: str | None = None,
        **_unused: Any,
    ) -> None:
        self.api_key = api_key or _env_string("OPENROUTER_API_KEY")
        self.base_url = (
            base_url or _env_string("OPENROUTER_BASE_URL", OPENROUTER_BASE_URL)
        ).rstrip("/")
        self.model = self._configured_model(model)
        self.reasoning_effort = (
            reasoning_effort or _env_string("OPENROUTER_REASONING_EFFORT") or "high"
        )
        if self.model == GPT_61_SOL_MODEL and self.reasoning_effort not in {
            "low", "medium", "high", "xhigh", "max"
        }:
            raise LLMError(
                "OPENROUTER_REASONING_EFFORT must be low, medium, high, xhigh, or max"
            )
        self.timeout_seconds = timeout_seconds or _env_float("OPENROUTER_TIMEOUT_SECONDS", 300.0)
        self.max_tokens = max_tokens or _env_int("OPENROUTER_MAX_TOKENS", 32768)
        self.thinking_budget_tokens = (
            thinking_budget_tokens
            if thinking_budget_tokens is not None
            else _env_int("OPENROUTER_THINKING_BUDGET_TOKENS", 16384)
        )
        self.thinking_enabled = (
            thinking_enabled
            if thinking_enabled is not None
            else _env_flag("OPENROUTER_THINKING_ENABLED", True)
        )
        self.stream = stream if stream is not None else _env_flag("OPENROUTER_STREAM", True)
        self.retry_attempts = _env_int("OPENROUTER_RETRY_ATTEMPTS", 3)
        self.retry_delay_seconds = _env_float("OPENROUTER_RETRY_DELAY_SECONDS", 10.0)

    def _configured_model(self, model: str | None) -> str:
        configured_model = model or _env_string("OPENROUTER_MODEL") or DEFAULT_MODEL
        configured_model = MODEL_ALIASES.get(configured_model, configured_model)
        if configured_model not in SUPPORTED_MODELS:
            allowed = ", ".join(sorted(SUPPORTED_MODELS))
            raise LLMError(f"OPENROUTER_MODEL must be one of: {allowed}")
        return configured_model

    def chat(self, messages: list[dict[str, Any]]) -> str:
        if not self.api_key:
            raise LLMError("OPENROUTER_API_KEY is not configured")
        if not self.base_url:
            raise LLMError("OPENROUTER_BASE_URL is not configured")
        if "{" in self.base_url or "}" in self.base_url:
            raise LLMError(
                "OPENROUTER_BASE_URL still contains a placeholder"
            )
        return self._chat_messages(messages)

    def _chat_messages(self, messages: list[dict[str, Any]]) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._messages(messages),
        }
        if self.model in QWEN_MODELS:
            payload["temperature"] = 0.4
        self._apply_thinking(payload)
        if self.stream:
            payload["stream"] = True
        data = json.dumps(payload).encode("utf-8")

        def make_request() -> urllib.request.Request:
            return urllib.request.Request(
                self._messages_url(),
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
                method="POST",
            )

        if self.stream:
            return self._open_stream_with_retries(
                make_request, self._read_chat_completions_sse
            )

        response_body = self._open_with_retries(make_request)
        try:
            decoded: dict[str, Any] = json.loads(response_body)
            content = decoded["choices"][0]["message"]["content"]
            if isinstance(content, str):
                return content
        except (IndexError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise LLMError(
                "OpenRouter response did not match chat completions format"
            ) from exc
        raise LLMError("OpenRouter response did not include text content")

    def _open_stream_with_retries(self, make_request, reader) -> str:
        for attempt in range(self.retry_attempts + 1):
            try:
                with urllib.request.urlopen(
                    make_request(), timeout=self.timeout_seconds
                ) as response:
                    return reader(response)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                exc.close()
                if (
                    not self._is_retryable_http_error(exc.code, body)
                    or attempt >= self.retry_attempts
                ):
                    raise LLMError(
                        f"OpenRouter returned HTTP {exc.code}: {body}"
                    ) from exc
                delay = self._retry_delay(body)
                print(
                    f"Retrying streaming OpenRouter messages request after "
                    f"HTTP {exc.code} in {delay:g}s "
                    f"(attempt {attempt + 1}/{self.retry_attempts})",
                    flush=True,
                )
                time.sleep(delay)
            except urllib.error.URLError as exc:
                raise LLMError(f"OpenRouter request failed: {exc.reason}") from exc
            except EmptyStreamError as exc:
                if attempt >= self.retry_attempts:
                    raise LLMError(str(exc)) from exc
                print(
                    f"Retrying streaming OpenRouter messages request after "
                    f"empty text stream ({exc}) in {self.retry_delay_seconds:g}s "
                    f"(attempt {attempt + 1}/{self.retry_attempts})",
                    flush=True,
                )
                time.sleep(self.retry_delay_seconds)
            except (http.client.HTTPException, TimeoutError, OSError) as exc:
                raise LLMError(f"OpenRouter connection failed: {exc}") from exc
        raise LLMError("OpenRouter streaming request failed")

    def _open_with_retries(self, make_request) -> str:
        for attempt in range(self.retry_attempts + 1):
            try:
                with urllib.request.urlopen(
                    make_request(), timeout=self.timeout_seconds
                ) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                exc.close()
                if (
                    not self._is_retryable_http_error(exc.code, body)
                    or attempt >= self.retry_attempts
                ):
                    raise LLMError(
                        f"OpenRouter returned HTTP {exc.code}: {body}"
                    ) from exc
                delay = self._retry_delay(body)
                print(
                    f"Retrying OpenRouter messages request after HTTP {exc.code} "
                    f"in {delay:g}s (attempt {attempt + 1}/{self.retry_attempts})",
                    flush=True,
                )
                time.sleep(delay)
            except urllib.error.URLError as exc:
                raise LLMError(f"OpenRouter request failed: {exc.reason}") from exc
            except (http.client.HTTPException, TimeoutError, OSError) as exc:
                raise LLMError(f"OpenRouter connection failed: {exc}") from exc
        raise LLMError("OpenRouter request failed")

    def _read_chat_completions_sse(self, response) -> str:
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
            if "error" in event:
                error = event.get("error", {})
                message = error.get("message") or json.dumps(error, ensure_ascii=False)
                raise LLMError(f"OpenRouter streaming error: {message}")
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            delta = choices[0].get("delta", {})
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                text_parts.append(delta["content"])
        text = "".join(text_parts)
        if not text:
            raise EmptyStreamError(
                "OpenRouter streaming response did not include text content"
            )
        return text

    def _messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted = []
        for message in messages:
            role = message.get("role")
            if role not in {"system", "user", "assistant"}:
                role = "user"
            converted.append(
                {"role": role, "content": self._content(message.get("content", ""))}
            )
        return converted

    def _content(self, content: Any) -> Any:
        if not isinstance(content, list):
            return content
        converted = []
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image":
                converted.append(block)
                continue
            source = block.get("source", {})
            if not isinstance(source, dict):
                converted.append(block)
                continue
            if source.get("type") == "base64":
                media_type = source.get("media_type", "image/png")
                url = f"data:{media_type};base64,{source.get('data', '')}"
            else:
                url = source.get("url", "")
            converted.append({"type": "image_url", "image_url": {"url": url}})
        return converted

    def _messages_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _apply_thinking(self, payload: dict[str, Any]) -> None:
        if self.model == GPT_61_SOL_MODEL:
            if not self.thinking_enabled:
                raise LLMError(
                    "GPT-6.1 Sol requires thinking; set OPENROUTER_THINKING_ENABLED=true "
                    "and use OPENROUTER_REASONING_EFFORT=low to reduce reasoning"
                )
            payload["reasoning"] = {"effort": self.reasoning_effort, "exclude": True}
            return
        if not self.thinking_enabled:
            payload["reasoning"] = {"enabled": False}
            return
        if self.thinking_budget_tokens <= 0:
            raise LLMError("OPENROUTER_THINKING_BUDGET_TOKENS must be positive")
        payload["reasoning"] = {
            "enabled": True,
            "max_tokens": self.thinking_budget_tokens,
            "exclude": True,
        }

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


# Preserve imports used by existing integrations.
OpenRouterQwenClient = OpenRouterClient
