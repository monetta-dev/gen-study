"""OpenAI-compatible chat-completions client."""

from __future__ import annotations

import json
import os
import http.client
import time
import urllib.error
import urllib.request
from typing import Any

OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"

OPENCODE_GO_ANTHROPIC_MODELS = {
    "minimax-m3",
    "minimax-m2.7",
    "minimax-m2.5",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
}

OPENCODE_GO_OPENAI_MODELS = {
    "glm-5.1",
    "glm-5",
    "kimi-k2.5",
    "kimi-k2.6",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "mimo-v2.5",
    "mimo-v2.5-pro",
}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise DeepSeekError(f"{name} must be an integer") from exc


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise DeepSeekError(f"{name} must be a number") from exc


class DeepSeekError(RuntimeError):
    """Raised when the configured LLM API request fails."""


class DeepSeekClient:
    """Small OpenAI-compatible client for model chat completions."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-pro"
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("LLM_TIMEOUT_SECONDS")
            or os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "300")
        )
        self.api_style = self._resolve_api_style()
        default_max_tokens = 4096 if self._uses_opencode_go_messages_api() else 8192
        default_thinking_budget = 1024 if self._uses_opencode_go_messages_api() else 4096
        self.max_tokens = _env_int("LLM_MAX_TOKENS", default_max_tokens)
        self.thinking_budget_tokens = _env_int(
            "LLM_THINKING_BUDGET_TOKENS",
            default_thinking_budget,
        )
        self.retry_attempts = _env_int("LLM_RETRY_ATTEMPTS", 1)
        self.retry_delay_seconds = _env_float("LLM_RETRY_DELAY_SECONDS", 10.0)
        self.thinking_display = os.getenv("LLM_THINKING_DISPLAY", "omitted").strip().lower()
        self.thinking_mode = os.getenv("LLM_THINKING_MODE", "auto").strip().lower()
        self.thinking_field = self._resolve_thinking_field()
        self.stream = self._resolve_streaming_enabled()
        self.user_agent = os.getenv(
            "LLM_USER_AGENT",
            "gen-study/0.1 Python LLM client",
        ).strip()

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise DeepSeekError("LLM_API_KEY or DEEPSEEK_API_KEY is not configured")
        if self.api_style == "anthropic":
            return self._chat_anthropic(messages)
        if self.api_style != "openai":
            raise DeepSeekError("LLM_API_STYLE must be `openai` or `anthropic`")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
        }
        self._apply_thinking_config(payload)
        data = json.dumps(payload).encode("utf-8")

        def make_request() -> urllib.request.Request:
            return urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": self.user_agent,
                },
                method="POST",
            )

        response_body = self._open_with_retries(
            make_request,
            "chat-completions",
        )

        try:
            decoded: dict[str, Any] = json.loads(response_body)
            return decoded["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise DeepSeekError("LLM API response did not match chat-completions format") from exc

    def _chat_anthropic(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0.4,
            "system": self._anthropic_system_prompt(messages),
            "messages": self._anthropic_messages(messages),
        }
        self._apply_thinking_config(payload)
        if self.stream:
            payload["stream"] = True
        data = json.dumps(payload).encode("utf-8")

        def make_request() -> urllib.request.Request:
            return urllib.request.Request(
                f"{self.base_url}/messages",
                data=data,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": os.getenv("LLM_ANTHROPIC_VERSION", "2023-06-01"),
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": self.user_agent,
                },
                method="POST",
            )

        if self.stream:
            return self._open_stream_with_retries(make_request, "messages")

        response_body = self._open_with_retries(make_request, "messages")

        try:
            decoded: dict[str, Any] = json.loads(response_body)
            content = decoded["content"]
            if isinstance(content, list):
                text_parts = [
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                return "".join(text_parts)
            if isinstance(content, str):
                return content
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise DeepSeekError("LLM API response did not match messages format") from exc
        raise DeepSeekError("LLM API response did not include text content")

    def _open_stream_with_retries(
        self,
        make_request,
        response_format: str,
    ) -> str:
        for attempt in range(self.retry_attempts + 1):
            try:
                with urllib.request.urlopen(
                    make_request(),
                    timeout=self.timeout_seconds,
                ) as response:
                    return self._read_anthropic_sse(response)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                exc.close()
                if not self._is_retryable_http_error(exc.code, body):
                    raise DeepSeekError(
                        f"LLM API returned HTTP {exc.code}: {body}"
                    ) from exc
                if attempt >= self.retry_attempts:
                    raise DeepSeekError(
                        f"LLM API returned HTTP {exc.code}: {body}"
                    ) from exc
                delay = self._retry_delay(body)
                print(
                    f"Retrying streaming LLM {response_format} request after "
                    f"HTTP {exc.code} in {delay:g}s "
                    f"(attempt {attempt + 1}/{self.retry_attempts})",
                    flush=True,
                )
                time.sleep(delay)
            except urllib.error.URLError as exc:
                raise DeepSeekError(f"LLM API request failed: {exc.reason}") from exc
            except (http.client.HTTPException, TimeoutError, OSError) as exc:
                raise DeepSeekError(f"LLM API connection failed: {exc}") from exc
        raise DeepSeekError("LLM streaming request failed")

    def _read_anthropic_sse(self, response) -> str:
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
                raise DeepSeekError(f"LLM streaming error: {message}")
            delta = event.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                text_parts.append(delta.get("text", ""))
        text = "".join(text_parts)
        if not text:
            raise DeepSeekError("LLM streaming response did not include text content")
        return text

    def _open_with_retries(
        self,
        make_request,
        response_format: str,
    ) -> str:
        last_http_error: urllib.error.HTTPError | None = None
        for attempt in range(self.retry_attempts + 1):
            try:
                with urllib.request.urlopen(
                    make_request(),
                    timeout=self.timeout_seconds,
                ) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                exc.close()
                if not self._is_retryable_http_error(exc.code, body):
                    raise DeepSeekError(
                        f"LLM API returned HTTP {exc.code}: {body}"
                    ) from exc
                last_http_error = exc
                if attempt >= self.retry_attempts:
                    raise DeepSeekError(
                        f"LLM API returned HTTP {exc.code}: {body}"
                    ) from exc
                delay = self._retry_delay(body)
                print(
                    f"Retrying LLM {response_format} request after HTTP {exc.code} "
                    f"in {delay:g}s (attempt {attempt + 1}/{self.retry_attempts})",
                    flush=True,
                )
                time.sleep(delay)
            except urllib.error.URLError as exc:
                raise DeepSeekError(f"LLM API request failed: {exc.reason}") from exc
            except (http.client.HTTPException, TimeoutError, OSError) as exc:
                raise DeepSeekError(f"LLM API connection failed: {exc}") from exc
        if last_http_error is not None:
            raise DeepSeekError(f"LLM API returned HTTP {last_http_error.code}")
        raise DeepSeekError("LLM API request failed")

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

    def _anthropic_system_prompt(self, messages: list[dict[str, str]]) -> str:
        return "\n\n".join(
            message["content"]
            for message in messages
            if message.get("role") == "system" and message.get("content")
        )

    def _anthropic_messages(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        converted = []
        for message in messages:
            role = message.get("role")
            if role == "system":
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            converted.append({"role": role, "content": message.get("content", "")})
        return converted

    def _apply_thinking_config(self, payload: dict[str, Any]) -> None:
        if self.thinking_mode in {"", "disabled", "false", "0", "off", "none"}:
            return

        field = self.thinking_field
        if not field:
            if self.thinking_mode in {"enabled", "required", "true", "1", "on"}:
                raise DeepSeekError(
                    "Thinking is enabled, but no supported thinking field is known "
                    "for this provider/model. Set LLM_THINKING_FIELD explicitly."
                )
            return

        if field == "anthropic_thinking":
            if self.thinking_budget_tokens >= self.max_tokens:
                raise DeepSeekError(
                    "LLM_THINKING_BUDGET_TOKENS must be less than LLM_MAX_TOKENS"
                )
            thinking = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }
            if self.thinking_display in {"omitted", "summarized"}:
                thinking["display"] = self.thinking_display
            payload["thinking"] = thinking
        elif field == "thinking":
            payload["thinking"] = {"type": "enabled"}
        elif field == "enable_thinking":
            payload["enable_thinking"] = True
        else:
            payload[field] = True

    def _resolve_api_style(self) -> str:
        configured = os.getenv("LLM_API_STYLE", "").strip().lower()
        if configured:
            return configured
        if self.provider == "opencode-go":
            if not os.getenv("LLM_BASE_URL"):
                self.base_url = OPENCODE_GO_BASE_URL
            if self.model in OPENCODE_GO_ANTHROPIC_MODELS:
                return "anthropic"
            if self.model in OPENCODE_GO_OPENAI_MODELS:
                return "openai"
        return "openai"

    def _uses_opencode_go_messages_api(self) -> bool:
        return self.provider == "opencode-go" and self.api_style == "anthropic"

    def _resolve_streaming_enabled(self) -> bool:
        configured = os.getenv("LLM_STREAM", "").strip().lower()
        if configured:
            return configured in {"1", "true", "yes", "on"}
        return self._uses_opencode_go_messages_api()

    def _resolve_thinking_field(self) -> str:
        configured = os.getenv("LLM_THINKING_FIELD", "").strip()
        if configured:
            return configured
        if self.thinking_mode in {"", "disabled", "false", "0", "off", "none"}:
            return ""
        if self.api_style == "anthropic":
            return "anthropic_thinking"
        if self.provider == "deepseek":
            return "thinking"
        return ""
