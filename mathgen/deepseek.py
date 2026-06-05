"""DeepSeek chat-completions client."""

from __future__ import annotations

import json
import os
import http.client
import urllib.error
import urllib.request
from typing import Any


class DeepSeekError(RuntimeError):
    """Raised when the DeepSeek API request fails."""


class DeepSeekClient:
    """Small OpenAI-compatible client for DeepSeek chat completions."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = (base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com").rstrip("/")
        self.model = model or os.getenv("DEEPSEEK_MODEL") or "deepseek-v4-pro"
        self.timeout_seconds = timeout_seconds or float(
            os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "300")
        )

    def chat(self, messages: list[dict[str, str]]) -> str:
        if not self.api_key:
            raise DeepSeekError("DEEPSEEK_API_KEY is not configured")

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.4,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled"},
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise DeepSeekError(f"DeepSeek API returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise DeepSeekError(f"DeepSeek API request failed: {exc.reason}") from exc
        except (http.client.HTTPException, TimeoutError, OSError) as exc:
            raise DeepSeekError(f"DeepSeek API connection failed: {exc}") from exc

        try:
            decoded: dict[str, Any] = json.loads(response_body)
            return decoded["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise DeepSeekError("DeepSeek API response did not match chat-completions format") from exc
