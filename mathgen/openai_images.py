"""OpenAI Image API client for generated visual assets."""

from __future__ import annotations

import base64
import json
import os
import time
import uuid
import urllib.error
import urllib.request
from typing import Any


OPENAI_IMAGES_BASE_URL = "https://api.openai.com/v1"


class ImageGenerationError(RuntimeError):
    """Raised when image generation fails."""


def _env_string(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    value = _env_string(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ImageGenerationError(f"{name} must be a number") from exc


def _env_int(name: str, default: int) -> int:
    value = _env_string(name)
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ImageGenerationError(f"{name} must be an integer") from exc
    if parsed < 0:
        raise ImageGenerationError(f"{name} must be non-negative")
    return parsed


class OpenAIImageClient:
    """Small standard-library client for OpenAI image generations."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.api_key = api_key or _env_string("OPENAI_API_KEY")
        self.base_url = (base_url or _env_string("OPENAI_BASE_URL", OPENAI_IMAGES_BASE_URL)).rstrip("/")
        self.model = model or _env_string("OPENAI_IMAGE_MODEL") or "gpt-image-2"
        self.size = _env_string("OPENAI_IMAGE_SIZE")
        self.quality = _env_string("OPENAI_IMAGE_QUALITY")
        self.n = _env_int("OPENAI_IMAGE_N", 0)
        self.timeout_seconds = timeout_seconds or _env_float(
            "OPENAI_IMAGE_TIMEOUT_SECONDS", 300.0
        )
        self.retry_attempts = _env_int("OPENAI_IMAGE_RETRY_ATTEMPTS", 1)
        self.retry_delay_seconds = _env_float("OPENAI_IMAGE_RETRY_DELAY_SECONDS", 60.0)

    def generate_png_base64(self, prompt: str) -> str:
        if not self.api_key:
            raise ImageGenerationError("OPENAI_API_KEY is not configured")
        if not prompt.strip():
            raise ImageGenerationError("image prompt must be non-empty")

        payload: dict[str, str | int] = self._image_fields(prompt)
        if self.n:
            payload["n"] = self.n
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/images/generations",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        response_body = self._open_with_retries(request)

        return self._image_base64_from_response(response_body)

    def edit_png_base64(self, image_base64: str, prompt: str) -> str:
        if not self.api_key:
            raise ImageGenerationError("OPENAI_API_KEY is not configured")
        if not prompt.strip():
            raise ImageGenerationError("image edit prompt must be non-empty")
        try:
            image_bytes = base64.b64decode(image_base64, validate=True)
        except ValueError as exc:
            raise ImageGenerationError("image_base64 must be valid base64") from exc

        body, content_type = self._multipart_body(
            fields=self._image_fields(prompt),
            files=[
                {
                    "field": "image",
                    "filename": "image.png",
                    "content_type": "image/png",
                    "data": image_bytes,
                }
            ],
        )
        request = urllib.request.Request(
            f"{self.base_url}/images/edits",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": content_type,
                "Accept": "application/json",
            },
            method="POST",
        )
        response_body = self._open_with_retries(request)
        return self._image_base64_from_response(response_body)

    def _image_fields(self, prompt: str) -> dict[str, str]:
        fields = {
            "model": self.model,
            "prompt": prompt,
        }
        if self.size:
            fields["size"] = self.size
        if self.quality:
            fields["quality"] = self.quality
        if self.n:
            fields["n"] = str(self.n)
        return fields

    def _open_with_retries(self, request: urllib.request.Request) -> str:
        for attempt in range(self.retry_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return response.read().decode("utf-8")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                exc.close()
                if not self._is_retryable_http_error(exc.code, body) or attempt >= self.retry_attempts:
                    raise ImageGenerationError(
                        f"OpenAI Image API returned HTTP {exc.code}: {body}"
                    ) from exc
                delay = self._retry_delay(body)
                print(
                    f"Retrying OpenAI Image API request after HTTP {exc.code} "
                    f"in {delay:g}s (attempt {attempt + 1}/{self.retry_attempts})",
                    flush=True,
                )
                time.sleep(delay)
            except urllib.error.URLError as exc:
                raise ImageGenerationError(f"OpenAI Image API request failed: {exc.reason}") from exc
            except (TimeoutError, OSError) as exc:
                raise ImageGenerationError(f"OpenAI Image API connection failed: {exc}") from exc
        raise ImageGenerationError("OpenAI Image API request failed")

    def _image_base64_from_response(self, response_body: str) -> str:
        try:
            decoded: dict[str, Any] = json.loads(response_body)
            image_base64 = decoded["data"][0]["b64_json"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ImageGenerationError(
                "OpenAI Image API response did not include data[0].b64_json"
            ) from exc
        if not isinstance(image_base64, str) or not image_base64.strip():
            raise ImageGenerationError("OpenAI Image API returned an empty image")
        return image_base64.strip()

    def _multipart_body(
        self, fields: dict[str, str], files: list[dict[str, object]]
    ) -> tuple[bytes, str]:
        boundary = f"gen-study-{uuid.uuid4().hex}"
        chunks: list[bytes] = []
        for name, value in fields.items():
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode("utf-8"),
                    f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"),
                    value.encode("utf-8"),
                    b"\r\n",
                ]
            )
        for file in files:
            field = str(file["field"])
            filename = str(file["filename"])
            content_type = str(file["content_type"])
            data = file["data"]
            if not isinstance(data, bytes):
                raise ImageGenerationError("multipart file data must be bytes")
            chunks.extend(
                [
                    f"--{boundary}\r\n".encode("utf-8"),
                    (
                        f'Content-Disposition: form-data; name="{field}"; '
                        f'filename="{filename}"\r\n'
                    ).encode("utf-8"),
                    f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"),
                    data,
                    b"\r\n",
                ]
            )
        chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
        return b"".join(chunks), f"multipart/form-data; boundary={boundary}"

    def _is_retryable_http_error(self, code: int, body: str) -> bool:
        if code in {408, 409, 425, 429, 500, 502, 503, 504, 520, 524}:
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
