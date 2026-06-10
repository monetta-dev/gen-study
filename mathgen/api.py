"""HTTP API for exam question generation and verification."""

from __future__ import annotations

import hmac
import json
import mimetypes
import os
import sys
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .agents import (
    AdjudicatorAgent,
    ComplianceValidatorAgent,
    CriticalValidatorAgent,
    GeneratorAgent,
    InteractionTracer,
    MathGenerationService,
    RepairAgent,
)
from .env import load_dotenv
from .llm import LLMError, OpenCodeGoClient
from .models import ContractError, GenerateRequest

VIEWER_ROOT = Path(__file__).resolve().parents[1] / "viewer"


def build_service() -> MathGenerationService:
    load_dotenv()
    tracer = InteractionTracer(enabled=_env_flag("GENERATION_TRACE"))
    generator_client = OpenCodeGoClient()
    adjudicator_client = OpenCodeGoClient()
    repair_client = OpenCodeGoClient()
    critical_validator_client = OpenCodeGoClient()
    compliance_validator_client = OpenCodeGoClient()
    return MathGenerationService(
        generator=GeneratorAgent(generator_client, tracer=tracer),
        adjudicator=AdjudicatorAgent(adjudicator_client, tracer=tracer),
        repair_agent=RepairAgent(repair_client, tracer=tracer),
        critical_validator=CriticalValidatorAgent(
            critical_validator_client, tracer=tracer
        ),
        compliance_validator=ComplianceValidatorAgent(
            compliance_validator_client, tracer=tracer
        ),
        max_repair_attempts=_env_optional_int("MAX_REPAIR_ATTEMPTS"),
        tracer=tracer,
    )


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ContractError(f"`{name}` must be an integer") from exc
    if parsed < 0:
        raise ContractError(f"`{name}` must be non-negative")
    return parsed


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name, "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ContractError(f"`{name}` must be an integer") from exc
    if parsed < 0:
        raise ContractError(f"`{name}` must be non-negative")
    return parsed


def _env_string(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


class MathGenerationHandler(BaseHTTPRequestHandler):
    service: MathGenerationService | None = None
    debug_tracebacks = False
    api_key = ""
    max_body_bytes = 65536
    allowed_origin = "*"

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if (
            self.path == "/"
            or self.path == "/viewer"
            or self.path.startswith("/viewer/")
            or self.path in {"/app.js", "/config.js", "/styles.css"}
        ):
            self._send_static()
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "not_found"})

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._send_cors_headers()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, X-API-Key, Authorization",
        )
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_POST(self) -> None:
        request_id = uuid.uuid4().hex[:12]
        if self.path != "/generate":
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "not_found"})
            return
        if not self._is_authorized():
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"status": "error", "error": "unauthorized"},
            )
            return

        try:
            payload = self._read_json_body()
            request = GenerateRequest.from_dict(payload)
            service = self.service or build_service()
            result = service.generate_verified(request)
        except ContractError as exc:
            self._log_error(request_id, "invalid_request", exc)
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {
                    "status": "error",
                    "error": "invalid_request",
                    "detail": str(exc),
                    "request_id": request_id,
                },
            )
            return
        except LLMError as exc:
            self._log_error(request_id, "llm_api_error", exc)
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {
                    "status": "error",
                    "error": "llm_api_error",
                    "detail": str(exc),
                    "request_id": request_id,
                },
            )
            return
        except json.JSONDecodeError:
            self._log_error(request_id, "invalid_json")
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error": "invalid_json", "request_id": request_id},
            )
            return
        except Exception as exc:
            self._log_error(request_id, "internal_server_error", exc)
            detail = str(exc) if self.debug_tracebacks else "Unexpected server error"
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "status": "error",
                    "error": "internal_server_error",
                    "detail": detail,
                    "request_id": request_id,
                },
            )
            return

        status = HTTPStatus.OK if result.get("status") == "success" else HTTPStatus.UNPROCESSABLE_ENTITY
        self._send_json(status, result)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _log_error(
        self,
        request_id: str,
        error: str,
        exc: BaseException | None = None,
    ) -> None:
        message = f"[request_id={request_id}] {error}"
        if exc is not None:
            message = f"{message}: {exc}"
        print(message, file=sys.stderr, flush=True)
        if exc is not None and self.debug_tracebacks:
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length > self.max_body_bytes:
            raise ContractError(
                f"Request body must be at most {self.max_body_bytes} bytes"
            )
        raw_body = self.rfile.read(content_length).decode("utf-8")
        payload = json.loads(raw_body)
        if not isinstance(payload, dict):
            raise ContractError("Request body must be a JSON object")
        return payload

    def _is_authorized(self) -> bool:
        if not self.api_key:
            return True
        provided_key = self.headers.get("X-API-Key", "").strip()
        if hmac.compare_digest(provided_key, self.api_key):
            return True
        auth_header = self.headers.get("Authorization", "").strip()
        prefix = "Bearer "
        return auth_header.startswith(prefix) and hmac.compare_digest(
            auth_header[len(prefix) :],
            self.api_key,
        )

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            if self.debug_tracebacks:
                self.log_message("client disconnected before response could be sent")

    def _send_static(self) -> None:
        path = self.path.split("?", 1)[0]
        relative_path = "index.html"
        if path.startswith("/viewer/"):
            relative_path = path[len("/viewer/") :] or "index.html"
        elif path == "/viewer":
            relative_path = "index.html"
        elif path in {"/app.js", "/config.js", "/styles.css"}:
            relative_path = path.lstrip("/")

        target = (VIEWER_ROOT / relative_path).resolve()
        if VIEWER_ROOT not in target.parents and target != VIEWER_ROOT:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"status": "error", "error": "not_found"},
            )
            return
        if not target.is_file():
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"status": "error", "error": "not_found"},
            )
            return

        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_cors_headers(self) -> None:
        if self.allowed_origin:
            self.send_header("Access-Control-Allow-Origin", self.allowed_origin)
            self.send_header("Vary", "Origin")


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    service: MathGenerationService | None = None,
    debug_tracebacks: bool = False,
    api_key: str = "",
    max_body_bytes: int = 65536,
    allowed_origin: str = "*",
) -> ThreadingHTTPServer:
    class Handler(MathGenerationHandler):
        pass

    Handler.service = service
    Handler.debug_tracebacks = debug_tracebacks
    Handler.api_key = api_key
    Handler.max_body_bytes = max_body_bytes
    Handler.allowed_origin = allowed_origin
    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    load_dotenv()
    default_host = "0.0.0.0" if _env_string("PORT") else "127.0.0.1"
    host = _env_string("API_HOST", default_host)
    port = _env_int("API_PORT", _env_int("PORT", 8000))
    debug_tracebacks = _env_flag("API_DEBUG")
    api_key = _env_string("GEN_STUDY_API_KEY")
    max_body_bytes = _env_int("API_MAX_BODY_BYTES", 65536)
    allowed_origin = _env_string("API_ALLOWED_ORIGIN", "*")
    server = create_server(
        host=host,
        port=port,
        debug_tracebacks=debug_tracebacks,
        api_key=api_key,
        max_body_bytes=max_body_bytes,
        allowed_origin=allowed_origin,
    )
    print(f"Serving exam question generation API on http://{host}:{port}")
    if api_key:
        print("API key authentication is enabled for POST /generate")
    print(
        "POST /generate with JSON: "
        "{\"subject\":\"chemistry\",\"level\":\"University of Tokyo\",\"topic\":\"Acid-Base Titration\"}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
