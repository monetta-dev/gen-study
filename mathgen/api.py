"""HTTP API for exam question generation and verification."""

from __future__ import annotations

import json
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .agents import (
    ComplianceValidatorAgent,
    CriticalValidatorAgent,
    GeneratorAgent,
    InteractionTracer,
    MathGenerationService,
)
from .deepseek import DeepSeekClient, DeepSeekError
from .env import load_dotenv
from .models import ContractError, GenerateRequest


def build_service() -> MathGenerationService:
    load_dotenv()
    tracer = InteractionTracer(enabled=_env_flag("GENERATION_TRACE"))
    generator_client = DeepSeekClient()
    critical_validator_client = DeepSeekClient()
    compliance_validator_client = DeepSeekClient()
    return MathGenerationService(
        generator=GeneratorAgent(generator_client, tracer=tracer),
        critical_validator=CriticalValidatorAgent(
            critical_validator_client, tracer=tracer
        ),
        compliance_validator=ComplianceValidatorAgent(
            compliance_validator_client, tracer=tracer
        ),
        tracer=tracer,
    )


def _env_flag(name: str) -> bool:
    import os

    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


class MathGenerationHandler(BaseHTTPRequestHandler):
    service: MathGenerationService | None = None
    debug_tracebacks = True

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/generate":
            self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "not_found"})
            return

        try:
            payload = self._read_json_body()
            request = GenerateRequest.from_dict(payload)
            service = self.service or build_service()
            result = service.generate_verified(request)
        except ContractError as exc:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error": "invalid_request", "detail": str(exc)},
            )
            return
        except DeepSeekError as exc:
            self._send_json(
                HTTPStatus.BAD_GATEWAY,
                {"status": "error", "error": "deepseek_api_error", "detail": str(exc)},
            )
            return
        except json.JSONDecodeError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"status": "error", "error": "invalid_json"},
            )
            return
        except Exception as exc:
            if self.debug_tracebacks:
                traceback.print_exc()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "status": "error",
                    "error": "internal_server_error",
                    "detail": str(exc),
                },
            )
            return

        status = HTTPStatus.OK if result.get("status") == "success" else HTTPStatus.UNPROCESSABLE_ENTITY
        self._send_json(status, result)

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _read_json_body(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length).decode("utf-8")
        payload = json.loads(raw_body)
        if not isinstance(payload, dict):
            raise ContractError("Request body must be a JSON object")
        return payload

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            if self.debug_tracebacks:
                self.log_message("client disconnected before response could be sent")


def create_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    service: MathGenerationService | None = None,
    debug_tracebacks: bool = True,
) -> ThreadingHTTPServer:
    class Handler(MathGenerationHandler):
        pass

    Handler.service = service
    Handler.debug_tracebacks = debug_tracebacks
    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    load_dotenv()
    server = create_server()
    print("Serving exam question generation API on http://127.0.0.1:8000")
    print(
        "POST /generate with JSON: "
        "{\"subject\":\"chemistry\",\"level\":\"University of Tokyo\",\"topic\":\"Acid-Base Titration\"}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
