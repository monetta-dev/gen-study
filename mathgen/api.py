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
    ComplianceValidatorAgent,
    CriticalValidatorAgent,
    GeneratorAgent,
    InteractionTracer,
    MathGenerationService,
    RepairAgent,
)
from .composite_renderer import (
    CompositeLayoutAgent,
    CompositeRenderError,
    CompositeRenderService,
)
from .env import load_dotenv
from .llm import LLMError, OpenCodeGoClient
from .models import (
    ContractError,
    GenerateRequest,
    ModelOutputError,
    VisualGenerateRequest,
)
from .problem_renderer import ProblemRenderer, ProblemRenderError
from .visual_agents import (
    TikZGenerationAgent,
    TikZRepairAgent,
    TikZValidatorAgent,
    VisualGenerationService,
    VisualInstructionAgent,
)
from .tikz_renderer import TikZCompileAgent, TikZRenderer

VIEWER_ROOT = Path(__file__).resolve().parents[1] / "viewer"


def build_service(capture_trace: bool = False) -> MathGenerationService:
    load_dotenv()
    tracer = InteractionTracer(
        enabled=_env_flag("GENERATION_TRACE"),
        capture=capture_trace,
    )
    generator_client = _opencode_client_for_role("GENERATOR")
    repair_client = _opencode_client_for_role("REPAIR")
    critical_validator_client = _opencode_client_for_role("CRITICAL_VALIDATOR")
    compliance_validator_client = _opencode_client_for_role("COMPLIANCE_VALIDATOR")
    return MathGenerationService(
        generator=GeneratorAgent(generator_client, tracer=tracer),
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


def build_visual_service(capture_trace: bool = False) -> VisualGenerationService:
    load_dotenv()
    tracer = InteractionTracer(
        enabled=_env_flag("GENERATION_TRACE"),
        capture=capture_trace,
    )
    instruction_client = _opencode_client_for_role("VISUAL_INSTRUCTION")
    tikz_client = _opencode_client_for_role("VISUAL_TIKZ")
    tikz_validator_client = _opencode_client_for_role("VISUAL_TIKZ_VALIDATOR")
    tikz_repair_client = _opencode_client_for_role("VISUAL_TIKZ_REPAIR")
    renderer = None
    if _env_flag("VISUAL_RENDER_ENABLED", True):
        tikz_compile_client = _opencode_client_for_role("VISUAL_TIKZ_COMPILE")
        renderer = TikZRenderer(
            compile_agent=TikZCompileAgent(tikz_compile_client, tracer=tracer),
            timeout_seconds=_env_float("VISUAL_COMPILE_TIMEOUT_SECONDS", 30.0),
            max_actions=_env_int("VISUAL_COMPILE_MAX_ACTIONS", 5),
            tracer=tracer,
        )
    return VisualGenerationService(
        instruction_agent=VisualInstructionAgent(instruction_client, tracer=tracer),
        tikz_agent=TikZGenerationAgent(tikz_client, tracer=tracer),
        tikz_validator=TikZValidatorAgent(tikz_validator_client, tracer=tracer),
        tikz_repair_agent=TikZRepairAgent(tikz_repair_client, tracer=tracer),
        max_repair_attempts=_env_int("VISUAL_TIKZ_REPAIR_ATTEMPTS", 2),
        renderer=renderer,
        tracer=tracer,
    )


def build_problem_renderer() -> ProblemRenderer:
    load_dotenv()
    return ProblemRenderer(
        timeout_seconds=_env_float("PROBLEM_RENDER_TIMEOUT_SECONDS", 180.0),
        dpi=_env_int("PROBLEM_RENDER_DPI", 180),
    )


def build_composite_service(capture_trace: bool = False) -> CompositeRenderService:
    load_dotenv()
    tracer = InteractionTracer(
        enabled=_env_flag("GENERATION_TRACE"),
        capture=capture_trace,
    )
    composite_client = _opencode_client_for_role("COMPOSITE_LAYOUT")
    return CompositeRenderService(
        layout_agent=CompositeLayoutAgent(composite_client, tracer=tracer),
        tracer=tracer,
    )


def _opencode_client_for_role(role: str) -> OpenCodeGoClient:
    return OpenCodeGoClient(
        model=_env_optional_string(f"{role}_MODEL"),
        max_tokens=_env_optional_int(f"{role}_MAX_TOKENS"),
        thinking_budget_tokens=_env_optional_int(f"{role}_THINKING_BUDGET_TOKENS"),
        thinking_enabled=_env_optional_flag(f"{role}_THINKING_ENABLED"),
        stream=_env_optional_flag(f"{role}_STREAM"),
    )


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


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


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ContractError(f"`{name}` must be a number") from exc
    if parsed <= 0:
        raise ContractError(f"`{name}` must be positive")
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


def _env_optional_flag(name: str) -> bool | None:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return None
    return value in {"1", "true", "yes", "on"}


def _env_string(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_optional_string(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _payload_trace_enabled(payload: dict[str, Any]) -> bool:
    value = payload.get("trace", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _payload_render_enabled(payload: dict[str, Any]) -> bool:
    for key in ("render_output", "render_png", "include_rendered_output"):
        value = payload.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


class MathGenerationHandler(BaseHTTPRequestHandler):
    service: MathGenerationService | None = None
    visual_service: VisualGenerationService | None = None
    problem_renderer: ProblemRenderer | None = None
    composite_service: CompositeRenderService | None = None
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
        if self.path == "/generate":
            self._handle_generate(request_id)
            return
        if self.path == "/visuals/generate":
            self._handle_visual_generate(request_id)
            return
        if self.path == "/render/composite":
            self._handle_composite_render(request_id)
            return
        if self.path == "/dev/generate-full":
            self._handle_dev_generate_full(request_id)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"status": "error", "error": "not_found"})

    def _handle_generate(self, request_id: str) -> None:
        if not self._is_authorized():
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"status": "error", "error": "unauthorized"},
            )
            return

        try:
            payload = self._read_json_body()
            include_trace = _payload_trace_enabled(payload)
            render_output = _payload_render_enabled(payload)
            request = GenerateRequest.from_dict(payload)
            service = self.service or build_service(capture_trace=include_trace)
            result = service.generate_verified(request)
            if render_output and result.get("status") == "success":
                renderer = self.problem_renderer or build_problem_renderer()
                result["rendered_output"] = renderer.render(
                    str(result.get("problem", "")),
                    str(result.get("solution", "")),
                ).to_dict()
            self._attach_trace(result, service, include_trace)
        except ProblemRenderError as exc:
            self._send_render_error(request_id, exc)
            return
        except ContractError as exc:
            self._send_contract_error(request_id, exc)
            return
        except LLMError as exc:
            self._send_llm_error(request_id, exc)
            return
        except ModelOutputError as exc:
            self._send_model_output_error(request_id, exc)
            return
        except json.JSONDecodeError:
            self._send_json_error(request_id)
            return
        except Exception as exc:
            self._send_internal_error(request_id, exc)
            return

        status = HTTPStatus.OK if result.get("status") == "success" else HTTPStatus.UNPROCESSABLE_ENTITY
        self._send_json(status, result)

    def _handle_visual_generate(self, request_id: str) -> None:
        if not self._is_authorized():
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"status": "error", "error": "unauthorized"},
            )
            return

        try:
            payload = self._read_json_body()
            include_trace = _payload_trace_enabled(payload)
            request = VisualGenerateRequest.from_dict(payload)
            service = self.visual_service or build_visual_service(
                capture_trace=include_trace
            )
            result = service.generate_visual(request)
            self._attach_trace(result, service, include_trace)
        except ContractError as exc:
            self._send_contract_error(request_id, exc)
            return
        except LLMError as exc:
            self._send_llm_error(request_id, exc)
            return
        except ModelOutputError as exc:
            self._send_model_output_error(request_id, exc)
            return
        except json.JSONDecodeError:
            self._send_json_error(request_id)
            return
        except Exception as exc:
            self._send_internal_error(request_id, exc)
            return

        status = HTTPStatus.OK if result.get("status") == "success" else HTTPStatus.UNPROCESSABLE_ENTITY
        self._send_json(status, result)

    def _handle_composite_render(self, request_id: str) -> None:
        if not _env_flag("COMPOSITE_RENDER_ENABLED"):
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"status": "error", "error": "not_found", "request_id": request_id},
            )
            return
        if not self._is_authorized():
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"status": "error", "error": "unauthorized"},
            )
            return

        try:
            payload = self._read_json_body()
            include_trace = _payload_trace_enabled(payload)
            service = self.composite_service or build_composite_service(
                capture_trace=include_trace
            )
            result = service.render_composite(payload)
            self._attach_trace(result, service, include_trace)
        except CompositeRenderError as exc:
            self._send_composite_render_error(request_id, exc)
            return
        except ContractError as exc:
            self._send_contract_error(request_id, exc)
            return
        except LLMError as exc:
            self._send_llm_error(request_id, exc)
            return
        except ModelOutputError as exc:
            self._send_model_output_error(request_id, exc)
            return
        except json.JSONDecodeError:
            self._send_json_error(request_id)
            return
        except Exception as exc:
            self._send_internal_error(request_id, exc)
            return

        status = HTTPStatus.OK if result.get("status") == "success" else HTTPStatus.UNPROCESSABLE_ENTITY
        self._send_json(status, result)

    def _handle_dev_generate_full(self, request_id: str) -> None:
        if not _env_flag("COMPOSITE_RENDER_ENABLED"):
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"status": "error", "error": "not_found", "request_id": request_id},
            )
            return
        if not self._is_authorized():
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"status": "error", "error": "unauthorized"},
            )
            return

        try:
            payload = self._read_json_body()
            include_trace = _payload_trace_enabled(payload)
            request = GenerateRequest.from_dict(payload)

            generation_service = self.service or build_service(
                capture_trace=include_trace
            )
            generation = generation_service.generate_verified(request)
            self._attach_trace(generation, generation_service, include_trace)
            if generation.get("status") != "success":
                self._send_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {
                        "status": "failed",
                        "stage": "generate",
                        "generation": generation,
                    },
                )
                return

            renderer = self.problem_renderer or build_problem_renderer()
            generation["rendered_output"] = renderer.render(
                str(generation.get("problem", "")),
                str(generation.get("solution", "")),
            ).to_dict()

            visual_request = VisualGenerateRequest(
                problem=str(generation.get("problem", "")),
                solution=str(generation.get("solution", "")),
                subject=request.subject,
                level=request.level,
                topic=request.topic,
            )
            visual_service = self.visual_service or build_visual_service(
                capture_trace=include_trace
            )
            visuals = visual_service.generate_visual(visual_request)
            self._attach_trace(visuals, visual_service, include_trace)
            if visuals.get("status") != "success":
                self._send_json(
                    HTTPStatus.UNPROCESSABLE_ENTITY,
                    {
                        "status": "failed",
                        "stage": "visuals",
                        "generation": generation,
                        "visuals": visuals,
                    },
                )
                return

            composite_service = self.composite_service or build_composite_service(
                capture_trace=include_trace
            )
            composite = composite_service.render_composite(
                {
                    "problem": generation.get("problem", ""),
                    "solution": generation.get("solution", ""),
                    "rendered_output": generation.get("rendered_output"),
                    "visual_assets": visuals.get("visual_assets", []),
                }
            )
            self._attach_trace(composite, composite_service, include_trace)
            result = {
                "status": "success",
                "stage": "complete",
                "generation": generation,
                "visuals": visuals,
                "composite": composite,
            }
        except ProblemRenderError as exc:
            self._send_render_error(request_id, exc)
            return
        except CompositeRenderError as exc:
            self._send_composite_render_error(request_id, exc)
            return
        except ContractError as exc:
            self._send_contract_error(request_id, exc)
            return
        except LLMError as exc:
            self._send_llm_error(request_id, exc)
            return
        except ModelOutputError as exc:
            self._send_model_output_error(request_id, exc)
            return
        except json.JSONDecodeError:
            self._send_json_error(request_id)
            return
        except Exception as exc:
            self._send_internal_error(request_id, exc)
            return

        self._send_json(HTTPStatus.OK, result)

    def _send_contract_error(self, request_id: str, exc: ContractError) -> None:
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

    def _send_llm_error(self, request_id: str, exc: LLMError) -> None:
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

    def _send_model_output_error(
        self, request_id: str, exc: ModelOutputError
    ) -> None:
        self._log_error(request_id, "llm_output_error", exc)
        self._send_json(
            HTTPStatus.BAD_GATEWAY,
            {
                "status": "error",
                "error": "llm_output_error",
                "detail": str(exc),
                "request_id": request_id,
            },
        )

    def _send_render_error(self, request_id: str, exc: ProblemRenderError) -> None:
        self._log_error(request_id, "render_error", exc)
        detail = str(exc) if self.debug_tracebacks else "Problem rendering failed"
        self._send_json(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {
                "status": "error",
                "error": "render_error",
                "detail": detail,
                "request_id": request_id,
            },
        )

    def _send_composite_render_error(
        self, request_id: str, exc: CompositeRenderError
    ) -> None:
        self._log_error(request_id, "composite_render_error", exc)
        detail = str(exc) if self.debug_tracebacks else "Composite rendering failed"
        self._send_json(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            {
                "status": "error",
                "error": "composite_render_error",
                "detail": detail,
                "request_id": request_id,
            },
        )

    def _send_json_error(self, request_id: str) -> None:
        self._log_error(request_id, "invalid_json")
        self._send_json(
            HTTPStatus.BAD_REQUEST,
            {"status": "error", "error": "invalid_json", "request_id": request_id},
        )

    def _send_internal_error(self, request_id: str, exc: Exception) -> None:
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

    def _attach_trace(
        self,
        result: dict[str, Any],
        service: MathGenerationService | VisualGenerationService,
        include_trace: bool,
    ) -> None:
        if include_trace and hasattr(service, "tracer"):
            result["trace"] = service.tracer.to_response()

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
        if path in {"/config.js", "/viewer/config.js"}:
            self._send_viewer_config()
            return
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

    def _send_viewer_config(self) -> None:
        load_dotenv()
        config = {
            "apiUrl": _env_string("GUI_API_URL", ""),
            "visualApiUrl": _env_string("GUI_VISUAL_API_URL", ""),
            "enableRenderOutput": _env_flag("PROBLEM_RENDER_OUTPUT_ENABLED"),
            "enableCompositeRender": _env_flag("COMPOSITE_RENDER_ENABLED"),
            "includeTrace": _env_flag("GUI_INCLUDE_TRACE"),
            "showProcess": _env_flag("GUI_SHOW_PROCESS"),
            "showFullRawResponse": _env_flag("GUI_SHOW_FULL_RAW_RESPONSE"),
        }
        body = (
            "window.GEN_STUDY_CONFIG = "
            f"{json.dumps(config, ensure_ascii=False, indent=2)};\n"
        ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/javascript; charset=utf-8")
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
    visual_service: VisualGenerationService | None = None,
    problem_renderer: ProblemRenderer | None = None,
    composite_service: CompositeRenderService | None = None,
    debug_tracebacks: bool = False,
    api_key: str = "",
    max_body_bytes: int = 65536,
    allowed_origin: str = "*",
) -> ThreadingHTTPServer:
    class Handler(MathGenerationHandler):
        pass

    Handler.service = service
    Handler.visual_service = visual_service
    Handler.problem_renderer = problem_renderer
    Handler.composite_service = composite_service
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
        print("API key authentication is enabled for POST /generate and /visuals/generate")
    print(
        "POST /generate with JSON: "
        "{\"subject\":\"chemistry\",\"level\":\"University of Tokyo\",\"topic\":\"Acid-Base Titration\"}"
    )
    print(
        "POST /visuals/generate with JSON: "
        "{\"subject\":\"physics\",\"problem\":\"...\",\"solution\":\"...\"}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
