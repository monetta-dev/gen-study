import json
import os
import threading
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from mathgen.agents import (
    ComplianceValidatorAgent,
    CriticalValidatorAgent,
    GeneratorAgent,
    InteractionTracer,
    MathGenerationService,
    RepairAgent,
)
from mathgen.api import MathGenerationHandler, create_server
from mathgen.llm import GPT_61_SOL_MODEL
import mathgen.api as api_module
from mathgen.models import ModelOutputError


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, messages):
        if isinstance(self.responses[0], Exception):
            raise self.responses.pop(0)
        return self.responses.pop(0)


class BrokenPipeWriter:
    def write(self, body):
        raise BrokenPipeError("client closed")


class FakeHandlerForSendJson:
    debug_tracebacks = False

    def __init__(self):
        self.wfile = BrokenPipeWriter()
        self.status = None
        self.headers = []
        self.ended = False

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers.append((key, value))

    def end_headers(self):
        self.ended = True

    def _send_cors_headers(self):
        return

    def log_message(self, format, *args):
        raise AssertionError("log_message should not be called when debug_tracebacks is false")


class FakeProblemRenderer:
    def __init__(self):
        self.calls = []

    def render(self, problem, solution):
        self.calls.append((problem, solution))
        return type(
            "FakeRenderedOutput",
            (),
            {
                "to_dict": lambda self: {
                    "status": "compiled",
                    "format": "png",
                    "assets": [
                        {
                            "kind": "problem",
                            "page": 1,
                            "title": "問題",
                            "mime_type": "image/png",
                            "png_base64": "cHJvYmxlbQ==",
                        }
                    ],
                }
            },
        )()


class FakeCompositeService:
    tracer = type("FakeTracer", (), {"to_response": lambda self: ["composite trace"]})()

    def __init__(self):
        self.calls = []

    def render_composite(self, payload):
        self.calls.append(payload)
        return {
            "status": "success",
            "composite_assets": [
                {
                    "kind": "problem",
                    "mime_type": "image/png",
                    "png_base64": "cHJvYmxlbQ==",
                },
                {
                    "kind": "solution",
                    "mime_type": "image/png",
                    "png_base64": "c29sdXRpb24=",
                },
            ],
            "layout_decision": {"status": "test", "placements": [], "notes": ""},
        }


class FakeFullPipelineProblemRenderer:
    def render(self, problem, solution):
        return type(
            "FakeRenderedOutput",
            (),
            {
                "to_dict": lambda self: {
                    "status": "compiled",
                    "format": "png",
                    "assets": [
                        {
                            "kind": "problem",
                            "page": 1,
                            "title": "問題",
                            "mime_type": "image/png",
                            "png_base64": "cHJvYmxlbQ==",
                        },
                        {
                            "kind": "solution",
                            "page": 1,
                            "title": "解答",
                            "mime_type": "image/png",
                            "png_base64": "c29sdXRpb24=",
                        },
                    ],
                }
            },
        )()


class FakeFullPipelineVisualService:
    tracer = type("FakeTracer", (), {"to_response": lambda self: []})()

    def generate_visual(self, request):
        return {
            "status": "success",
            "visual_needed": True,
            "visual_assets": [
                {
                    "id": "visual_1",
                    "placement": "problem",
                    "purpose": "diagram",
                    "mime_type": "image/png",
                    "png_base64": "dmlzdWFs",
                }
            ],
        }


class ApiTests(unittest.TestCase):
    def start_server(self, service, **kwargs):
        debug_tracebacks = kwargs.pop("debug_tracebacks", False)
        server = create_server(
            host="127.0.0.1",
            port=0,
            service=service,
            debug_tracebacks=debug_tracebacks,
            **kwargs,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def post_json(self, server, payload, headers=None, path="/generate"):
        body = json.dumps(payload).encode("utf-8")
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}{path}",
            data=body,
            headers=request_headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def get(self, server, path):
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}{path}",
            method="GET",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.headers, response.read().decode("utf-8")

    def options(self, server, path):
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}{path}",
            method="OPTIONS",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.headers

    def test_generate_endpoint_returns_verified_problem(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient(['{"problem":"問 $x$","solution":"解 $x=1$"}'])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(
                FakeClient(['{"passed":true,"solution_accurate":true,"issues":[]}'])
            ),
            ComplianceValidatorAgent(
                FakeClient(['{"passed":true,"curriculum_compliant":true,"issues":[]}'])
            ),
        )
        server = self.start_server(service)

        status, payload = self.post_json(
            server,
            {"level": "University of Tokyo", "topic": "Probability Recurrence Relations"},
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["problem"], "問 $x$")

    def test_generate_endpoint_can_return_rendered_png_output(self):
        renderer = FakeProblemRenderer()
        service = MathGenerationService(
            GeneratorAgent(FakeClient(['{"problem":"問 $x$","solution":"解 $x=1$"}'])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(
                FakeClient(['{"passed":true,"solution_accurate":true,"issues":[]}'])
            ),
            ComplianceValidatorAgent(
                FakeClient(['{"passed":true,"curriculum_compliant":true,"issues":[]}'])
            ),
        )
        server = self.start_server(service, problem_renderer=renderer)

        status, payload = self.post_json(
            server,
            {
                "level": "University of Tokyo",
                "topic": "Probability Recurrence Relations",
                "render_output": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["rendered_output"]["status"], "compiled")
        self.assertEqual(payload["rendered_output"]["assets"][0]["mime_type"], "image/png")
        self.assertEqual(renderer.calls, [("問 $x$", "解 $x=1$")])

    def test_generate_endpoint_can_return_process_trace(self):
        tracer = InteractionTracer(capture=True)
        service = MathGenerationService(
            GeneratorAgent(FakeClient(['{"problem":"問","solution":"解"}']), tracer=tracer),
            RepairAgent(FakeClient([]), tracer=tracer),
            CriticalValidatorAgent(
                FakeClient(['{"passed":true,"solution_accurate":true,"issues":[]}']),
                tracer=tracer,
            ),
            ComplianceValidatorAgent(
                FakeClient(['{"passed":true,"curriculum_compliant":true,"issues":[]}']),
                tracer=tracer,
            ),
            tracer=tracer,
        )
        server = self.start_server(service)

        status, payload = self.post_json(
            server,
            {
                "level": "University of Tokyo",
                "topic": "Probability Recurrence Relations",
                "trace": True,
            },
        )

        self.assertEqual(status, 200)
        self.assertIn("trace", payload)
        self.assertTrue(any("Generator request" in record for record in payload["trace"]))

    def test_all_services_default_to_sol_with_role_reasoning_override(self):
        with patch.dict(os.environ, {"GENERATOR_REASONING_EFFORT": "medium"}, clear=True):
            with patch.object(api_module, "load_dotenv"):
                service = api_module.build_service()
                visuals = api_module.build_visual_service()
                composite = api_module.build_composite_service()
        agents = [
            service.generator, service.repair_agent, service.critical_validator,
            service.compliance_validator, visuals.instruction_agent,
            visuals.tikz_agent, visuals.tikz_validator, visuals.tikz_repair_agent,
            visuals.renderer.compile_agent, composite.layout_agent,
        ]
        self.assertEqual([agent.client.model for agent in agents], [GPT_61_SOL_MODEL] * 10)
        self.assertEqual(service.generator.client.reasoning_effort, "medium")
        self.assertEqual(service.repair_agent.client.reasoning_effort, "high")

    def test_build_service_uses_role_specific_client_settings(self):
        previous = {
            key: os.environ.get(key)
            for key in (
                "OPENROUTER_API_KEY",
                "GENERATOR_MODEL",
                "REPAIR_MODEL",
                "CRITICAL_VALIDATOR_MODEL",
                "COMPLIANCE_VALIDATOR_MODEL",
                "REPAIR_THINKING_BUDGET_TOKENS",
                "CRITICAL_VALIDATOR_THINKING_BUDGET_TOKENS",
                "COMPLIANCE_VALIDATOR_THINKING_BUDGET_TOKENS",
                "COMPLIANCE_VALIDATOR_THINKING_ENABLED",
            )
        }
        previous_cwd = os.getcwd()
        calls = []

        class FakeOpenRouterClient:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def chat(self, messages):
                raise AssertionError("chat should not be called")

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                os.environ["OPENROUTER_API_KEY"] = "test-key"
                os.environ["GENERATOR_MODEL"] = "qwen3.7-plus"
                os.environ.pop("REPAIR_MODEL", None)
                os.environ.pop("CRITICAL_VALIDATOR_MODEL", None)
                os.environ.pop("COMPLIANCE_VALIDATOR_MODEL", None)
                os.environ["REPAIR_THINKING_BUDGET_TOKENS"] = "2048"
                os.environ["CRITICAL_VALIDATOR_THINKING_BUDGET_TOKENS"] = "8192"
                os.environ["COMPLIANCE_VALIDATOR_THINKING_BUDGET_TOKENS"] = "4096"
                os.environ["COMPLIANCE_VALIDATOR_THINKING_ENABLED"] = "true"
                with patch.object(api_module, "OpenRouterClient", FakeOpenRouterClient):
                    api_module.build_service()
            finally:
                os.chdir(previous_cwd)
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertIn(
            {
                "model": "qwen3.7-plus",
                "reasoning_effort": None,
                "max_tokens": None,
                "thinking_budget_tokens": None,
                "thinking_enabled": None,
                "stream": None,
            },
            calls,
        )
        self.assertIn(
            {
                "model": None,
                "reasoning_effort": None,
                "max_tokens": None,
                "thinking_budget_tokens": 2048,
                "thinking_enabled": None,
                "stream": None,
            },
            calls,
        )
        self.assertIn(
            {
                "model": None,
                "reasoning_effort": None,
                "max_tokens": None,
                "thinking_budget_tokens": 8192,
                "thinking_enabled": None,
                "stream": None,
            },
            calls,
        )
        self.assertIn(
            {
                "model": None,
                "reasoning_effort": None,
                "max_tokens": None,
                "thinking_budget_tokens": 4096,
                "thinking_enabled": True,
                "stream": None,
            },
            calls,
        )

    def test_build_visual_service_uses_role_specific_thinking_budgets(self):
        previous = {
            key: os.environ.get(key)
            for key in (
                "OPENROUTER_API_KEY",
                "VISUAL_INSTRUCTION_THINKING_BUDGET_TOKENS",
                "VISUAL_TIKZ_THINKING_BUDGET_TOKENS",
                "VISUAL_TIKZ_COMPILE_THINKING_BUDGET_TOKENS",
                "VISUAL_TIKZ_VALIDATOR_THINKING_BUDGET_TOKENS",
                "VISUAL_TIKZ_REPAIR_THINKING_BUDGET_TOKENS",
                "VISUAL_RENDER_ENABLED",
            )
        }
        previous_cwd = os.getcwd()
        calls = []

        class FakeOpenRouterClient:
            def __init__(self, **kwargs):
                calls.append(kwargs)

            def chat(self, messages):
                raise AssertionError("chat should not be called")

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                os.chdir(tmpdir)
                os.environ["OPENROUTER_API_KEY"] = "test-key"
                os.environ["VISUAL_INSTRUCTION_THINKING_BUDGET_TOKENS"] = "1024"
                os.environ["VISUAL_TIKZ_THINKING_BUDGET_TOKENS"] = "8192"
                os.environ["VISUAL_TIKZ_COMPILE_THINKING_BUDGET_TOKENS"] = "3072"
                os.environ["VISUAL_TIKZ_VALIDATOR_THINKING_BUDGET_TOKENS"] = "2048"
                os.environ["VISUAL_TIKZ_REPAIR_THINKING_BUDGET_TOKENS"] = "4096"
                os.environ["VISUAL_RENDER_ENABLED"] = "true"
                with patch.object(api_module, "OpenRouterClient", FakeOpenRouterClient):
                    api_module.build_visual_service()
            finally:
                os.chdir(previous_cwd)
                for key, value in previous.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(
            [call["thinking_budget_tokens"] for call in calls],
            [1024, 8192, 2048, 4096, 3072],
        )

    def test_generate_endpoint_rejects_invalid_request(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(FakeClient([])),
            ComplianceValidatorAgent(FakeClient([])),
        )
        server = self.start_server(service)

        with self.assertRaises(urllib.error.HTTPError) as context:
            self.post_json(server, {"level": "University of Tokyo"})

        error = context.exception
        self.assertEqual(error.code, 400)
        payload = json.loads(error.read().decode("utf-8"))
        error.close()
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"], "invalid_request")

    def test_generate_endpoint_returns_json_for_unexpected_errors(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient([RuntimeError("provider disconnected")])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(FakeClient([])),
            ComplianceValidatorAgent(FakeClient([])),
        )
        server = self.start_server(service)

        with self.assertRaises(urllib.error.HTTPError) as context:
            self.post_json(
                server,
                {"level": "University of Tokyo", "topic": "Probability Recurrence Relations"},
            )

        error = context.exception
        self.assertEqual(error.code, 500)
        payload = json.loads(error.read().decode("utf-8"))
        error.close()
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"], "internal_server_error")
        self.assertEqual(payload["detail"], "Unexpected server error")
        self.assertRegex(payload["request_id"], r"^[0-9a-f]{12}$")

    def test_generate_endpoint_reports_model_output_errors_as_bad_gateway(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient([ModelOutputError("Model response JSON was invalid")])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(FakeClient([])),
            ComplianceValidatorAgent(FakeClient([])),
        )
        server = self.start_server(service)

        with self.assertRaises(urllib.error.HTTPError) as context:
            self.post_json(
                server,
                {"level": "University of Tokyo", "topic": "Probability Recurrence Relations"},
            )

        error = context.exception
        self.assertEqual(error.code, 502)
        payload = json.loads(error.read().decode("utf-8"))
        error.close()
        self.assertEqual(payload["status"], "error")
        self.assertEqual(payload["error"], "llm_output_error")
        self.assertEqual(payload["detail"], "Model response JSON was invalid")

    def test_generate_endpoint_returns_debug_detail_when_enabled(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient([RuntimeError("provider disconnected")])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(FakeClient([])),
            ComplianceValidatorAgent(FakeClient([])),
        )
        server = self.start_server(service, debug_tracebacks=True)

        with self.assertRaises(urllib.error.HTTPError) as context:
            self.post_json(
                server,
                {"level": "University of Tokyo", "topic": "Probability Recurrence Relations"},
            )

        error = context.exception
        self.assertEqual(error.code, 500)
        payload = json.loads(error.read().decode("utf-8"))
        error.close()
        self.assertEqual(payload["error"], "internal_server_error")
        self.assertEqual(payload["detail"], "provider disconnected")
        self.assertRegex(payload["request_id"], r"^[0-9a-f]{12}$")

    def test_generate_endpoint_requires_api_key_when_configured(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient(['{"problem":"問","solution":"解"}'])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(
                FakeClient(['{"passed":true,"solution_accurate":true,"issues":[]}'])
            ),
            ComplianceValidatorAgent(
                FakeClient(['{"passed":true,"curriculum_compliant":true,"issues":[]}'])
            ),
        )
        server = self.start_server(service, api_key="secret")

        with self.assertRaises(urllib.error.HTTPError) as context:
            self.post_json(
                server,
                {"level": "University of Tokyo", "topic": "Probability Recurrence Relations"},
            )

        error = context.exception
        self.assertEqual(error.code, 401)
        payload = json.loads(error.read().decode("utf-8"))
        error.close()
        self.assertEqual(payload["error"], "unauthorized")

        status, payload = self.post_json(
            server,
            {"level": "University of Tokyo", "topic": "Probability Recurrence Relations"},
            headers={"X-API-Key": "secret"},
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "success")

    def test_generate_endpoint_rejects_oversized_request_body(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(FakeClient([])),
            ComplianceValidatorAgent(FakeClient([])),
        )
        server = self.start_server(service, max_body_bytes=10)

        with self.assertRaises(urllib.error.HTTPError) as context:
            self.post_json(
                server,
                {"level": "University of Tokyo", "topic": "Probability Recurrence Relations"},
            )

        error = context.exception
        self.assertEqual(error.code, 400)
        payload = json.loads(error.read().decode("utf-8"))
        error.close()
        self.assertEqual(payload["error"], "invalid_request")
        self.assertIn("at most 10 bytes", payload["detail"])

    def test_root_serves_frontend(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(FakeClient([])),
            ComplianceValidatorAgent(FakeClient([])),
        )
        server = self.start_server(service)

        status, headers, body = self.get(server, "/")

        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn("generateForm", body)

    def test_composite_render_endpoint_is_hidden_when_disabled(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(FakeClient([])),
            ComplianceValidatorAgent(FakeClient([])),
        )
        previous = os.environ.get("COMPOSITE_RENDER_ENABLED")
        os.environ["COMPOSITE_RENDER_ENABLED"] = "false"
        try:
            server = self.start_server(service, composite_service=FakeCompositeService())

            with self.assertRaises(urllib.error.HTTPError) as context:
                self.post_json(
                    server,
                    {"problem": "問", "solution": "解"},
                    path="/render/composite",
                )
            context.exception.close()
        finally:
            if previous is None:
                os.environ.pop("COMPOSITE_RENDER_ENABLED", None)
            else:
                os.environ["COMPOSITE_RENDER_ENABLED"] = previous

        self.assertEqual(context.exception.code, 404)

    def test_composite_render_endpoint_returns_assets_when_enabled(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(FakeClient([])),
            ComplianceValidatorAgent(FakeClient([])),
        )
        composite_service = FakeCompositeService()
        previous = os.environ.get("COMPOSITE_RENDER_ENABLED")
        os.environ["COMPOSITE_RENDER_ENABLED"] = "true"
        try:
            server = self.start_server(
                service,
                composite_service=composite_service,
            )

            status, payload = self.post_json(
                server,
                {"problem": "問", "solution": "解", "trace": True},
                path="/render/composite",
            )
        finally:
            if previous is None:
                os.environ.pop("COMPOSITE_RENDER_ENABLED", None)
            else:
                os.environ["COMPOSITE_RENDER_ENABLED"] = previous

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["composite_assets"][0]["kind"], "problem")
        self.assertEqual(composite_service.calls[0]["problem"], "問")

    def test_dev_generate_full_endpoint_is_hidden_when_disabled(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(FakeClient([])),
            ComplianceValidatorAgent(FakeClient([])),
        )
        previous = os.environ.get("COMPOSITE_RENDER_ENABLED")
        os.environ["COMPOSITE_RENDER_ENABLED"] = "false"
        try:
            server = self.start_server(service)

            with self.assertRaises(urllib.error.HTTPError) as context:
                self.post_json(
                    server,
                    {"level": "University of Tokyo", "topic": "Mechanics"},
                    path="/dev/generate-full",
                )
            context.exception.close()
        finally:
            if previous is None:
                os.environ.pop("COMPOSITE_RENDER_ENABLED", None)
            else:
                os.environ["COMPOSITE_RENDER_ENABLED"] = previous

        self.assertEqual(context.exception.code, 404)

    def test_dev_generate_full_endpoint_runs_existing_services(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient(['{"problem":"問","solution":"解"}'])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(
                FakeClient(['{"passed":true,"solution_accurate":true,"issues":[]}'])
            ),
            ComplianceValidatorAgent(
                FakeClient(['{"passed":true,"curriculum_compliant":true,"issues":[]}'])
            ),
        )
        composite_service = FakeCompositeService()
        previous = os.environ.get("COMPOSITE_RENDER_ENABLED")
        os.environ["COMPOSITE_RENDER_ENABLED"] = "true"
        try:
            server = self.start_server(
                service,
                visual_service=FakeFullPipelineVisualService(),
                problem_renderer=FakeFullPipelineProblemRenderer(),
                composite_service=composite_service,
            )

            status, payload = self.post_json(
                server,
                {
                    "subject": "physics",
                    "level": "University of Tokyo",
                    "topic": "Mechanics",
                },
                path="/dev/generate-full",
            )
        finally:
            if previous is None:
                os.environ.pop("COMPOSITE_RENDER_ENABLED", None)
            else:
                os.environ["COMPOSITE_RENDER_ENABLED"] = previous

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["stage"], "complete")
        self.assertEqual(payload["generation"]["rendered_output"]["status"], "compiled")
        self.assertEqual(payload["visuals"]["visual_assets"][0]["id"], "visual_1")
        self.assertEqual(payload["composite"]["composite_assets"][0]["kind"], "problem")

    def test_options_returns_cors_headers(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(FakeClient([])),
            ComplianceValidatorAgent(FakeClient([])),
        )
        server = self.start_server(service, allowed_origin="https://example.test")

        status, headers = self.options(server, "/generate")

        self.assertEqual(status, 204)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "https://example.test")
        self.assertIn("X-API-Key", headers["Access-Control-Allow-Headers"])

    def test_send_json_ignores_disconnected_clients(self):
        handler = FakeHandlerForSendJson()

        MathGenerationHandler._send_json(handler, 200, {"status": "success"})

        self.assertEqual(handler.status, 200)
        self.assertTrue(handler.ended)

    def test_visuals_generate_endpoint_returns_visual_asset(self):
        class FakeVisualService:
            tracer = type("FakeTracer", (), {"to_response": lambda self: ["visual trace"]})()

            def generate_visual(self, request):
                return {
                    "status": "success",
                    "visual_needed": True,
                    "visual_validation_status": "pass",
                    "visual_assets": [
                        {
                            "mime_type": "text/x-tex",
                            "tikz_source": "\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}",
                            "placement": "problem",
                            "purpose": "diagram",
                            "prompt": "Draw a cart.",
                        }
                    ],
                    "instruction": {"needed": True},
                    "validation": {"status": "pass"},
                    "attempts": [],
                }

        service = MathGenerationService(
            GeneratorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(FakeClient([])),
            ComplianceValidatorAgent(FakeClient([])),
        )
        server = self.start_server(service, visual_service=FakeVisualService())

        status, payload = self.post_json(
            server,
            {
                "subject": "physics",
                "level": "University of Tokyo",
                "topic": "Mechanics",
                "problem": "問題文",
                "solution": "解答",
                "trace": True,
            },
            path="/visuals/generate",
        )

        self.assertEqual(status, 200)
        self.assertEqual(payload["status"], "success")
        self.assertIn("\\begin{tikzpicture}", payload["visual_assets"][0]["tikz_source"])
        self.assertEqual(payload["trace"], ["visual trace"])
