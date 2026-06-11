import json
import threading
import unittest
import urllib.error
import urllib.request

from mathgen.agents import (
    AdjudicatorAgent,
    ComplianceValidatorAgent,
    CriticalValidatorAgent,
    GeneratorAgent,
    InteractionTracer,
    MathGenerationService,
    RepairAgent,
)
from mathgen.api import MathGenerationHandler, create_server
from mathgen.models import ModelOutputError
from mathgen.openai_images import ImageGenerationError


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
            AdjudicatorAgent(FakeClient([])),
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

    def test_generate_endpoint_can_return_process_trace(self):
        tracer = InteractionTracer(capture=True)
        service = MathGenerationService(
            GeneratorAgent(FakeClient(['{"problem":"問","solution":"解"}']), tracer=tracer),
            AdjudicatorAgent(FakeClient([]), tracer=tracer),
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

    def test_generate_endpoint_rejects_invalid_request(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient([])),
            AdjudicatorAgent(FakeClient([])),
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
            AdjudicatorAgent(FakeClient([])),
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
            AdjudicatorAgent(FakeClient([])),
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
            AdjudicatorAgent(FakeClient([])),
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
            AdjudicatorAgent(FakeClient([])),
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
            AdjudicatorAgent(FakeClient([])),
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
            AdjudicatorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(FakeClient([])),
            ComplianceValidatorAgent(FakeClient([])),
        )
        server = self.start_server(service)

        status, headers, body = self.get(server, "/")

        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn("generateForm", body)

    def test_options_returns_cors_headers(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient([])),
            AdjudicatorAgent(FakeClient([])),
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
                            "mime_type": "image/png",
                            "base64": "png-base64",
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
            AdjudicatorAgent(FakeClient([])),
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
        self.assertEqual(payload["visual_assets"][0]["base64"], "png-base64")
        self.assertEqual(payload["trace"], ["visual trace"])

    def test_visuals_generate_endpoint_returns_image_generation_error(self):
        class FakeVisualService:
            def generate_visual(self, request):
                raise ImageGenerationError("OPENAI_API_KEY is not configured")

        service = MathGenerationService(
            GeneratorAgent(FakeClient([])),
            AdjudicatorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(FakeClient([])),
            ComplianceValidatorAgent(FakeClient([])),
        )
        server = self.start_server(service, visual_service=FakeVisualService())

        with self.assertRaises(urllib.error.HTTPError) as context:
            self.post_json(
                server,
                {
                    "subject": "physics",
                    "problem": "問題文",
                    "solution": "解答",
                },
                path="/visuals/generate",
            )

        error = context.exception
        self.assertEqual(error.code, 502)
        payload = json.loads(error.read().decode("utf-8"))
        error.close()
        self.assertEqual(payload["error"], "image_generation_error")
