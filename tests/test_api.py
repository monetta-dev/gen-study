import json
import threading
import unittest
import urllib.error
import urllib.request

from mathgen.agents import GeneratorAgent, MathGenerationService, ValidatorAgent
from mathgen.api import create_server


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)

    def chat(self, messages):
        if isinstance(self.responses[0], Exception):
            raise self.responses.pop(0)
        return self.responses.pop(0)


class ApiTests(unittest.TestCase):
    def start_server(self, service):
        server = create_server(
            host="127.0.0.1",
            port=0,
            service=service,
            debug_tracebacks=False,
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        return server

    def post_json(self, server, payload):
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{server.server_port}/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode("utf-8"))

    def test_generate_endpoint_returns_verified_problem(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient(['{"problem":"問 $x$","solution":"解 $x=1$"}'])),
            ValidatorAgent(
                FakeClient(
                    ['{"passed":true,"solution_accurate":true,"curriculum_compliant":true,"issues":[]}']
                )
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

    def test_generate_endpoint_rejects_invalid_request(self):
        service = MathGenerationService(
            GeneratorAgent(FakeClient([])),
            ValidatorAgent(FakeClient([])),
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
            ValidatorAgent(FakeClient([])),
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
        self.assertEqual(payload["detail"], "provider disconnected")
