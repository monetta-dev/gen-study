import json
import http.client
import unittest
from unittest.mock import patch

from mathgen.deepseek import DeepSeekClient, DeepSeekError


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": '{"problem":"p","solution":"s"}'}}]}
        ).encode("utf-8")


class DeepSeekClientTests(unittest.TestCase):
    def test_chat_sends_model_json_contract_and_thinking_mode(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        client = DeepSeekClient(
            api_key="test-key",
            base_url="https://example.test",
            model="deepseek-v4-pro",
            timeout_seconds=12,
        )

        with patch("urllib.request.urlopen", fake_urlopen):
            content = client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(content, '{"problem":"p","solution":"s"}')
        self.assertEqual(captured["url"], "https://example.test/chat/completions")
        self.assertEqual(captured["timeout"], 12)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(captured["payload"]["model"], "deepseek-v4-pro")
        self.assertEqual(captured["payload"]["response_format"], {"type": "json_object"})
        self.assertEqual(captured["payload"]["thinking"], {"type": "enabled"})

    def test_default_timeout_allows_long_generation(self):
        client = DeepSeekClient(api_key="test-key")

        self.assertEqual(client.timeout_seconds, 300)

    def test_chat_wraps_connection_failures_as_deepseek_error(self):
        def fake_urlopen(request, timeout):
            raise http.client.RemoteDisconnected("remote end closed connection")

        client = DeepSeekClient(api_key="test-key", base_url="https://example.test")

        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(DeepSeekError, "connection failed"):
                client.chat([{"role": "user", "content": "hello"}])
