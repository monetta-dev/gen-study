import http.client
import io
import json
import os
import urllib.error
import unittest
from unittest.mock import patch

from mathgen.llm import LLMError, OpenCodeGoClient


class FakeOpenAIResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": '{"problem":"p","solution":"s"}'}}]}
        ).encode("utf-8")


class FakeAnthropicResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(
            {"content": [{"type": "text", "text": '{"problem":"p","solution":"s"}'}]}
        ).encode("utf-8")


class FakeStreamingAnthropicResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        lines = [
            'event: message_start\n',
            'data: {"type":"message_start","message":{"content":[]}}\n',
            '\n',
            'event: content_block_delta\n',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{\\"problem\\":\\"p\\","}}\n',
            '\n',
            'event: content_block_delta\n',
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"\\"solution\\":\\"s\\"}"}}\n',
            '\n',
            'event: message_stop\n',
            'data: {"type":"message_stop"}\n',
        ]
        return iter(line.encode("utf-8") for line in lines)


class OpenCodeGoClientTests(unittest.TestCase):
    def setUp(self):
        self.previous_env = {
            key: os.environ.get(key)
            for key in (
                "OPENCODE_GO_API_KEY",
                "OPENCODE_GO_BASE_URL",
                "OPENCODE_GO_MODEL",
                "OPENCODE_GO_TIMEOUT_SECONDS",
                "OPENCODE_GO_API_STYLE",
                "OPENCODE_GO_MAX_TOKENS",
                "OPENCODE_GO_THINKING_BUDGET_TOKENS",
                "OPENCODE_GO_THINKING_DISPLAY",
                "OPENCODE_GO_THINKING_ENABLED",
                "OPENCODE_GO_ANTHROPIC_VERSION",
                "OPENCODE_GO_STREAM",
                "OPENCODE_GO_USER_AGENT",
                "OPENCODE_GO_RETRY_ATTEMPTS",
                "OPENCODE_GO_RETRY_DELAY_SECONDS",
            )
        }
        for key in self.previous_env:
            os.environ.pop(key, None)

    def tearDown(self):
        for key, value in self.previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_qwen_max_streams_messages_with_large_thinking_budget(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeStreamingAnthropicResponse()

        os.environ["OPENCODE_GO_API_KEY"] = "opencode-key"
        os.environ["OPENCODE_GO_MODEL"] = "qwen3.7-max"

        client = OpenCodeGoClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            content = client.chat(
                [
                    {"role": "system", "content": "Return strict JSON."},
                    {"role": "user", "content": "Generate a question."},
                ]
            )

        self.assertEqual(content, '{"problem":"p","solution":"s"}')
        self.assertEqual(captured["url"], "https://opencode.ai/zen/go/v1/messages")
        self.assertEqual(captured["timeout"], 300)
        self.assertEqual(captured["headers"]["X-api-key"], "opencode-key")
        self.assertEqual(captured["headers"]["Anthropic-version"], "2023-06-01")
        self.assertEqual(captured["headers"]["User-agent"], "gen-study/0.1 OpenCode Go client")
        self.assertEqual(captured["payload"]["model"], "qwen3.7-max")
        self.assertEqual(captured["payload"]["max_tokens"], 32768)
        self.assertEqual(captured["payload"]["stream"], True)
        self.assertEqual(captured["payload"]["system"], "Return strict JSON.")
        self.assertEqual(
            captured["payload"]["messages"],
            [{"role": "user", "content": "Generate a question."}],
        )
        self.assertEqual(
            captured["payload"]["thinking"],
            {
                "type": "enabled",
                "budget_tokens": 16384,
                "display": "omitted",
            },
        )

    def test_anthropic_streaming_can_be_disabled(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeAnthropicResponse()

        os.environ["OPENCODE_GO_API_KEY"] = "opencode-key"
        os.environ["OPENCODE_GO_MODEL"] = "qwen3.7-max"
        os.environ["OPENCODE_GO_STREAM"] = "false"

        client = OpenCodeGoClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertNotIn("stream", captured["payload"])

    def test_anthropic_messages_preserve_multimodal_content_blocks(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeAnthropicResponse()

        os.environ["OPENCODE_GO_API_KEY"] = "opencode-key"
        os.environ["OPENCODE_GO_MODEL"] = "qwen3.7-plus"
        os.environ["OPENCODE_GO_STREAM"] = "false"

        client = OpenCodeGoClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat(
                [
                    {"role": "system", "content": "Return strict JSON."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Check this image."},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "png-base64",
                                },
                            },
                        ],
                    },
                ]
            )

        self.assertEqual(
            captured["payload"]["messages"][0]["content"][1]["source"]["data"],
            "png-base64",
        )

    def test_openai_compatible_opencode_model_uses_chat_completions(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeOpenAIResponse()

        os.environ["OPENCODE_GO_API_KEY"] = "opencode-key"
        os.environ["OPENCODE_GO_MODEL"] = "kimi-k2.6"

        client = OpenCodeGoClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            content = client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(content, '{"problem":"p","solution":"s"}')
        self.assertEqual(captured["url"], "https://opencode.ai/zen/go/v1/chat/completions")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer opencode-key")
        self.assertEqual(captured["payload"]["model"], "kimi-k2.6")
        self.assertEqual(captured["payload"]["response_format"], {"type": "json_object"})

    def test_blank_numeric_environment_values_use_defaults(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeAnthropicResponse()

        os.environ["OPENCODE_GO_API_KEY"] = "opencode-key"
        os.environ["OPENCODE_GO_MODEL"] = "qwen3.7-max"
        os.environ["OPENCODE_GO_MAX_TOKENS"] = ""
        os.environ["OPENCODE_GO_THINKING_BUDGET_TOKENS"] = ""
        os.environ["OPENCODE_GO_STREAM"] = "false"

        client = OpenCodeGoClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(captured["payload"]["max_tokens"], 32768)
        self.assertEqual(captured["payload"]["thinking"]["budget_tokens"], 16384)

    def test_rejects_thinking_budget_greater_than_max_tokens(self):
        os.environ["OPENCODE_GO_API_KEY"] = "opencode-key"
        os.environ["OPENCODE_GO_MODEL"] = "qwen3.7-max"
        os.environ["OPENCODE_GO_MAX_TOKENS"] = "8192"
        os.environ["OPENCODE_GO_THINKING_BUDGET_TOKENS"] = "8192"

        client = OpenCodeGoClient()

        with self.assertRaisesRegex(LLMError, "must be less"):
            client.chat([{"role": "user", "content": "hello"}])

    def test_retryable_http_error_retries_then_succeeds(self):
        calls = []

        def fake_urlopen(request, timeout):
            calls.append(request.full_url)
            if len(calls) == 1:
                raise urllib.error.HTTPError(
                    request.full_url,
                    524,
                    "timeout",
                    {},
                    io.BytesIO(b'{"retryable":true,"retry_after":0}'),
                )
            return FakeAnthropicResponse()

        os.environ["OPENCODE_GO_API_KEY"] = "opencode-key"
        os.environ["OPENCODE_GO_MODEL"] = "qwen3.7-max"
        os.environ["OPENCODE_GO_STREAM"] = "false"
        os.environ["OPENCODE_GO_RETRY_ATTEMPTS"] = "1"
        os.environ["OPENCODE_GO_RETRY_DELAY_SECONDS"] = "0"

        client = OpenCodeGoClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            content = client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(content, '{"problem":"p","solution":"s"}')
        self.assertEqual(len(calls), 2)

    def test_non_retryable_http_error_does_not_retry(self):
        calls = []

        def fake_urlopen(request, timeout):
            calls.append(request.full_url)
            raise urllib.error.HTTPError(
                request.full_url,
                403,
                "forbidden",
                {},
                io.BytesIO(b'{"retryable":false}'),
            )

        os.environ["OPENCODE_GO_API_KEY"] = "opencode-key"
        os.environ["OPENCODE_GO_RETRY_ATTEMPTS"] = "2"

        client = OpenCodeGoClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(LLMError, "HTTP 403"):
                client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(len(calls), 1)

    def test_unknown_model_requires_explicit_api_style(self):
        os.environ["OPENCODE_GO_API_KEY"] = "opencode-key"
        os.environ["OPENCODE_GO_MODEL"] = "new-model"

        with self.assertRaisesRegex(LLMError, "Unsupported OpenCode Go model"):
            OpenCodeGoClient()

    def test_chat_wraps_connection_failures(self):
        def fake_urlopen(request, timeout):
            raise http.client.RemoteDisconnected("remote end closed connection")

        client = OpenCodeGoClient(api_key="test-key", base_url="https://example.test")

        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(LLMError, "connection failed"):
                client.chat([{"role": "user", "content": "hello"}])
