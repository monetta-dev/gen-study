import http.client
import io
import json
import os
import urllib.error
import unittest
from unittest.mock import patch

from mathgen.llm import (
    GPT_61_SOL_MODEL, LLMError, OpenRouterClient, OpenRouterQwenClient,
    QWEN_MODEL, QWEN_PLUS_MODEL,
)


class FakeMessagesResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps(
            {"choices": [{"message": {"content": '{"problem":"p","solution":"s"}'}}]}
        ).encode("utf-8")


class FakeStreamingMessagesResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"{\\"problem\\":\\"p\\","}}]}\n',
            '\n',
            'data: {"choices":[{"delta":{"content":"\\"solution\\":\\"s\\"}"}}]}\n',
            'data: [DONE]\n',
        ]
        return iter(line.encode("utf-8") for line in lines)


class FakeSingleChunkStreamingResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        lines = [
            'data: {"choices":[{"delta":{"content":"{\\"problem\\":\\"p\\",\\"solution\\":\\"s\\"}"}}]}\n',
            'data: [DONE]\n',
        ]
        return iter(line.encode("utf-8") for line in lines)


class FakeEmptyStreamingResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def __iter__(self):
        lines = [
            'event: message_start\n',
            'data: {"type":"message_start","message":{"content":[]}}\n',
            '\n',
            'event: message_stop\n',
            'data: {"type":"message_stop"}\n',
        ]
        return iter(line.encode("utf-8") for line in lines)


class OpenRouterQwenClientTests(unittest.TestCase):
    def setUp(self):
        self.previous_env = {
            key: os.environ.get(key)
            for key in (
                "OPENROUTER_API_KEY",
                "OPENROUTER_BASE_URL",
                "OPENROUTER_MODEL",
                "OPENROUTER_REASONING_EFFORT",
                "OPENROUTER_TIMEOUT_SECONDS",
                "OPENROUTER_MAX_TOKENS",
                "OPENROUTER_THINKING_BUDGET_TOKENS",
                "OPENROUTER_THINKING_ENABLED",
                "OPENROUTER_STREAM",
                "OPENROUTER_RETRY_ATTEMPTS",
                "OPENROUTER_RETRY_DELAY_SECONDS",
            )
        }
        for key in self.previous_env:
            os.environ.pop(key, None)
        os.environ["OPENROUTER_BASE_URL"] = (
            "https://openrouter.ai/api/v1"
        )

    def tearDown(self):
        for key, value in self.previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_default_sol_streams_messages_with_reasoning_effort(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeStreamingMessagesResponse()

        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"

        client = OpenRouterQwenClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            content = client.chat(
                [
                    {"role": "system", "content": "Return strict JSON."},
                    {"role": "user", "content": "Generate a question."},
                ]
            )

        self.assertEqual(content, '{"problem":"p","solution":"s"}')
        self.assertEqual(captured["url"], "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(captured["timeout"], 300)
        self.assertEqual(captured["headers"]["Authorization"], "Bearer dashscope-key")
        self.assertEqual(captured["payload"]["model"], GPT_61_SOL_MODEL)
        self.assertEqual(captured["payload"]["max_tokens"], 32768)
        self.assertEqual(captured["payload"]["stream"], True)
        self.assertEqual(
            captured["payload"]["messages"],
            [
                {"role": "system", "content": "Return strict JSON."},
                {"role": "user", "content": "Generate a question."},
            ],
        )
        self.assertEqual(
            captured["payload"]["reasoning"],
            {"effort": "high", "exclude": True},
        )

        self.assertNotIn("temperature", captured["payload"])

    def test_sol_alias_and_reasoning_effort(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeMessagesResponse()

        os.environ["OPENROUTER_MODEL"] = "gpt-6.1-sol"
        os.environ["OPENROUTER_REASONING_EFFORT"] = "medium"
        # Qwen budgets must not affect GPT requests.
        os.environ["OPENROUTER_THINKING_BUDGET_TOKENS"] = "0"
        client = OpenRouterClient(api_key="test-key", stream=False)
        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat([{"role": "user", "content": "hello"}])
        self.assertEqual(captured["payload"]["model"], GPT_61_SOL_MODEL)
        self.assertEqual(captured["payload"]["reasoning"],
                         {"effort": "medium", "exclude": True})
        self.assertNotIn("temperature", captured["payload"])
        self.assertEqual(OpenRouterClient(reasoning_effort="low").reasoning_effort, "low")

    def test_sol_rejects_unsupported_reasoning_settings(self):
        for effort in ("none", "minimal", "invalid"):
            with self.subTest(effort=effort):
                with self.assertRaisesRegex(LLMError, "OPENROUTER_REASONING_EFFORT"):
                    OpenRouterClient(reasoning_effort=effort)
        client = OpenRouterClient(api_key="test-key", thinking_enabled=False)
        with patch("urllib.request.urlopen") as request:
            with self.assertRaisesRegex(LLMError, "requires thinking"):
                client.chat([{"role": "user", "content": "hello"}])
            request.assert_not_called()

    def test_blank_model_uses_sol_default(self):
        os.environ["OPENROUTER_MODEL"] = " "
        os.environ["OPENROUTER_REASONING_EFFORT"] = " "
        self.assertEqual(OpenRouterClient().model, GPT_61_SOL_MODEL)
        self.assertEqual(OpenRouterClient().reasoning_effort, "high")

    def test_streaming_can_be_disabled(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeMessagesResponse()

        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"
        os.environ["OPENROUTER_STREAM"] = "false"

        client = OpenRouterQwenClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            content = client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(content, '{"problem":"p","solution":"s"}')
        self.assertNotIn("stream", captured["payload"])

    def test_custom_base_url_adds_chat_completions_path(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            return FakeMessagesResponse()

        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"
        os.environ["OPENROUTER_BASE_URL"] = (
            "https://example.test/api/v1"
        )
        os.environ["OPENROUTER_STREAM"] = "false"

        client = OpenRouterQwenClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(
            captured["url"],
            "https://example.test/api/v1/chat/completions",
        )

    def test_global_model_can_use_qwen_plus(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeMessagesResponse()

        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"
        os.environ["OPENROUTER_MODEL"] = QWEN_PLUS_MODEL
        os.environ["OPENROUTER_STREAM"] = "false"

        client = OpenRouterQwenClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(captured["payload"]["model"], QWEN_PLUS_MODEL)

    def test_legacy_role_model_alias_maps_to_openrouter_id(self):
        os.environ["OPENROUTER_API_KEY"] = "openrouter-key"

        client = OpenRouterQwenClient(model="qwen3.7-plus")

        self.assertEqual(client.model, QWEN_PLUS_MODEL)

    def test_explicit_model_overrides_global_model(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeMessagesResponse()

        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"
        os.environ["OPENROUTER_MODEL"] = QWEN_PLUS_MODEL
        os.environ["OPENROUTER_STREAM"] = "false"

        client = OpenRouterQwenClient(model=QWEN_MODEL)

        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(captured["payload"]["model"], QWEN_MODEL)

    def test_thinking_can_be_disabled(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeMessagesResponse()

        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"
        os.environ["OPENROUTER_STREAM"] = "false"
        os.environ["OPENROUTER_THINKING_ENABLED"] = "false"

        client = OpenRouterQwenClient(model=QWEN_MODEL)

        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(captured["payload"]["reasoning"], {"enabled": False})

    def test_messages_preserve_multimodal_content_blocks(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeMessagesResponse()

        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"
        os.environ["OPENROUTER_STREAM"] = "false"

        client = OpenRouterQwenClient()

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
            captured["payload"]["messages"][1]["content"][1]["image_url"]["url"],
            "data:image/png;base64,png-base64",
        )

    def test_stream_reader_accepts_single_content_chunk(self):
        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"

        client = OpenRouterQwenClient()

        with patch("urllib.request.urlopen", lambda request, timeout: FakeSingleChunkStreamingResponse()):
            content = client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(content, '{"problem":"p","solution":"s"}')

    def test_empty_stream_retries_then_succeeds(self):
        calls = []

        def fake_urlopen(request, timeout):
            calls.append(request.full_url)
            if len(calls) == 1:
                return FakeEmptyStreamingResponse()
            return FakeStreamingMessagesResponse()

        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"
        os.environ["OPENROUTER_RETRY_ATTEMPTS"] = "1"
        os.environ["OPENROUTER_RETRY_DELAY_SECONDS"] = "0"

        client = OpenRouterQwenClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            content = client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(content, '{"problem":"p","solution":"s"}')
        self.assertEqual(len(calls), 2)

    def test_blank_numeric_environment_values_use_defaults(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeMessagesResponse()

        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"
        os.environ["OPENROUTER_MAX_TOKENS"] = ""
        os.environ["OPENROUTER_THINKING_BUDGET_TOKENS"] = ""
        os.environ["OPENROUTER_STREAM"] = "false"

        client = OpenRouterQwenClient(model=QWEN_MODEL)

        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(captured["payload"]["max_tokens"], 32768)
        self.assertEqual(captured["payload"]["reasoning"]["max_tokens"], 16384)

    def test_thinking_budget_is_independent_from_max_tokens(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeMessagesResponse()

        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"
        os.environ["OPENROUTER_MAX_TOKENS"] = "8192"
        os.environ["OPENROUTER_THINKING_BUDGET_TOKENS"] = "8192"
        os.environ["OPENROUTER_STREAM"] = "false"

        client = OpenRouterQwenClient(model=QWEN_MODEL)

        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(captured["payload"]["max_tokens"], 8192)
        self.assertEqual(captured["payload"]["reasoning"]["max_tokens"], 8192)

    def test_rejects_non_positive_thinking_budget(self):
        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"
        os.environ["OPENROUTER_THINKING_BUDGET_TOKENS"] = "0"

        client = OpenRouterQwenClient(model=QWEN_MODEL)

        with self.assertRaisesRegex(LLMError, "must be positive"):
            client.chat([{"role": "user", "content": "hello"}])

    def test_requires_api_key(self):
        os.environ.pop("OPENROUTER_API_KEY", None)

        client = OpenRouterQwenClient()

        with self.assertRaisesRegex(LLMError, "OPENROUTER_API_KEY"):
            client.chat([{"role": "user", "content": "hello"}])

    def test_rejects_unresolved_workspace_placeholder(self):
        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"
        os.environ["OPENROUTER_BASE_URL"] = (
            "https://{WorkspaceId}.ap-northeast-1.maas.aliyuncs.com/"
            "compatible-mode/v1"
        )

        client = OpenRouterQwenClient()

        with self.assertRaisesRegex(LLMError, "contains a placeholder"):
            client.chat([{"role": "user", "content": "hello"}])

    def test_rejects_unsupported_model_override(self):
        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"
        os.environ["OPENROUTER_MODEL"] = "other-model"

        with self.assertRaisesRegex(LLMError, "qwen3.7"):
            OpenRouterQwenClient()

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
            return FakeMessagesResponse()

        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"
        os.environ["OPENROUTER_STREAM"] = "false"
        os.environ["OPENROUTER_RETRY_ATTEMPTS"] = "1"
        os.environ["OPENROUTER_RETRY_DELAY_SECONDS"] = "0"

        client = OpenRouterQwenClient()

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

        os.environ["OPENROUTER_API_KEY"] = "dashscope-key"
        os.environ["OPENROUTER_RETRY_ATTEMPTS"] = "2"

        client = OpenRouterQwenClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(LLMError, "HTTP 403"):
                client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(len(calls), 1)

    def test_chat_wraps_connection_failures(self):
        def fake_urlopen(request, timeout):
            raise http.client.RemoteDisconnected("remote end closed connection")

        client = OpenRouterQwenClient(api_key="test-key", base_url="https://example.test")

        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(LLMError, "connection failed"):
                client.chat([{"role": "user", "content": "hello"}])
