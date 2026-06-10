import json
import http.client
import io
import os
import urllib.error
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


class DeepSeekClientTests(unittest.TestCase):
    def setUp(self):
        self.previous_env = {
            key: os.environ.get(key)
            for key in (
                "LLM_PROVIDER",
                "LLM_API_KEY",
                "LLM_BASE_URL",
                "LLM_MODEL",
                "LLM_TIMEOUT_SECONDS",
                "LLM_API_STYLE",
                "LLM_MAX_TOKENS",
                "LLM_THINKING_BUDGET_TOKENS",
                "LLM_THINKING_DISPLAY",
                "LLM_ANTHROPIC_VERSION",
                "LLM_THINKING_MODE",
                "LLM_THINKING_FIELD",
                "LLM_USER_AGENT",
                "LLM_RETRY_ATTEMPTS",
                "LLM_RETRY_DELAY_SECONDS",
                "LLM_STREAM",
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
        self.assertEqual(captured["headers"]["User-agent"], "gen-study/0.1 Python LLM client")
        self.assertEqual(captured["payload"]["model"], "deepseek-v4-pro")
        self.assertEqual(captured["payload"]["response_format"], {"type": "json_object"})
        self.assertEqual(captured["payload"]["thinking"], {"type": "enabled"})

    def test_chat_can_use_qwen_compatible_configuration_without_thinking(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        os.environ["LLM_PROVIDER"] = "qwen"
        os.environ["LLM_API_KEY"] = "qwen-key"
        os.environ["LLM_BASE_URL"] = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        os.environ["LLM_MODEL"] = "qwen-max"
        os.environ["LLM_THINKING_MODE"] = "disabled"

        client = DeepSeekClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(
            captured["url"],
            "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        )
        self.assertEqual(captured["payload"]["model"], "qwen-max")
        self.assertNotIn("thinking", captured["payload"])
        self.assertNotIn("enable_thinking", captured["payload"])

    def test_chat_can_send_custom_thinking_field(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeResponse()

        os.environ["LLM_PROVIDER"] = "qwen"
        os.environ["LLM_THINKING_MODE"] = "enabled"
        os.environ["LLM_THINKING_FIELD"] = "enable_thinking"

        client = DeepSeekClient(api_key="test-key", base_url="https://example.test")

        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(captured["payload"]["enable_thinking"], True)

    def test_chat_can_use_anthropic_compatible_messages_api(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeAnthropicResponse()

        os.environ["LLM_API_STYLE"] = "anthropic"
        os.environ["LLM_API_KEY"] = "opencode-key"
        os.environ["LLM_BASE_URL"] = "https://opencode.ai/zen/go/v1"
        os.environ["LLM_MODEL"] = "qwen3.7-max"
        os.environ["LLM_MAX_TOKENS"] = "4096"
        os.environ["LLM_THINKING_MODE"] = "disabled"

        client = DeepSeekClient()

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
        self.assertEqual(captured["headers"]["User-agent"], "gen-study/0.1 Python LLM client")
        self.assertEqual(captured["payload"]["model"], "qwen3.7-max")
        self.assertEqual(captured["payload"]["max_tokens"], 4096)
        self.assertEqual(captured["payload"]["system"], "Return strict JSON.")
        self.assertEqual(
            captured["payload"]["messages"],
            [{"role": "user", "content": "Generate a question."}],
        )
        self.assertNotIn("response_format", captured["payload"])

    def test_opencode_go_qwen_max_infers_messages_api_and_enables_thinking(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeStreamingAnthropicResponse()

        os.environ["LLM_PROVIDER"] = "opencode-go"
        os.environ["LLM_API_KEY"] = "opencode-key"
        os.environ["LLM_MODEL"] = "qwen3.7-max"
        os.environ["LLM_MAX_TOKENS"] = "8192"
        os.environ["LLM_THINKING_MODE"] = "enabled"
        os.environ["LLM_THINKING_BUDGET_TOKENS"] = "4096"
        os.environ["LLM_THINKING_DISPLAY"] = "omitted"

        client = DeepSeekClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(captured["url"], "https://opencode.ai/zen/go/v1/messages")
        self.assertEqual(captured["payload"]["model"], "qwen3.7-max")
        self.assertEqual(
            captured["payload"]["thinking"],
            {
                "type": "enabled",
                "budget_tokens": 4096,
                "display": "omitted",
            },
        )
        self.assertEqual(captured["payload"]["stream"], True)

    def test_anthropic_streaming_collects_text_deltas(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeStreamingAnthropicResponse()

        os.environ["LLM_PROVIDER"] = "opencode-go"
        os.environ["LLM_API_KEY"] = "opencode-key"
        os.environ["LLM_MODEL"] = "qwen3.7-max"
        os.environ["LLM_THINKING_MODE"] = "enabled"

        client = DeepSeekClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            content = client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(content, '{"problem":"p","solution":"s"}')
        self.assertEqual(captured["payload"]["stream"], True)

    def test_anthropic_streaming_can_be_disabled(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeAnthropicResponse()

        os.environ["LLM_PROVIDER"] = "opencode-go"
        os.environ["LLM_API_KEY"] = "opencode-key"
        os.environ["LLM_MODEL"] = "qwen3.7-max"
        os.environ["LLM_STREAM"] = "false"
        os.environ["LLM_THINKING_MODE"] = "disabled"

        client = DeepSeekClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertNotIn("stream", captured["payload"])

    def test_blank_numeric_environment_values_use_defaults(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeAnthropicResponse()

        os.environ["LLM_PROVIDER"] = "opencode-go"
        os.environ["LLM_API_KEY"] = "opencode-key"
        os.environ["LLM_MODEL"] = "qwen3.7-max"
        os.environ["LLM_MAX_TOKENS"] = ""
        os.environ["LLM_THINKING_MODE"] = "enabled"
        os.environ["LLM_THINKING_BUDGET_TOKENS"] = ""
        os.environ["LLM_STREAM"] = "false"

        client = DeepSeekClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(captured["payload"]["max_tokens"], 4096)
        self.assertEqual(captured["payload"]["thinking"]["budget_tokens"], 1024)

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

        os.environ["LLM_PROVIDER"] = "opencode-go"
        os.environ["LLM_API_KEY"] = "opencode-key"
        os.environ["LLM_MODEL"] = "qwen3.7-max"
        os.environ["LLM_THINKING_MODE"] = "disabled"
        os.environ["LLM_RETRY_ATTEMPTS"] = "1"
        os.environ["LLM_RETRY_DELAY_SECONDS"] = "0"
        os.environ["LLM_STREAM"] = "false"

        client = DeepSeekClient()

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

        os.environ["LLM_API_KEY"] = "test-key"
        os.environ["LLM_RETRY_ATTEMPTS"] = "2"

        client = DeepSeekClient(base_url="https://example.test")

        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(DeepSeekError, "HTTP 403"):
                client.chat([{"role": "user", "content": "hello"}])

        self.assertEqual(len(calls), 1)

    def test_enabled_thinking_fails_when_provider_has_no_known_field(self):
        os.environ["LLM_PROVIDER"] = "unknown-provider"
        os.environ["LLM_API_KEY"] = "test-key"
        os.environ["LLM_THINKING_MODE"] = "enabled"

        client = DeepSeekClient(base_url="https://example.test")

        with self.assertRaisesRegex(DeepSeekError, "no supported thinking field"):
            client.chat([{"role": "user", "content": "hello"}])

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
