import io
import json
import os
import urllib.error
import unittest
from unittest.mock import patch

from mathgen.openai_images import ImageGenerationError, OpenAIImageClient


class FakeImageResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self):
        return json.dumps({"data": [{"b64_json": "png-base64"}]}).encode("utf-8")


class OpenAIImageClientTests(unittest.TestCase):
    def setUp(self):
        self.previous_env = {
            key: os.environ.get(key)
            for key in (
                "OPENAI_API_KEY",
                "OPENAI_BASE_URL",
                "OPENAI_IMAGE_MODEL",
                "OPENAI_IMAGE_SIZE",
                "OPENAI_IMAGE_QUALITY",
                "OPENAI_IMAGE_N",
                "OPENAI_IMAGE_TIMEOUT_SECONDS",
                "OPENAI_IMAGE_RETRY_ATTEMPTS",
                "OPENAI_IMAGE_RETRY_DELAY_SECONDS",
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

    def test_generates_image_with_gpt_image_default(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeImageResponse()

        os.environ["OPENAI_API_KEY"] = "openai-key"

        client = OpenAIImageClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            image_base64 = client.generate_png_base64("Draw a physics diagram.")

        self.assertEqual(image_base64, "png-base64")
        self.assertEqual(captured["url"], "https://api.openai.com/v1/images/generations")
        self.assertEqual(captured["headers"]["Authorization"], "Bearer openai-key")
        self.assertEqual(captured["payload"]["model"], "gpt-image-2")
        self.assertEqual(captured["payload"]["prompt"], "Draw a physics diagram.")
        self.assertNotIn("size", captured["payload"])
        self.assertNotIn("quality", captured["payload"])
        self.assertNotIn("n", captured["payload"])
        self.assertEqual(captured["timeout"], 300.0)

    def test_optional_image_parameters_are_sent_when_configured(self):
        captured = {}

        def fake_urlopen(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            return FakeImageResponse()

        os.environ["OPENAI_API_KEY"] = "openai-key"
        os.environ["OPENAI_IMAGE_SIZE"] = "1024x1024"
        os.environ["OPENAI_IMAGE_QUALITY"] = "high"
        os.environ["OPENAI_IMAGE_N"] = "1"

        client = OpenAIImageClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            client.generate_png_base64("Draw a physics diagram.")

        self.assertEqual(captured["payload"]["size"], "1024x1024")
        self.assertEqual(captured["payload"]["quality"], "high")
        self.assertEqual(captured["payload"]["n"], 1)

    def test_missing_api_key_is_clear(self):
        client = OpenAIImageClient()

        with self.assertRaisesRegex(ImageGenerationError, "OPENAI_API_KEY"):
            client.generate_png_base64("Draw a diagram.")

    def test_http_errors_include_response_body(self):
        def fake_urlopen(request, timeout):
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "bad request",
                {},
                io.BytesIO(b'{"error":{"message":"bad prompt"}}'),
            )

        client = OpenAIImageClient(api_key="openai-key")

        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(ImageGenerationError, "bad prompt"):
                client.generate_png_base64("Draw a diagram.")

    def test_retryable_http_error_retries_then_succeeds(self):
        calls = []

        def fake_urlopen(request, timeout):
            calls.append(request.full_url)
            if len(calls) == 1:
                raise urllib.error.HTTPError(
                    request.full_url,
                    520,
                    "unknown error",
                    {},
                    io.BytesIO(b'{"retryable":true,"retry_after":0}'),
                )
            return FakeImageResponse()

        os.environ["OPENAI_API_KEY"] = "openai-key"
        os.environ["OPENAI_IMAGE_RETRY_ATTEMPTS"] = "1"
        os.environ["OPENAI_IMAGE_RETRY_DELAY_SECONDS"] = "0"

        client = OpenAIImageClient()

        with patch("urllib.request.urlopen", fake_urlopen):
            image_base64 = client.generate_png_base64("Draw a diagram.")

        self.assertEqual(image_base64, "png-base64")
        self.assertEqual(len(calls), 2)

    def test_non_retryable_http_error_does_not_retry(self):
        calls = []

        def fake_urlopen(request, timeout):
            calls.append(request.full_url)
            raise urllib.error.HTTPError(
                request.full_url,
                400,
                "bad request",
                {},
                io.BytesIO(b'{"retryable":false}'),
            )

        client = OpenAIImageClient(api_key="openai-key")

        with patch("urllib.request.urlopen", fake_urlopen):
            with self.assertRaisesRegex(ImageGenerationError, "HTTP 400"):
                client.generate_png_base64("Draw a diagram.")

        self.assertEqual(len(calls), 1)
