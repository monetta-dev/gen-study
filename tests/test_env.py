import os
import tempfile
import unittest
from pathlib import Path

from mathgen.env import load_dotenv


class EnvTests(unittest.TestCase):
    def test_load_dotenv_defaults_to_env_local(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            previous_cwd = os.getcwd()
            previous_value = os.environ.get("GENERATION_TRACE")
            os.environ.pop("GENERATION_TRACE", None)
            try:
                os.chdir(tmpdir)
                Path(".env").write_text("GENERATION_TRACE=false\n", encoding="utf-8")
                Path(".env.local").write_text("GENERATION_TRACE=true\n", encoding="utf-8")

                load_dotenv()

                self.assertEqual(os.environ["GENERATION_TRACE"], "true")
            finally:
                os.chdir(previous_cwd)
                if previous_value is None:
                    os.environ.pop("GENERATION_TRACE", None)
                else:
                    os.environ["GENERATION_TRACE"] = previous_value

    def test_load_dotenv_sets_values_without_overriding_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "OPENROUTER_API_KEY=from-file",
                        "OPENROUTER_TIMEOUT_SECONDS='600'",
                        "# ignored comment",
                    ]
                ),
                encoding="utf-8",
            )

            previous_key = os.environ.get("OPENROUTER_API_KEY")
            previous_timeout = os.environ.get("OPENROUTER_TIMEOUT_SECONDS")
            os.environ["OPENROUTER_API_KEY"] = "from-env"
            os.environ.pop("OPENROUTER_TIMEOUT_SECONDS", None)
            try:
                load_dotenv(env_path)

                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "from-env")
                self.assertEqual(os.environ["OPENROUTER_TIMEOUT_SECONDS"], "600")
            finally:
                if previous_key is None:
                    os.environ.pop("OPENROUTER_API_KEY", None)
                else:
                    os.environ["OPENROUTER_API_KEY"] = previous_key

                if previous_timeout is None:
                    os.environ.pop("OPENROUTER_TIMEOUT_SECONDS", None)
                else:
                    os.environ["OPENROUTER_TIMEOUT_SECONDS"] = previous_timeout

    def test_load_dotenv_refreshes_values_it_loaded_earlier(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env.local"
            env_path.write_text("GENERATOR_MODEL=\n", encoding="utf-8")

            previous_model = os.environ.get("GENERATOR_MODEL")
            os.environ.pop("GENERATOR_MODEL", None)
            try:
                load_dotenv(env_path)
                self.assertEqual(os.environ["GENERATOR_MODEL"], "")

                env_path.write_text("GENERATOR_MODEL=qwen3.7-plus\n", encoding="utf-8")
                load_dotenv(env_path)

                self.assertEqual(os.environ["GENERATOR_MODEL"], "qwen3.7-plus")
            finally:
                if previous_model is None:
                    os.environ.pop("GENERATOR_MODEL", None)
                else:
                    os.environ["GENERATOR_MODEL"] = previous_model
