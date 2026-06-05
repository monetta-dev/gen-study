import os
import tempfile
import unittest
from pathlib import Path

from mathgen.env import load_dotenv


class EnvTests(unittest.TestCase):
    def test_load_dotenv_sets_values_without_overriding_environment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "DEEPSEEK_API_KEY=from-file",
                        "DEEPSEEK_TIMEOUT_SECONDS='600'",
                        "# ignored comment",
                    ]
                ),
                encoding="utf-8",
            )

            previous_key = os.environ.get("DEEPSEEK_API_KEY")
            previous_timeout = os.environ.get("DEEPSEEK_TIMEOUT_SECONDS")
            os.environ["DEEPSEEK_API_KEY"] = "from-env"
            os.environ.pop("DEEPSEEK_TIMEOUT_SECONDS", None)
            try:
                load_dotenv(env_path)

                self.assertEqual(os.environ["DEEPSEEK_API_KEY"], "from-env")
                self.assertEqual(os.environ["DEEPSEEK_TIMEOUT_SECONDS"], "600")
            finally:
                if previous_key is None:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
                else:
                    os.environ["DEEPSEEK_API_KEY"] = previous_key

                if previous_timeout is None:
                    os.environ.pop("DEEPSEEK_TIMEOUT_SECONDS", None)
                else:
                    os.environ["DEEPSEEK_TIMEOUT_SECONDS"] = previous_timeout
