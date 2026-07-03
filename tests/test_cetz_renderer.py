import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mathgen.cetz_renderer import (
    CetzRenderer,
    _run_typst_compile,
    _typst_document,
)


class CetzRendererTests(unittest.TestCase):
    def test_typst_document_wraps_cetz_source(self):
        document = _typst_document("import cetz.draw: *\nline((0,0), (1,0))")

        self.assertIn('#import "@preview/cetz:0.5.2"', document)
        self.assertIn("#cetz.canvas({", document)
        self.assertIn("line((0,0), (1,0))", document)

    def test_missing_typst_returns_command_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("shutil.which", return_value=None):
                result = _run_typst_compile(Path(tmpdir), "main.typ", "out.png", 30, 180)

        self.assertEqual(result.returncode, 127)
        self.assertIn("typst", result.stderr)

    def test_renderer_returns_failed_result_when_typst_is_missing(self):
        with patch("shutil.which", return_value=None):
            result = CetzRenderer().render_png("import cetz.draw: *\nline((0,0), (1,0))")

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.engine, "typst")
        self.assertIn("Command not found", result.issues[0])


if __name__ == "__main__":
    unittest.main()
