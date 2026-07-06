import subprocess
import unittest
from unittest.mock import patch

from mathgen.cetz_converter import TylaxCetzConverter, _extract_cetz_canvas_body
from mathgen.models import ContractError


class TylaxCetzConverterTests(unittest.TestCase):
    def test_extracts_canvas_body_from_typst_document(self):
        output = """#import "@preview/cetz:0.5.2"

#cetz.canvas({
  import cetz.draw: *
  line((0, 0), (1, 0))
})
"""

        body = _extract_cetz_canvas_body(output)

        self.assertIn("line((0, 0), (1, 0))", body)
        self.assertNotIn("#cetz.canvas", body)
        self.assertNotIn("#import", body)

    def test_converts_tikz_with_tylax_command(self):
        converter = TylaxCetzConverter(timeout_seconds=3.0)

        def fake_run(argv, cwd, capture_output, text, timeout, check):
            self.assertEqual(argv, ["t2l", "tikz", "input.tex", "-o", "output.typ"])
            self.assertEqual(timeout, 3.0)
            (cwd / "output.typ").write_text(
                "#cetz.canvas({\nimport cetz.draw: *\nline((0,0), (1,0))\n})",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(argv, 0, "", "")

        with patch("mathgen.cetz_converter.shutil.which", return_value="/usr/bin/t2l"):
            with patch("mathgen.cetz_converter.subprocess.run", side_effect=fake_run):
                result = converter.convert_tikz_to_cetz(
                    "\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}"
                )

        self.assertIn("line((0,0), (1,0))", result.cetz_source)
        self.assertEqual(result.transcript[0]["returncode"], 0)

    def test_reports_missing_tylax_command(self):
        converter = TylaxCetzConverter()

        with patch("mathgen.cetz_converter.shutil.which", return_value=None):
            with self.assertRaises(ContractError) as context:
                converter.convert_tikz_to_cetz(
                    "\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}"
                )

        self.assertIn("Command not found", str(context.exception))


if __name__ == "__main__":
    unittest.main()
