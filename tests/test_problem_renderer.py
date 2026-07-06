import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mathgen.problem_renderer import (
    ProblemRenderError,
    RenderedPage,
    RenderedProblemOutput,
    _document,
    _normalize_body_for_latex,
    _run_lualatex,
)


class ProblemRendererTests(unittest.TestCase):
    def test_document_uses_luatex_japanese_article_class(self):
        document = _document("問題", "二次関数 $y=x^2$ を考える。")

        self.assertIn("\\documentclass[12pt,a4paper]{ltjsarticle}", document)
        self.assertIn("\\section*{問題}", document)
        self.assertIn("二次関数 $y=x^2$ を考える。", document)

    def test_rendered_output_contract(self):
        output = RenderedProblemOutput(
            status="compiled",
            format="png",
            assets=(
                RenderedPage(
                    kind="problem",
                    page=1,
                    title="問題",
                    mime_type="image/png",
                    png_base64="abc",
                ),
            ),
        )

        self.assertEqual(output.to_dict()["assets"][0]["kind"], "problem")

    def test_normalizes_literal_escaped_newlines_for_latex(self):
        body = "条件を満たす。\\n$\\vec{a}=\\vec{OA}$\\r\\n次を求めよ。\\t以上。"

        normalized = _normalize_body_for_latex(body)

        self.assertEqual(
            normalized,
            "条件を満たす。\n$\\vec{a}=\\vec{OA}$\n次を求めよ。 以上。",
        )

    def test_normalization_preserves_latex_commands(self):
        body = "$\\frac{1}{2}\\theta + \\sum x_k + \\vec{a}$"

        self.assertEqual(_normalize_body_for_latex(body), body)

    def test_normalization_repairs_json_decoded_latex_control_escapes(self):
        body = "$s = \frac{1-t}{2}$ かつ $\theta = \theta_0$"

        self.assertEqual(
            _normalize_body_for_latex(body),
            "$s = \\frac{1-t}{2}$ かつ $\\theta = \\theta_0$",
        )

    def test_lualatex_timeout_is_render_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("shutil.which", return_value="/usr/bin/lualatex"):
                with patch(
                    "subprocess.run",
                    side_effect=subprocess.TimeoutExpired("lualatex", 180),
                ):
                    with self.assertRaisesRegex(
                        ProblemRenderError, "LaTeX rendering timed out"
                    ):
                        _run_lualatex(Path(tmpdir), 180)


if __name__ == "__main__":
    unittest.main()
