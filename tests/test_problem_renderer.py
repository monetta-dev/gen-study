import unittest

from mathgen.problem_renderer import RenderedPage, RenderedProblemOutput, _document


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


if __name__ == "__main__":
    unittest.main()
