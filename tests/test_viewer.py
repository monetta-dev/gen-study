from pathlib import Path
import unittest


class ViewerTests(unittest.TestCase):
    def test_viewer_static_files_include_latex_and_svg_rendering_hooks(self):
        root = Path(__file__).resolve().parents[1]
        html = (root / "viewer" / "index.html").read_text(encoding="utf-8")
        script = (root / "viewer" / "app.js").read_text(encoding="utf-8")
        styles = (root / "viewer" / "styles.css").read_text(encoding="utf-8")

        self.assertIn("MathJax", html)
        self.assertIn("tex-svg.js", html)
        self.assertIn("problemOutput", html)
        self.assertIn("solutionOutput", html)
        self.assertIn("diagramOutput", html)
        self.assertIn("iframe.srcdoc", script)
        self.assertIn("diagram.svg", script)
        self.assertIn("typesetPromise", script)
        self.assertIn(".diagram-output", styles)
