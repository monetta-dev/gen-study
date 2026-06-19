"""Render generated problem and solution text to PNG pages."""

from __future__ import annotations

import base64
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


class ProblemRenderError(RuntimeError):
    """Raised when problem/solution rendering fails."""


@dataclass(frozen=True)
class RenderedPage:
    kind: str
    page: int
    title: str
    mime_type: str
    png_base64: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "page": self.page,
            "title": self.title,
            "mime_type": self.mime_type,
            "png_base64": self.png_base64,
        }


@dataclass(frozen=True)
class RenderedProblemOutput:
    status: str
    format: str
    assets: tuple[RenderedPage, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "format": self.format,
            "assets": [asset.to_dict() for asset in self.assets],
        }


class ProblemRenderer:
    def __init__(self, timeout_seconds: float = 40.0, dpi: int = 180) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if dpi <= 0:
            raise ValueError("dpi must be positive")
        self.timeout_seconds = timeout_seconds
        self.dpi = dpi

    def render(self, problem: str, solution: str) -> RenderedProblemOutput:
        assets = [
            *self._render_document("problem", "問題", problem),
            *self._render_document("solution", "解答", solution),
        ]
        return RenderedProblemOutput(
            status="compiled",
            format="png",
            assets=tuple(assets),
        )

    def _render_document(self, kind: str, title: str, body: str) -> list[RenderedPage]:
        with tempfile.TemporaryDirectory(prefix=f"gen-study-{kind}-") as tmpdir:
            workdir = Path(tmpdir)
            tex_path = workdir / "main.tex"
            tex_path.write_text(_document(title, body), encoding="utf-8")
            _run_lualatex(workdir, self.timeout_seconds)
            pdf_path = workdir / "main.pdf"
            if not pdf_path.exists():
                raise ProblemRenderError(f"{kind} PDF was not produced")
            png_paths = _pdf_to_png_pages(pdf_path, workdir, self.timeout_seconds, self.dpi)
            return [
                RenderedPage(
                    kind=kind,
                    page=index,
                    title=title,
                    mime_type="image/png",
                    png_base64=base64.b64encode(path.read_bytes()).decode("ascii"),
                )
                for index, path in enumerate(png_paths, start=1)
            ]


def _document(title: str, body: str) -> str:
    return "\n".join(
        [
            "\\documentclass[12pt,a4paper]{ltjsarticle}",
            "\\usepackage[margin=22mm]{geometry}",
            "\\usepackage{amsmath,amssymb,mathtools}",
            "\\usepackage{enumitem}",
            "\\usepackage{mhchem}",
            "\\usepackage{physics}",
            "\\setlength{\\parindent}{0pt}",
            "\\setlength{\\parskip}{0.7em}",
            "\\begin{document}",
            f"\\section*{{{title}}}",
            body,
            "\\end{document}",
            "",
        ]
    )


def _run_lualatex(workdir: Path, timeout_seconds: float) -> None:
    if not shutil.which("lualatex"):
        raise ProblemRenderError("`lualatex` is not installed")
    command = [
        "lualatex",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "main.tex",
    ]
    completed = subprocess.run(
        command,
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        log_path = workdir / "main.log"
        log_excerpt = ""
        if log_path.exists():
            log_excerpt = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        raise ProblemRenderError(
            "LaTeX rendering failed: "
            f"{(completed.stderr or completed.stdout or log_excerpt)[-3000:]}"
        )


def _pdf_to_png_pages(
    pdf_path: Path, workdir: Path, timeout_seconds: float, dpi: int
) -> list[Path]:
    if not shutil.which("pdftoppm"):
        raise ProblemRenderError("`pdftoppm` is not installed")
    output_prefix = workdir / "page"
    completed = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            str(dpi),
            str(pdf_path),
            str(output_prefix),
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    paths = sorted(workdir.glob("page-*.png"))
    if completed.returncode != 0 or not paths:
        raise ProblemRenderError(
            "PDF to PNG conversion failed: "
            f"{(completed.stderr or completed.stdout)[-3000:]}"
        )
    return paths
