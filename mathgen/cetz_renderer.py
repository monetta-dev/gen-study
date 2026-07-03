"""CeTZ/Typst compilation and PNG rendering."""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .models import ContractError


@dataclass(frozen=True)
class CetzRenderResult:
    status: str
    png_base64: str
    mime_type: str
    compile_transcript: tuple[dict[str, object], ...]
    command_count: int
    engine: str
    issues: tuple[str, ...] = ()

    def to_asset_fields(self) -> dict[str, object]:
        return {
            "mime_type": self.mime_type,
            "png_base64": self.png_base64,
            "compile_status": self.status,
            "compile_engine": self.engine,
            "compile_command_count": self.command_count,
            "compile_transcript": list(self.compile_transcript),
        }


class CetzRenderer:
    def __init__(self, timeout_seconds: float = 30.0, ppi: int = 180) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if ppi <= 0:
            raise ValueError("ppi must be positive")
        self.timeout_seconds = timeout_seconds
        self.ppi = ppi

    def render_png(self, cetz_source: str) -> CetzRenderResult:
        with tempfile.TemporaryDirectory(prefix="gen-study-cetz-") as tmpdir:
            workdir = Path(tmpdir)
            typ_path = workdir / "main.typ"
            png_path = workdir / "rendered.png"
            typ_path.write_text(_typst_document(cetz_source), encoding="utf-8")
            result = _run_typst_compile(
                workdir,
                typ_path.name,
                png_path.name,
                self.timeout_seconds,
                self.ppi,
            )
            transcript = (
                {
                    "action": "run_command",
                    "argv": [
                        "typst",
                        "compile",
                        "--ppi",
                        str(self.ppi),
                        typ_path.name,
                        png_path.name,
                    ],
                    "returncode": result.returncode,
                    "stdout": _tail(result.stdout),
                    "stderr": _tail(result.stderr),
                    "files": _list_files(workdir),
                },
            )
            if result.returncode != 0 or not png_path.exists():
                return CetzRenderResult(
                    status="failed",
                    png_base64="",
                    mime_type="image/png",
                    compile_transcript=transcript,
                    command_count=1,
                    engine="typst",
                    issues=(
                        "Typst/CeTZ compilation failed: "
                        f"{_tail(result.stderr or result.stdout)}",
                    ),
                )
            return CetzRenderResult(
                status="compiled",
                png_base64=base64.b64encode(png_path.read_bytes()).decode("ascii"),
                mime_type="image/png",
                compile_transcript=transcript,
                command_count=1,
                engine="typst",
            )


@dataclass(frozen=True)
class _CompletedCommand:
    returncode: int
    stdout: str
    stderr: str


def _run_typst_compile(
    workdir: Path,
    source_name: str,
    output_name: str,
    timeout_seconds: float,
    ppi: int,
) -> _CompletedCommand:
    if not shutil.which("typst"):
        return _CompletedCommand(
            returncode=127,
            stdout="",
            stderr="Command not found: typst",
        )
    _validate_relative_name(source_name)
    _validate_relative_name(output_name)
    try:
        completed = subprocess.run(
            [
                "typst",
                "compile",
                "--ppi",
                str(ppi),
                source_name,
                output_name,
            ],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _CompletedCommand(
            returncode=124,
            stdout="",
            stderr=f"Typst/CeTZ rendering timed out after {timeout_seconds:.1f} seconds",
        )
    return _CompletedCommand(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _typst_document(cetz_source: str) -> str:
    return "\n".join(
        [
            '#import "@preview/cetz:0.5.2"',
            "",
            '#set page(width: auto, height: auto, margin: 4pt)',
            '#set text(lang: "ja", font: ("Noto Serif CJK JP", "Noto Sans CJK JP", "New Computer Modern"))',
            "",
            "#cetz.canvas({",
            cetz_source,
            "})",
            "",
        ]
    )


def _validate_relative_name(value: str) -> None:
    if os.path.isabs(value) or "/" in value or "\\" in value:
        raise ContractError("Typst compile paths must be local file names")


def _list_files(workdir: Path) -> list[str]:
    return sorted(path.name for path in workdir.iterdir() if path.is_file())


def _tail(value: str, limit: int = 2000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[-limit:]
