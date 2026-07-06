"""TikZ to CeTZ conversion using Tylax."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .models import ContractError


@dataclass(frozen=True)
class CetzConversionResult:
    cetz_source: str
    transcript: tuple[dict[str, object], ...]


class TylaxCetzConverter:
    def __init__(
        self,
        command: str = "t2l",
        timeout_seconds: float = 30.0,
    ) -> None:
        if not command.strip():
            raise ValueError("command must be non-empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.command = command
        self.timeout_seconds = timeout_seconds

    def convert_tikz_to_cetz(self, tikz_source: str) -> CetzConversionResult:
        if not isinstance(tikz_source, str) or not tikz_source.strip():
            raise ContractError("TikZ source must be a non-empty string")
        if "\\begin{tikzpicture}" not in tikz_source:
            raise ContractError("TikZ source must contain a tikzpicture environment")
        if not shutil.which(self.command):
            raise ContractError(f"Command not found: {self.command}")

        with tempfile.TemporaryDirectory(prefix="gen-study-tylax-") as tmpdir:
            workdir = Path(tmpdir)
            input_path = workdir / "input.tex"
            output_path = workdir / "output.typ"
            input_path.write_text(tikz_source, encoding="utf-8")
            argv = [self.command, "tikz", input_path.name, "-o", output_path.name]
            try:
                completed = subprocess.run(
                    argv,
                    cwd=workdir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise ContractError(
                    f"Tylax conversion timed out after {self.timeout_seconds:.1f} seconds"
                ) from exc

            transcript = (
                {
                    "action": "run_command",
                    "argv": argv,
                    "returncode": completed.returncode,
                    "stdout": _tail(completed.stdout),
                    "stderr": _tail(completed.stderr),
                    "files": _list_files(workdir),
                },
            )
            if completed.returncode != 0:
                raise ContractError(
                    "Tylax TikZ-to-CeTZ conversion failed: "
                    f"{_tail(completed.stderr or completed.stdout)}"
                )
            output = (
                output_path.read_text(encoding="utf-8")
                if output_path.exists()
                else completed.stdout
            )
            cetz_source = _extract_cetz_canvas_body(output)
            _validate_cetz_body(cetz_source)
            return CetzConversionResult(cetz_source=cetz_source, transcript=transcript)


def _extract_cetz_canvas_body(output: str) -> str:
    text = str(output or "").strip()
    if not text:
        raise ContractError("Tylax produced empty CeTZ output")
    marker = "#cetz.canvas"
    start = text.find(marker)
    if start < 0:
        return _strip_typst_imports(text)

    open_index = text.find("{", start + len(marker))
    if open_index < 0:
        raise ContractError("Tylax output contains #cetz.canvas without a body")
    close_index = _matching_brace_index(text, open_index)
    if close_index < 0:
        raise ContractError("Tylax output contains an unterminated #cetz.canvas body")
    return text[open_index + 1 : close_index].strip()


def _matching_brace_index(text: str, open_index: int) -> int:
    depth = 0
    in_string = False
    escaped = False
    for index in range(open_index, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _strip_typst_imports(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("#import"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _validate_cetz_body(cetz_source: str) -> None:
    if not cetz_source.strip():
        raise ContractError("Tylax produced empty CeTZ body")
    if any(
        token in cetz_source
        for token in ("#import", "#cetz.canvas", "\\begin{tikzpicture}", "\\draw")
    ):
        raise ContractError("Tylax output was not normalized to a CeTZ canvas body")


def _list_files(workdir: Path) -> list[str]:
    return sorted(path.name for path in workdir.iterdir() if path.is_file())


def _tail(value: str, limit: int = 2000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[-limit:]
