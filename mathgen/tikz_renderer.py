"""Agentic TikZ compilation and PNG rendering."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .agents import InteractionTracer, chat_json
from .models import ContractError


class CompileChatClient(Protocol):
    def chat(self, messages: list[dict[str, object]]) -> str:
        """Return assistant content for compile actions."""


@dataclass(frozen=True)
class CompileAction:
    action: str
    argv: tuple[str, ...]
    reason: str
    issues: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "CompileAction":
        action = data.get("action")
        if action not in {"run_command", "finish", "request_tikz_repair"}:
            raise ContractError(
                "`action` must be `run_command`, `finish`, or `request_tikz_repair`"
            )
        raw_argv = data.get("argv", [])
        if raw_argv is None:
            raw_argv = []
        if not isinstance(raw_argv, list) or not all(
            isinstance(item, str) and item.strip() for item in raw_argv
        ):
            raise ContractError("`argv` must be an array of non-empty strings")
        raw_issues = data.get("issues", [])
        if raw_issues is None:
            raw_issues = []
        if not isinstance(raw_issues, list) or not all(
            isinstance(item, str) for item in raw_issues
        ):
            raise ContractError("`issues` must be an array of strings")
        if action == "run_command" and not raw_argv:
            raise ContractError("`argv` is required for `run_command`")
        if action == "request_tikz_repair" and not raw_issues:
            raise ContractError("`issues` is required for `request_tikz_repair`")
        reason = data.get("reason", "")
        if reason is None:
            reason = ""
        if not isinstance(reason, str):
            raise ContractError("`reason` must be a string")
        return cls(
            action=action,
            argv=tuple(item.strip() for item in raw_argv),
            reason=reason.strip(),
            issues=tuple(item.strip() for item in raw_issues if item.strip()),
        )


@dataclass(frozen=True)
class CommandObservation:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    log_excerpt: str
    files: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "argv": list(self.argv),
            "returncode": self.returncode,
            "stdout": _tail(self.stdout),
            "stderr": _tail(self.stderr),
            "log_excerpt": self.log_excerpt,
            "files": list(self.files),
        }


@dataclass(frozen=True)
class TikZRenderResult:
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


def tikz_compile_system_prompt() -> str:
    return """You are TikZCompileAgent, an agentic LaTeX compilation operator.
Your job is to compile an already generated TikZ figure to PDF inside a sandbox.
You can choose commands, inspect command outputs, logs, and generated files, then decide the next command.
Do not rewrite the diagram. If the source itself is genuinely invalid, request TikZ repair.
Prefer LuaLaTeX with luatexja for Japanese labels. Try another available engine only when logs justify it.
Use only allowed commands and only the provided files.
Return strict JSON only.
Return one of these shapes:
{"action":"run_command","argv":["lualatex","-interaction=nonstopmode","-halt-on-error","main-lualatex.tex"],"reason":"..."}
{"action":"finish","reason":"A PDF was produced."}
{"action":"request_tikz_repair","issues":["..."],"reason":"..."}"""


class TikZCompileAgent:
    def __init__(
        self, client: CompileChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def next_action(self, payload: dict[str, object]) -> CompileAction:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": tikz_compile_system_prompt()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        return chat_json(
            "TikZCompileAgent",
            self.client,
            self.tracer,
            messages,
            CompileAction.from_dict,
            (
                '{"action":"run_command",'
                '"argv":["lualatex","-interaction=nonstopmode","-halt-on-error",'
                '"main-lualatex.tex"],"reason":"..."}'
            ),
        )


class TikZRenderer:
    def __init__(
        self,
        compile_agent: TikZCompileAgent,
        timeout_seconds: float = 30.0,
        max_actions: int = 5,
        tracer: InteractionTracer | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_actions <= 0:
            raise ValueError("max_actions must be positive")
        self.compile_agent = compile_agent
        self.timeout_seconds = timeout_seconds
        self.max_actions = max_actions
        self.tracer = tracer or InteractionTracer()

    def render_png(self, tikz_source: str) -> TikZRenderResult:
        with tempfile.TemporaryDirectory(prefix="gen-study-tikz-") as tmpdir:
            workdir = Path(tmpdir)
            _write_compile_documents(workdir, tikz_source)
            observations: list[dict[str, object]] = []
            transcript: list[dict[str, object]] = []
            available_commands = _available_commands()

            for _ in range(self.max_actions):
                payload = {
                    "goal": "Compile the TikZ source to a PDF.",
                    "available_commands": available_commands,
                    "files": _list_files(workdir),
                    "documents": [
                        "main-lualatex.tex",
                        "main-xelatex.tex",
                        "main-pdflatex.tex",
                    ],
                    "tikz_source": tikz_source,
                    "observations": observations,
                }
                action = self.compile_agent.next_action(payload)
                transcript.append(_action_record(action))

                if action.action == "request_tikz_repair":
                    return TikZRenderResult(
                        status="repair_required",
                        png_base64="",
                        mime_type="image/png",
                        compile_transcript=tuple(transcript),
                        command_count=_command_count(transcript),
                        engine="",
                        issues=action.issues,
                    )

                if action.action == "finish":
                    pdf_path = _newest_pdf(workdir)
                    if pdf_path is None:
                        observations.append(
                            {
                                "error": "finish requested but no PDF file exists",
                                "files": _list_files(workdir),
                            }
                        )
                        continue
                    png_base64 = _pdf_to_png_base64(
                        pdf_path, workdir, self.timeout_seconds
                    )
                    return TikZRenderResult(
                        status="compiled",
                        png_base64=png_base64,
                        mime_type="image/png",
                        compile_transcript=tuple(transcript),
                        command_count=_command_count(transcript),
                        engine=_engine_from_transcript(transcript),
                    )

                observation = self._run_allowed_command(action.argv, workdir)
                observations.append(observation.to_dict())
                transcript.append({"observation": observation.to_dict()})
                pdf_path = _newest_pdf(workdir)
                if pdf_path is not None:
                    png_base64 = _pdf_to_png_base64(
                        pdf_path, workdir, self.timeout_seconds
                    )
                    return TikZRenderResult(
                        status="compiled",
                        png_base64=png_base64,
                        mime_type="image/png",
                        compile_transcript=tuple(transcript),
                        command_count=_command_count(transcript),
                        engine=_engine_from_transcript(transcript),
                    )

            return TikZRenderResult(
                status="failed",
                png_base64="",
                mime_type="image/png",
                compile_transcript=tuple(transcript),
                command_count=_command_count(transcript),
                engine=_engine_from_transcript(transcript),
                issues=("TikZ compilation did not produce a PDF within the action limit.",),
            )

    def _run_allowed_command(
        self, argv: tuple[str, ...], workdir: Path
    ) -> CommandObservation:
        try:
            _validate_argv(argv)
        except ContractError as exc:
            return CommandObservation(
                argv=argv,
                returncode=126,
                stdout="",
                stderr=str(exc),
                log_excerpt=_collect_log_excerpt(workdir),
                files=tuple(_list_files(workdir)),
            )
        try:
            completed = subprocess.run(
                list(argv),
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
            return CommandObservation(
                argv=argv,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                log_excerpt=_collect_log_excerpt(workdir),
                files=tuple(_list_files(workdir)),
            )
        except FileNotFoundError:
            return CommandObservation(
                argv=argv,
                returncode=127,
                stdout="",
                stderr=f"Command not found: {argv[0]}",
                log_excerpt=_collect_log_excerpt(workdir),
                files=tuple(_list_files(workdir)),
            )
        except subprocess.TimeoutExpired as exc:
            return CommandObservation(
                argv=argv,
                returncode=124,
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + "\nCommand timed out.",
                log_excerpt=_collect_log_excerpt(workdir),
                files=tuple(_list_files(workdir)),
            )


def _write_compile_documents(workdir: Path, tikz_source: str) -> None:
    wrappers = {
        "main-lualatex.tex": _latex_document(
            [
                "\\documentclass[tikz,border=3pt]{standalone}",
                "\\usepackage{luatexja}",
                "\\usepackage{amsmath,amssymb}",
                "\\usepackage{tikz}",
                "\\usetikzlibrary{arrows.meta,calc,decorations.pathmorphing,intersections,patterns,positioning}",
            ],
            tikz_source,
        ),
        "main-xelatex.tex": _latex_document(
            [
                "\\documentclass[tikz,border=3pt]{standalone}",
                "\\usepackage{fontspec}",
                "\\usepackage{xeCJK}",
                "\\usepackage{amsmath,amssymb}",
                "\\usepackage{tikz}",
                "\\usetikzlibrary{arrows.meta,calc,decorations.pathmorphing,intersections,patterns,positioning}",
            ],
            tikz_source,
        ),
        "main-pdflatex.tex": _latex_document(
            [
                "\\documentclass[tikz,border=3pt]{standalone}",
                "\\usepackage[utf8]{inputenc}",
                "\\usepackage{amsmath,amssymb}",
                "\\usepackage{tikz}",
                "\\usetikzlibrary{arrows.meta,calc,decorations.pathmorphing,intersections,patterns,positioning}",
            ],
            tikz_source,
        ),
    }
    for filename, content in wrappers.items():
        (workdir / filename).write_text(content, encoding="utf-8")


def _latex_document(preamble: list[str], tikz_source: str) -> str:
    return "\n".join(
        [
            *preamble,
            "\\begin{document}",
            tikz_source,
            "\\end{document}",
            "",
        ]
    )


def _available_commands() -> list[str]:
    candidates = [
        "lualatex",
        "xelatex",
        "pdflatex",
        "latexmk",
        "platex",
        "dvipdfmx",
    ]
    return [command for command in candidates if shutil.which(command)]


def _validate_argv(argv: tuple[str, ...]) -> None:
    allowed = {
        "lualatex",
        "xelatex",
        "pdflatex",
        "latexmk",
        "platex",
        "dvipdfmx",
    }
    command = argv[0]
    if command not in allowed:
        raise ContractError(f"`{command}` is not an allowed compile command")
    for item in argv:
        if any(token in item for token in (";", "&&", "||", "|", "`", "$(", ">", "<")):
            raise ContractError("compile command arguments must not contain shell syntax")
        if os.path.isabs(item):
            raise ContractError("compile command arguments must use relative paths")


def _pdf_to_png_base64(pdf_path: Path, workdir: Path, timeout_seconds: float) -> str:
    if not shutil.which("pdftoppm"):
        raise ContractError("PDF compiled, but `pdftoppm` is not installed")
    output_prefix = workdir / "rendered"
    completed = subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-singlefile",
            "-r",
            "180",
            str(pdf_path),
            str(output_prefix),
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    png_path = workdir / "rendered.png"
    if completed.returncode != 0 or not png_path.exists():
        raise ContractError(
            "PDF compiled, but PNG conversion failed: "
            f"{_tail(completed.stderr or completed.stdout)}"
        )
    return base64.b64encode(png_path.read_bytes()).decode("ascii")


def _newest_pdf(workdir: Path) -> Path | None:
    pdfs = sorted(workdir.glob("*.pdf"), key=lambda path: path.stat().st_mtime)
    return pdfs[-1] if pdfs else None


def _list_files(workdir: Path) -> list[str]:
    return sorted(path.name for path in workdir.iterdir() if path.is_file())


def _collect_log_excerpt(workdir: Path) -> str:
    logs = sorted(workdir.glob("*.log"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return ""
    return _tail(logs[-1].read_text(encoding="utf-8", errors="replace"), 4000)


def _action_record(action: CompileAction) -> dict[str, object]:
    record: dict[str, object] = {
        "action": action.action,
        "reason": action.reason,
    }
    if action.argv:
        record["argv"] = list(action.argv)
    if action.issues:
        record["issues"] = list(action.issues)
    return record


def _command_count(transcript: list[dict[str, object]] | tuple[dict[str, object], ...]) -> int:
    return sum(1 for item in transcript if item.get("action") == "run_command")


def _engine_from_transcript(
    transcript: list[dict[str, object]] | tuple[dict[str, object], ...]
) -> str:
    for item in transcript:
        argv = item.get("argv")
        if isinstance(argv, list) and argv and isinstance(argv[0], str):
            return argv[0]
    return ""


def _tail(value: str, limit: int = 2000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[-limit:]
