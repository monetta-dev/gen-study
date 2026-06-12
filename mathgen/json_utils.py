"""Helpers for extracting strict JSON from LLM responses."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from .models import ContractError


def parse_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating fenced JSON wrappers from model output."""
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = _parse_json_like(text)
    except (json.JSONDecodeError, ValueError, SyntaxError) as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ContractError("Model response did not contain a JSON object") from exc
        try:
            parsed = _parse_json_like(text[start : end + 1])
        except (json.JSONDecodeError, ValueError, SyntaxError) as nested_exc:
            raise ContractError("Model response JSON was invalid") from nested_exc

    if not isinstance(parsed, dict):
        raise ContractError("Model response must be a JSON object")
    return parsed


def _parse_json_like(text: str) -> Any:
    candidates = _json_candidates(text)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc

    for candidate in candidates:
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError) as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise ValueError("No JSON candidate")


def _json_candidates(text: str) -> tuple[str, ...]:
    no_trailing_commas = re.sub(r",\s*([}\]])", r"\1", text)
    escaped_controls = _escape_control_chars_in_strings(text)
    escaped_without_trailing = _escape_control_chars_in_strings(no_trailing_commas)

    candidates = []
    for candidate in (
        text,
        no_trailing_commas,
        escaped_controls,
        escaped_without_trailing,
    ):
        if candidate not in candidates:
            candidates.append(candidate)
    return tuple(candidates)


def _escape_control_chars_in_strings(text: str) -> str:
    output = []
    in_string = False
    escaped = False

    for char in text:
        if escaped:
            output.append(char)
            escaped = False
            continue
        if char == "\\" and in_string:
            output.append(char)
            escaped = True
            continue
        if char == '"':
            output.append(char)
            in_string = not in_string
            continue
        if in_string and ord(char) < 32:
            output.append(_escaped_control_char(char))
            continue
        output.append(char)

    return "".join(output)


def _escaped_control_char(char: str) -> str:
    if char == "\n":
        return "\\n"
    if char == "\r":
        return "\\r"
    if char == "\t":
        return "\\t"
    return f"\\u{ord(char):04x}"
