"""Helpers for extracting strict JSON from LLM responses."""

from __future__ import annotations

import json
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
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ContractError("Model response did not contain a JSON object") from exc
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as nested_exc:
            raise ContractError("Model response JSON was invalid") from nested_exc

    if not isinstance(parsed, dict):
        raise ContractError("Model response must be a JSON object")
    return parsed
