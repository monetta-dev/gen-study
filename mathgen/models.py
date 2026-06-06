"""Data contracts for the exam question generation API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ContractError(ValueError):
    """Raised when API or agent payloads do not match the expected contract."""


def _required_string(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"`{field}` must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class GenerateRequest:
    level: str
    topic: str
    subject: str = "mathematics"
    allow_diagram: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerateRequest":
        subject = _subject_string(data)
        allow_diagram = _optional_bool(data, "allow_diagram")
        if allow_diagram and subject != "physics":
            raise ContractError("`allow_diagram` is currently supported only for physics")
        return cls(
            level=_required_string(data, "level"),
            topic=_required_string(data, "topic"),
            subject=subject,
            allow_diagram=allow_diagram,
        )

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "level": self.level,
            "topic": self.topic,
            "subject": self.subject,
            "allow_diagram": self.allow_diagram,
        }


def _optional_bool(data: dict[str, Any], field: str) -> bool:
    value = data.get(field, False)
    if not isinstance(value, bool):
        raise ContractError(f"`{field}` must be a boolean")
    return value


def _subject_string(data: dict[str, Any]) -> str:
    raw_value = data.get("subject", "mathematics")
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ContractError("`subject` must be a non-empty string")

    normalized = raw_value.strip().lower()
    aliases = {
        "math": "mathematics",
        "mathematics": "mathematics",
        "数学": "mathematics",
        "physics": "physics",
        "物理": "physics",
        "chem": "chemistry",
        "chemistry": "chemistry",
        "化学": "chemistry",
    }
    subject = aliases.get(normalized)
    if subject is None:
        raise ContractError("`subject` must be `mathematics`, `physics`, or `chemistry`")
    return subject


@dataclass(frozen=True)
class GeneratedProblem:
    problem: str
    solution: str
    diagram_spec: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratedProblem":
        return cls(
            problem=_required_string(data, "problem"),
            solution=_required_string(data, "solution"),
            diagram_spec=_optional_diagram_spec(data),
        )

    def to_success_response(
        self, diagram: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "status": "success",
            "problem": self.problem,
            "solution": self.solution,
        }
        if diagram is not None:
            response["diagram"] = diagram
        return response


def _optional_diagram_spec(data: dict[str, Any]) -> dict[str, Any] | None:
    if "diagram_spec" not in data or data["diagram_spec"] is None:
        return None
    value = data["diagram_spec"]
    if not isinstance(value, dict):
        raise ContractError("`diagram_spec` must be an object or null")
    kind = value.get("kind")
    if not isinstance(kind, str) or not kind.strip():
        raise ContractError("`diagram_spec.kind` must be a non-empty string")
    return value


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    solution_accurate: bool
    curriculum_compliant: bool
    issues: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidationResult":
        passed = data.get("passed")
        solution_accurate = data.get("solution_accurate")
        curriculum_compliant = data.get("curriculum_compliant")
        issues = data.get("issues", [])

        if not isinstance(passed, bool):
            raise ContractError("`passed` must be a boolean")
        if not isinstance(solution_accurate, bool):
            raise ContractError("`solution_accurate` must be a boolean")
        if not isinstance(curriculum_compliant, bool):
            raise ContractError("`curriculum_compliant` must be a boolean")
        if not isinstance(issues, list) or not all(
            isinstance(issue, str) for issue in issues
        ):
            raise ContractError("`issues` must be an array of strings")

        return cls(
            passed=passed,
            solution_accurate=solution_accurate,
            curriculum_compliant=curriculum_compliant,
            issues=tuple(issue for issue in issues if issue.strip()),
        )

    def to_failure_response(self) -> dict[str, Any]:
        return {
            "status": "failed",
            "reason": "validator_rejected",
            "solution_accurate": self.solution_accurate,
            "curriculum_compliant": self.curriculum_compliant,
            "issues": list(self.issues),
        }
