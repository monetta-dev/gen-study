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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GenerateRequest":
        return cls(
            level=_required_string(data, "level"),
            topic=_required_string(data, "topic"),
            subject=_subject_string(data),
        )

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "topic": self.topic, "subject": self.subject}


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
    }
    subject = aliases.get(normalized)
    if subject is None:
        raise ContractError("`subject` must be `mathematics` or `physics`")
    return subject


@dataclass(frozen=True)
class GeneratedProblem:
    problem: str
    solution: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratedProblem":
        return cls(
            problem=_required_string(data, "problem"),
            solution=_required_string(data, "solution"),
        )

    def to_success_response(self) -> dict[str, str]:
        return {
            "status": "success",
            "problem": self.problem,
            "solution": self.solution,
        }


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
