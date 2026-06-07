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
        if "allow_visuals" in data or "allow_diagram" in data:
            raise ContractError(
                "`allow_visuals` and `allow_diagram` are no longer supported by this text API"
            )
        subject = _subject_string(data)
        return cls(
            level=_required_string(data, "level"),
            topic=_required_string(data, "topic"),
            subject=subject,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "level": self.level,
            "topic": self.topic,
            "subject": self.subject,
        }


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeneratedProblem":
        return cls(
            problem=_required_string(data, "problem"),
            solution=_required_string(data, "solution"),
        )

    def to_success_response(self) -> dict[str, Any]:
        return {
            "status": "success",
            "problem": self.problem,
            "solution": self.solution,
        }


@dataclass(frozen=True)
class CriticalValidationResult:
    passed: bool
    solution_accurate: bool
    issues: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CriticalValidationResult":
        passed = data.get("passed")
        solution_accurate = data.get("solution_accurate")
        issues = data.get("issues", [])

        if not isinstance(passed, bool):
            raise ContractError("`passed` must be a boolean")
        if not isinstance(solution_accurate, bool):
            raise ContractError("`solution_accurate` must be a boolean")
        if not isinstance(issues, list) or not all(
            isinstance(issue, str) for issue in issues
        ):
            raise ContractError("`issues` must be an array of strings")

        return cls(
            passed=passed,
            solution_accurate=solution_accurate,
            issues=tuple(issue for issue in issues if issue.strip()),
        )


@dataclass(frozen=True)
class ComplianceValidationResult:
    passed: bool
    curriculum_compliant: bool
    issues: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComplianceValidationResult":
        passed = data.get("passed")
        curriculum_compliant = data.get("curriculum_compliant")
        issues = data.get("issues", [])

        if not isinstance(passed, bool):
            raise ContractError("`passed` must be a boolean")
        if not isinstance(curriculum_compliant, bool):
            raise ContractError("`curriculum_compliant` must be a boolean")
        if not isinstance(issues, list) or not all(
            isinstance(issue, str) for issue in issues
        ):
            raise ContractError("`issues` must be an array of strings")

        return cls(
            passed=passed,
            curriculum_compliant=curriculum_compliant,
            issues=tuple(issue for issue in issues if issue.strip()),
        )


@dataclass(frozen=True)
class AdjudicationResult:
    decision: str
    reason: str
    repair_focus: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AdjudicationResult":
        decision = data.get("decision")
        reason = data.get("reason")
        repair_focus = data.get("repair_focus", "")

        allowed_decisions = {
            "repair_required",
            "validator_false_positive",
            "clarify_definition_required",
        }
        if decision not in allowed_decisions:
            raise ContractError(
                "`decision` must be `repair_required`, "
                "`validator_false_positive`, or `clarify_definition_required`"
            )
        if not isinstance(reason, str) or not reason.strip():
            raise ContractError("`reason` must be a non-empty string")
        if not isinstance(repair_focus, str):
            raise ContractError("`repair_focus` must be a string")

        return cls(
            decision=decision,
            reason=reason.strip(),
            repair_focus=repair_focus.strip(),
        )


@dataclass(frozen=True)
class ValidationResult:
    passed: bool
    solution_accurate: bool
    curriculum_compliant: bool
    issues: tuple[str, ...]

    def to_failure_response(self) -> dict[str, Any]:
        return {
            "status": "failed",
            "reason": "validator_rejected",
            "solution_accurate": self.solution_accurate,
            "curriculum_compliant": self.curriculum_compliant,
            "issues": list(self.issues),
        }
