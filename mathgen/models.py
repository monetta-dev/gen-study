"""Data contracts for the exam question generation API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ContractError(ValueError):
    """Raised when API or agent payloads do not match the expected contract."""


class ModelOutputError(RuntimeError):
    """Raised when an LLM response cannot satisfy an agent output contract."""


def _required_string(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"`{field}` must be a non-empty string")
    return value.strip()


def _optional_string(data: dict[str, Any], field: str, default: str = "") -> str:
    value = data.get(field, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ContractError(f"`{field}` must be a string")
    return value.strip()


def _string_list(data: dict[str, Any], field: str) -> tuple[str, ...]:
    value = data.get(field, [])
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ContractError(f"`{field}` must be an array of strings")
    return tuple(item.strip() for item in value if item.strip())


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


@dataclass(frozen=True)
class VisualGenerateRequest:
    problem: str
    solution: str
    subject: str = "mathematics"
    level: str = ""
    topic: str = ""
    placement_hint: str = "auto"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualGenerateRequest":
        return cls(
            problem=_required_string(data, "problem"),
            solution=_required_string(data, "solution"),
            subject=_subject_string(data),
            level=_optional_string(data, "level"),
            topic=_optional_string(data, "topic"),
            placement_hint=_optional_string(data, "placement_hint", "auto") or "auto",
        )


@dataclass(frozen=True)
class VisualInstruction:
    needed: bool
    visual_id: str
    placement: str
    purpose: str
    prompt: str
    must_include: tuple[str, ...]
    must_not_include: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualInstruction":
        needed = data.get("needed")
        if not isinstance(needed, bool):
            raise ContractError("`needed` must be a boolean")
        placement = _optional_string(data, "placement", "none") or "none"
        purpose = _optional_string(data, "purpose", "none") or "none"
        prompt = _optional_string(data, "prompt")
        if needed and not prompt:
            raise ContractError("`prompt` must be a non-empty string when `needed` is true")
        visual_id = _optional_string(data, "id") or _optional_string(data, "visual_id")
        return cls(
            needed=needed,
            visual_id=visual_id,
            placement=placement,
            purpose=purpose,
            prompt=prompt,
            must_include=_string_list(data, "must_include"),
            must_not_include=_string_list(data, "must_not_include"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "needed": self.needed,
            "id": self.visual_id,
            "placement": self.placement,
            "purpose": self.purpose,
            "prompt": self.prompt,
            "must_include": list(self.must_include),
            "must_not_include": list(self.must_not_include),
        }


@dataclass(frozen=True)
class VisualInstructionSet:
    needed: bool
    visuals: tuple[VisualInstruction, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualInstructionSet":
        if "visuals" not in data:
            instruction = VisualInstruction.from_dict(data)
            return cls(
                needed=instruction.needed,
                visuals=(_with_default_visual_id(instruction, 1),)
                if instruction.needed
                else (),
            )

        needed = data.get("needed")
        if not isinstance(needed, bool):
            raise ContractError("`needed` must be a boolean")
        raw_visuals = data.get("visuals", [])
        if not isinstance(raw_visuals, list):
            raise ContractError("`visuals` must be an array")
        visuals = tuple(
            _with_default_visual_id(VisualInstruction.from_dict(_visual_item(item)), index)
            for index, item in enumerate(raw_visuals, start=1)
            if isinstance(item, dict)
        )
        if needed and not visuals:
            raise ContractError("`visuals` must contain at least one item when `needed` is true")
        return cls(needed=needed, visuals=visuals if needed else ())

    def to_dict(self) -> dict[str, Any]:
        return {
            "needed": self.needed,
            "visuals": [visual.to_dict() for visual in self.visuals],
        }


def _with_default_visual_id(
    instruction: VisualInstruction, index: int
) -> VisualInstruction:
    if instruction.visual_id:
        return instruction
    return VisualInstruction(
        needed=instruction.needed,
        visual_id=f"visual_{index}",
        placement=instruction.placement,
        purpose=instruction.purpose,
        prompt=instruction.prompt,
        must_include=instruction.must_include,
        must_not_include=instruction.must_not_include,
    )


def _visual_item(data: dict[str, Any]) -> dict[str, Any]:
    if "needed" in data:
        return data
    return {**data, "needed": True}


@dataclass(frozen=True)
class VisualFinding:
    issue: str
    evidence: str
    severity: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualFinding":
        return cls(
            issue=_required_string(data, "issue"),
            evidence=_required_string(data, "evidence"),
            severity=_optional_string(data, "severity", "major") or "major",
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "issue": self.issue,
            "evidence": self.evidence,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class VisualValidationResult:
    status: str
    confidence: float
    findings: tuple[VisualFinding, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VisualValidationResult":
        status = data.get("status")
        if status not in {"pass", "repair_required", "uncertain"}:
            raise ContractError(
                "`status` must be `pass`, `repair_required`, or `uncertain`"
            )
        confidence = data.get("confidence", 0.0)
        if not isinstance(confidence, (int, float)):
            raise ContractError("`confidence` must be a number")
        raw_findings = data.get("findings", [])
        if not isinstance(raw_findings, list):
            raise ContractError("`findings` must be an array")
        findings = tuple(
            VisualFinding.from_dict(item)
            for item in raw_findings
            if isinstance(item, dict)
        )
        return cls(
            status=status,
            confidence=max(0.0, min(1.0, float(confidence))),
            findings=findings,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "findings": [finding.to_dict() for finding in self.findings],
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
