"""Generator and validator agent orchestration."""

from __future__ import annotations

import json
import sys
from typing import Protocol

from .diagrams import render_diagram
from .json_utils import parse_json_object
from .models import (
    ComplianceValidationResult,
    ContractError,
    CriticalValidationResult,
    GenerateRequest,
    GeneratedProblem,
    ValidationResult,
)


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant content for the provided chat messages."""


class TraceWriter(Protocol):
    def write(self, text: str) -> object:
        """Write trace text."""

    def flush(self) -> object:
        """Flush trace text."""


class InteractionTracer:
    def __init__(self, enabled: bool = False, writer: TraceWriter | None = None) -> None:
        self.enabled = enabled
        self.writer = writer or sys.stdout

    def log_request(self, agent_name: str, messages: list[dict[str, str]]) -> None:
        if not self.enabled:
            return
        self._write(f"\n--- {agent_name} request ---")
        for index, message in enumerate(messages, start=1):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            self._write(f"[{index}] role={role}\n{content}")

    def log_response(self, agent_name: str, content: str) -> None:
        if not self.enabled:
            return
        self._write(f"--- {agent_name} response ---\n{content}")

    def log_validation_result(
        self,
        validator_name: str,
        passed: bool,
        issues: tuple[str, ...],
        extra: str,
    ) -> None:
        if not self.enabled:
            return
        self._write(
            f"--- {validator_name} decision ---\n"
            f"passed={passed}, {extra}, issues={list(issues)}"
        )

    def log_attempt(self, attempt: int, request: GenerateRequest) -> None:
        if not self.enabled:
            return
        self._write(
            "\n=== Generation attempt "
            f"{attempt}: subject={request.subject}, level={request.level}, topic={request.topic} ==="
        )

    def log_final_result(self, result: dict[str, object]) -> None:
        if not self.enabled:
            return
        self._write(f"=== Final result: status={result.get('status')} ===\n")

    def _write(self, text: str) -> None:
        self.writer.write(f"{text}\n")
        self.writer.flush()


SUBJECT_INSTRUCTIONS = {
    "mathematics": {
        "generator_role": "an independent mathematics exam author",
        "subject_name": "mathematics",
        "curriculum_scope": "Japanese high school mathematics curriculum",
        "accuracy_scope": "formulas, calculations, logical steps, and existence of a valid solution",
        "constraints": (
            "Mathematical expressions must be written in LaTeX using $...$ or $$...$$."
        ),
        "rejection_scope": (
            "mathematical flaws, insufficient conditions, redundant contradictory "
            "conditions, or out-of-scope methods"
        ),
    },
    "physics": {
        "generator_role": "an independent physics exam author",
        "subject_name": "physics",
        "curriculum_scope": "Japanese high school physics curriculum guidelines",
        "accuracy_scope": (
            "physical laws, units, formulas, numerical calculations, assumptions, "
            "and existence of a valid solution"
        ),
        "constraints": (
            "Do not output raw diagrams, figures, image references, or SVG. If "
            "allow_diagram is false, the problem must be fully text-only and must "
            "not require a diagram to solve. If allow_diagram is true, you may "
            "include a structured diagram_spec object when it improves clarity. "
            "Mathematical and physical expressions must be written in LaTeX using "
            "$...$ or $$...$$."
        ),
        "rejection_scope": (
            "physics flaws, incorrect units, missing assumptions, insufficient "
            "conditions, diagram-dependent wording, redundant contradictory "
            "conditions, or out-of-scope methods"
        ),
    },
    "chemistry": {
        "generator_role": "an independent chemistry exam author",
        "subject_name": "chemistry",
        "curriculum_scope": "Japanese high school chemistry curriculum guidelines",
        "accuracy_scope": (
            "chemical concepts, formulas, reaction equations, stoichiometry, units, "
            "numerical calculations, assumptions, and existence of a valid solution"
        ),
        "constraints": (
            "Use only text; do not generate diagrams, structural drawings, image "
            "references, or questions that require a visual diagram to solve. "
            "Describe any experimental setup, molecular structure, or reaction "
            "conditions completely in words. Chemical formulas, reaction equations, "
            "and mathematical expressions must be written in text or LaTeX using "
            "$...$ or $$...$$."
        ),
        "rejection_scope": (
            "chemistry flaws, incorrect chemical formulas or reaction equations, "
            "incorrect units, missing assumptions, insufficient conditions, "
            "diagram-dependent wording, redundant contradictory conditions, or "
            "out-of-scope methods"
        ),
    },
}


def _subject_instruction(subject: str) -> dict[str, str]:
    return SUBJECT_INSTRUCTIONS[subject]


def generator_system_prompt(subject: str) -> str:
    instruction = _subject_instruction(subject)
    return f"""You are Generator, {instruction["generator_role"]}.
Create one original Japanese university entrance exam {instruction["subject_name"]} question.
Use DeepSeek thinking internally, but output only strict JSON.
Write both the problem and solution in Japanese.
The problem must be solvable within the {instruction["curriculum_scope"]}.
{instruction["constraints"]}
Return exactly this JSON shape:
{{"problem":"...","solution":"...","diagram_spec":null}}
If the user sets allow_diagram to true for physics and a diagram would make the problem clearer, include a structured diagram_spec object instead of null. Do not output raw SVG. Supported diagram_spec kinds are "inclined_plane", "force_diagram", and "circuit". If no diagram is needed, use null."""


def critical_validator_system_prompt(subject: str) -> str:
    instruction = _subject_instruction(subject)
    return f"""You are CriticalValidator, an independent {instruction["subject_name"]} correctness reviewer.
Your only job is to find genuine errors or inconsistencies in the problem, solution, and diagram_spec.
Focus on: {instruction["accuracy_scope"]}.
Reject if there is any calculation error, invalid equation, incorrect unit, logical gap, self-contradiction, missing condition, inconsistent condition, mismatch between the problem and solution, or diagram/text inconsistency.
Ignore elegance, difficulty calibration, Japanese wording quality, and curriculum scope unless they directly create a correctness problem.
Reject problems with {instruction["rejection_scope"]}.
Use DeepSeek thinking internally, but output only strict JSON.
Return exactly this JSON shape:
{{"passed":true|false,"solution_accurate":true|false,"issues":["..."]}}"""


def compliance_validator_system_prompt(subject: str) -> str:
    instruction = _subject_instruction(subject)
    return f"""You are ComplianceValidator, an independent {instruction["subject_name"]} policy reviewer.
Assume the problem and solution may be scientifically or mathematically correct. Your job is to decide whether they are acceptable for this system.
Evaluate this mandatory checklist:
1. Curriculum Compliance: solvable without knowledge beyond the {instruction["curriculum_scope"]}.
2. Japanese Output: the final problem and solution text are written in Japanese, with only mathematical notation, symbols, variable names, units, and standard terms left as needed.
3. Subject and Topic Alignment: the problem matches the requested subject and topic.
4. Diagram Policy: for physics and chemistry, reject any problem that requires a diagram, figure, structural drawing, image, or visual reference unless a sufficient diagram_spec is provided and diagrams are allowed.
Before deciding, extract the required knowledge, concepts, laws, formulas, and mathematical methods used in the problem and solution. Judge whether each item is taught within the subject curriculum and prerequisite Japanese high school mathematics. Reject if solving requires any item outside that scope, even when the final answer is correct.
High difficulty means harder combinations of high-school methods, not university-level subject knowledge or mathematics.
Use DeepSeek thinking internally, but output only strict JSON.
Return exactly this JSON shape:
{{"passed":true|false,"curriculum_compliant":true|false,"issues":["..."]}}"""


class GeneratorAgent:
    def __init__(
        self, client: ChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def generate(self, request: GenerateRequest) -> GeneratedProblem:
        messages = [
            {"role": "system", "content": generator_system_prompt(request.subject)},
            {
                "role": "user",
                "content": (
                        f"Generate one entrance exam {request.subject} question with solution.\n"
                        f"Subject: {request.subject}\n"
                        f"Difficulty level: {request.level}\n"
                        f"Topic: {request.topic}\n"
                        f"allow_diagram: {str(request.allow_diagram).lower()}"
                ),
            },
        ]
        self.tracer.log_request("Generator", messages)
        content = self.client.chat(messages)
        self.tracer.log_response("Generator", content)
        return GeneratedProblem.from_dict(parse_json_object(content))


def _validation_payload(
    request: GenerateRequest, generated: GeneratedProblem
) -> str:
    return json.dumps(
        {
            "subject": request.subject,
            "level": request.level,
            "topic": request.topic,
            "allow_diagram": request.allow_diagram,
            "problem": generated.problem,
            "solution": generated.solution,
            "diagram_spec": generated.diagram_spec,
        },
        ensure_ascii=False,
    )


class CriticalValidatorAgent:
    def __init__(
        self, client: ChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def validate(
        self, request: GenerateRequest, generated: GeneratedProblem
    ) -> CriticalValidationResult:
        messages = [
            {"role": "system", "content": critical_validator_system_prompt(request.subject)},
            {"role": "user", "content": _validation_payload(request, generated)},
        ]
        self.tracer.log_request("CriticalValidator", messages)
        content = self.client.chat(messages)
        self.tracer.log_response("CriticalValidator", content)
        validation = CriticalValidationResult.from_dict(parse_json_object(content))
        self.tracer.log_validation_result(
            "CriticalValidator",
            validation.passed,
            validation.issues,
            f"solution_accurate={validation.solution_accurate}",
        )
        return validation


class ComplianceValidatorAgent:
    def __init__(
        self, client: ChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def validate(
        self, request: GenerateRequest, generated: GeneratedProblem
    ) -> ComplianceValidationResult:
        messages = [
            {"role": "system", "content": compliance_validator_system_prompt(request.subject)},
            {"role": "user", "content": _validation_payload(request, generated)},
        ]
        self.tracer.log_request("ComplianceValidator", messages)
        content = self.client.chat(messages)
        self.tracer.log_response("ComplianceValidator", content)
        validation = ComplianceValidationResult.from_dict(parse_json_object(content))
        self.tracer.log_validation_result(
            "ComplianceValidator",
            validation.passed,
            validation.issues,
            f"curriculum_compliant={validation.curriculum_compliant}",
        )
        return validation


class MathGenerationService:
    def __init__(
        self,
        generator: GeneratorAgent,
        critical_validator: CriticalValidatorAgent,
        compliance_validator: ComplianceValidatorAgent,
        max_attempts: int = 2,
        tracer: InteractionTracer | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.generator = generator
        self.critical_validator = critical_validator
        self.compliance_validator = compliance_validator
        self.max_attempts = max_attempts
        self.tracer = tracer or InteractionTracer()

    def generate_verified(self, request: GenerateRequest) -> dict[str, object]:
        last_validation: ValidationResult | None = None

        for attempt in range(1, self.max_attempts + 1):
            self.tracer.log_attempt(attempt, request)
            generated = self.generator.generate(request)
            if generated.diagram_spec is not None and not request.allow_diagram:
                last_validation = ValidationResult(
                    passed=False,
                    solution_accurate=True,
                    curriculum_compliant=False,
                    issues=("diagram_spec was returned when allow_diagram was false",),
                )
                continue
            try:
                rendered_diagram = render_diagram(generated.diagram_spec)
            except ContractError as exc:
                last_validation = ValidationResult(
                    passed=False,
                    solution_accurate=True,
                    curriculum_compliant=False,
                    issues=(str(exc),),
                )
                continue
            critical_validation = self.critical_validator.validate(request, generated)
            if not critical_validation.passed:
                last_validation = ValidationResult(
                    passed=False,
                    solution_accurate=critical_validation.solution_accurate,
                    curriculum_compliant=True,
                    issues=critical_validation.issues,
                )
                continue

            compliance_validation = self.compliance_validator.validate(request, generated)
            if compliance_validation.passed:
                result = generated.to_success_response(rendered_diagram)
                self.tracer.log_final_result(result)
                return result
            last_validation = ValidationResult(
                passed=False,
                solution_accurate=True,
                curriculum_compliant=compliance_validation.curriculum_compliant,
                issues=compliance_validation.issues,
            )

        assert last_validation is not None
        result = last_validation.to_failure_response()
        self.tracer.log_final_result(result)
        return result
