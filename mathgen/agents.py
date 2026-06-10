"""Generator and validator agent orchestration."""

from __future__ import annotations

import json
import sys
from typing import Protocol

from .json_utils import parse_json_object
from .models import (
    AdjudicationResult,
    ComplianceValidationResult,
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

    def log_adjudication_result(self, result: AdjudicationResult) -> None:
        if not self.enabled:
            return
        self._write(
            "--- Adjudicator decision ---\n"
            f"decision={result.decision}, reason={result.reason}, "
            f"repair_focus={result.repair_focus}"
        )

    def log_attempt(self, attempt: int, request: GenerateRequest) -> None:
        if not self.enabled:
            return
        self._write(
            "\n=== Initial generation "
            f"{attempt}: subject={request.subject}, level={request.level}, topic={request.topic} ==="
        )

    def log_repair_attempt(
        self, attempt: int, validation_stage: str, issues: tuple[str, ...]
    ) -> None:
        if not self.enabled:
            return
        self._write(
            "\n=== Repair attempt "
            f"{attempt}: stage={validation_stage}, issues={list(issues)} ==="
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
            "Mathematical expressions must be written in LaTeX using $...$ or $$...$$. "
            "Use only text; do not generate diagrams, graphs, figures, image references, "
            "SVG, or questions that require visual assets to solve."
        ),
        "rejection_scope": (
            "mathematical flaws, insufficient conditions, redundant contradictory "
            "conditions, diagram-dependent wording, or out-of-scope methods"
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
            "Use only text; do not generate diagrams, graphs, figures, image references, "
            "SVG, or questions that require visual assets to solve. Describe setups, "
            "directions, forces, fields, circuits, and graphs completely in words. "
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
            "references, SVG, or questions that require visual assets to solve. "
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
Use internal reasoning if available, but output only strict JSON.
Write both the problem and solution in Japanese.
The problem must be solvable within the {instruction["curriculum_scope"]}.
{instruction["constraints"]}
Return exactly this JSON shape:
{{"problem":"...","solution":"..."}}"""


def critical_validator_system_prompt(subject: str) -> str:
    instruction = _subject_instruction(subject)
    return f"""You are CriticalValidator, an independent {instruction["subject_name"]} correctness reviewer.
Your only job is to find genuine errors or inconsistencies in the problem and solution.
Focus on: {instruction["accuracy_scope"]}.
Reject if there is any calculation error, invalid equation, incorrect unit, logical gap, self-contradiction, missing condition, inconsistent condition, or mismatch between the problem and solution.
Ignore elegance, difficulty calibration, Japanese wording quality, and curriculum scope unless they directly create a correctness problem.
Reject problems with {instruction["rejection_scope"]}.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{{"passed":true|false,"solution_accurate":true|false,"issues":["..."]}}"""


def compliance_validator_system_prompt(subject: str) -> str:
    instruction = _subject_instruction(subject)
    return f"""You are ComplianceValidator, an independent {instruction["subject_name"]} policy reviewer.
Assume the problem and solution may be scientifically or mathematically correct. Your job is to decide whether they are acceptable for this text-only system.
Evaluate this mandatory checklist:
1. Curriculum Compliance: solvable without knowledge beyond the {instruction["curriculum_scope"]}.
2. Japanese Output: the final problem and solution text are written in Japanese, with only mathematical notation, symbols, variable names, units, and standard terms left as needed.
3. Subject and Topic Alignment: the problem matches the requested subject and topic.
4. Level Alignment: the problem difficulty, complexity, and expected reasoning depth match the requested level. Reject if it is clearly too easy, too hard, or calibrated to a different exam level.
5. Text-only Policy: reject any problem that requires a diagram, graph, figure, structural drawing, image, or visual reference to solve.
Before deciding, extract the required knowledge, concepts, laws, formulas, and mathematical methods used in the problem and solution. Judge whether each item is taught within the subject curriculum and prerequisite Japanese high school mathematics. Reject if solving requires any item outside that scope, even when the final answer is correct.
High difficulty means harder combinations of high-school methods, not university-level subject knowledge or mathematics.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{{"passed":true|false,"curriculum_compliant":true|false,"issues":["..."]}}"""


def repair_system_prompt(subject: str) -> str:
    instruction = _subject_instruction(subject)
    return f"""You are RepairAgent, an independent {instruction["subject_name"]} revision specialist.
Revise the existing problem and solution using the validator feedback and adjudicator guidance.
Preserve the requested subject, topic, difficulty level, Japanese output, and overall intent.
Change only what is necessary to resolve the reported issues.
Do not introduce university-level subject knowledge or mathematics.
The repaired problem must remain solvable within the {instruction["curriculum_scope"]}.
{instruction["constraints"]}
Return exactly this JSON shape:
{{"problem":"...","solution":"..."}}"""


def adjudicator_system_prompt(subject: str) -> str:
    instruction = _subject_instruction(subject)
    return f"""You are Adjudicator, an independent reviewer of validator findings for {instruction["subject_name"]}.
Your job is not to solve or repair the problem.
Review the validator finding and decide whether the finding is valid.
Choose exactly one decision:
- repair_required: the validator finding is valid and the candidate should be repaired.
- validator_false_positive: the validator likely misunderstood, overreached, or made a high-probability false finding.
- clarify_definition_required: the problem or solution is ambiguous because a definition, convention, condition, or assumption needs clarification.
If clarification is required, repair_focus should describe the definition or condition to clarify.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{{"decision":"repair_required|validator_false_positive|clarify_definition_required","reason":"...","repair_focus":"..."}}"""


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
                    f"Topic: {request.topic}"
                ),
            },
        ]
        self.tracer.log_request("Generator", messages)
        content = self.client.chat(messages)
        self.tracer.log_response("Generator", content)
        return GeneratedProblem.from_dict(parse_json_object(content))


class RepairAgent:
    def __init__(
        self, client: ChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def repair(
        self,
        request: GenerateRequest,
        generated: GeneratedProblem,
        validation_stage: str,
        issues: tuple[str, ...],
        adjudication: AdjudicationResult,
    ) -> GeneratedProblem:
        messages = [
            {"role": "system", "content": repair_system_prompt(request.subject)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "subject": request.subject,
                        "level": request.level,
                        "topic": request.topic,
                        "validation_stage": validation_stage,
                        "validator_issues": list(issues),
                        "adjudicator_decision": adjudication.decision,
                        "adjudicator_reason": adjudication.reason,
                        "repair_focus": adjudication.repair_focus,
                        "problem": generated.problem,
                        "solution": generated.solution,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        self.tracer.log_request("RepairAgent", messages)
        content = self.client.chat(messages)
        self.tracer.log_response("RepairAgent", content)
        return GeneratedProblem.from_dict(parse_json_object(content))


class AdjudicatorAgent:
    def __init__(
        self, client: ChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def adjudicate(
        self,
        request: GenerateRequest,
        generated: GeneratedProblem,
        validation_stage: str,
        issues: tuple[str, ...],
    ) -> AdjudicationResult:
        messages = [
            {"role": "system", "content": adjudicator_system_prompt(request.subject)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "subject": request.subject,
                        "level": request.level,
                        "topic": request.topic,
                        "validation_stage": validation_stage,
                        "validator_issues": list(issues),
                        "problem": generated.problem,
                        "solution": generated.solution,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        self.tracer.log_request("Adjudicator", messages)
        content = self.client.chat(messages)
        self.tracer.log_response("Adjudicator", content)
        adjudication = AdjudicationResult.from_dict(parse_json_object(content))
        self.tracer.log_adjudication_result(adjudication)
        return adjudication


def _validation_payload(
    request: GenerateRequest, generated: GeneratedProblem
) -> str:
    return json.dumps(
        {
            "subject": request.subject,
            "level": request.level,
            "topic": request.topic,
            "problem": generated.problem,
            "solution": generated.solution,
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
        adjudicator: AdjudicatorAgent,
        repair_agent: RepairAgent,
        critical_validator: CriticalValidatorAgent,
        compliance_validator: ComplianceValidatorAgent,
        max_repair_attempts: int | None = None,
        tracer: InteractionTracer | None = None,
    ) -> None:
        if max_repair_attempts is not None and max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be at least 0")
        self.generator = generator
        self.adjudicator = adjudicator
        self.repair_agent = repair_agent
        self.critical_validator = critical_validator
        self.compliance_validator = compliance_validator
        self.max_repair_attempts = max_repair_attempts
        self.tracer = tracer or InteractionTracer()

    def generate_verified(self, request: GenerateRequest) -> dict[str, object]:
        last_validation: ValidationResult | None = None

        self.tracer.log_attempt(1, request)
        generated = self.generator.generate(request)

        repair_count = 0
        while True:
            critical_validation = self.critical_validator.validate(request, generated)
            if not critical_validation.passed:
                last_validation = ValidationResult(
                    passed=False,
                    solution_accurate=critical_validation.solution_accurate,
                    curriculum_compliant=True,
                    issues=critical_validation.issues,
                )
                adjudication = self.adjudicator.adjudicate(
                    request, generated, "critical", last_validation.issues
                )
                if adjudication.decision == "validator_false_positive":
                    compliance_validation = self.compliance_validator.validate(request, generated)
                    if compliance_validation.passed:
                        result = generated.to_success_response()
                        self.tracer.log_final_result(result)
                        return result
                    last_validation = ValidationResult(
                        passed=False,
                        solution_accurate=True,
                        curriculum_compliant=compliance_validation.curriculum_compliant,
                        issues=compliance_validation.issues,
                    )
                    adjudication = self.adjudicator.adjudicate(
                        request, generated, "compliance", last_validation.issues
                    )
                    if adjudication.decision == "validator_false_positive":
                        result = generated.to_success_response()
                        self.tracer.log_final_result(result)
                        return result
                    generated = self._repair_or_finish(
                        request,
                        generated,
                        "compliance",
                        last_validation.issues,
                        repair_count,
                        adjudication,
                    )
                    if generated is None:
                        break
                    repair_count += 1
                    continue
                generated = self._repair_or_finish(
                    request,
                    generated,
                    "critical",
                    last_validation.issues,
                    repair_count,
                    adjudication,
                )
                if generated is None:
                    break
                repair_count += 1
                continue

            compliance_validation = self.compliance_validator.validate(request, generated)
            if compliance_validation.passed:
                result = generated.to_success_response()
                self.tracer.log_final_result(result)
                return result
            last_validation = ValidationResult(
                passed=False,
                solution_accurate=True,
                curriculum_compliant=compliance_validation.curriculum_compliant,
                issues=compliance_validation.issues,
            )
            adjudication = self.adjudicator.adjudicate(
                request, generated, "compliance", last_validation.issues
            )
            if adjudication.decision == "validator_false_positive":
                result = generated.to_success_response()
                self.tracer.log_final_result(result)
                return result
            generated = self._repair_or_finish(
                request,
                generated,
                "compliance",
                last_validation.issues,
                repair_count,
                adjudication,
            )
            if generated is None:
                break
            repair_count += 1

        assert last_validation is not None
        result = last_validation.to_failure_response()
        self.tracer.log_final_result(result)
        return result

    def _repair_or_finish(
        self,
        request: GenerateRequest,
        generated: GeneratedProblem,
        validation_stage: str,
        issues: tuple[str, ...],
        repair_count: int,
        adjudication: AdjudicationResult,
    ) -> GeneratedProblem | None:
        if self.max_repair_attempts is not None and repair_count >= self.max_repair_attempts:
            return None
        next_attempt = repair_count + 1
        self.tracer.log_repair_attempt(next_attempt, validation_stage, issues)
        return self.repair_agent.repair(
            request, generated, validation_stage, issues, adjudication
        )
