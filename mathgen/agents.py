"""Generator and validator agent orchestration."""

from __future__ import annotations

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol, TypeVar

from .json_utils import parse_json_object
from .models import (
    ComplianceValidationResult,
    ContractError,
    CriticalValidationResult,
    GenerateRequest,
    GeneratedProblem,
    ModelOutputError,
    ValidationResult,
)


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]]) -> str:
        """Return the assistant content for the provided chat messages."""


class TraceWriter(Protocol):
    def write(self, text: str) -> object:
        """Write trace text."""

    def flush(self) -> object:
        """Flush trace text."""


class InteractionTracer:
    def __init__(
        self,
        enabled: bool = False,
        writer: TraceWriter | None = None,
        capture: bool = False,
    ) -> None:
        self.enabled = enabled
        self.writer = writer or sys.stdout
        self.capture = capture
        self.records: list[str] = []
        self._lock = threading.Lock()

    def log_request(self, agent_name: str, messages: list[dict[str, Any]]) -> None:
        if not self.enabled and not self.capture:
            return
        self._write(f"\n--- {agent_name} request ---")
        for index, message in enumerate(messages, start=1):
            role = message.get("role", "unknown")
            content = message.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(
                    _redact_trace_content(content),
                    ensure_ascii=False,
                    indent=2,
                )
            self._write(f"[{index}] role={role}\n{content}")

    def log_response(self, agent_name: str, content: str) -> None:
        if not self.enabled and not self.capture:
            return
        self._write(f"--- {agent_name} response ---\n{content}")

    def log_validation_result(
        self,
        validator_name: str,
        passed: bool,
        issues: tuple[str, ...],
        extra: str,
    ) -> None:
        if not self.enabled and not self.capture:
            return
        self._write(
            f"--- {validator_name} decision ---\n"
            f"passed={passed}, {extra}, issues={list(issues)}"
        )

    def log_attempt(self, attempt: int, request: GenerateRequest) -> None:
        if not self.enabled and not self.capture:
            return
        self._write(
            "\n=== Initial generation "
            f"{attempt}: subject={request.subject}, level={request.level}, topic={request.topic} ==="
        )

    def log_repair_attempt(
        self, attempt: int, validation_stage: str, issues: tuple[str, ...]
    ) -> None:
        if not self.enabled and not self.capture:
            return
        self._write(
            "\n=== Repair attempt "
            f"{attempt}: stage={validation_stage}, issues={list(issues)} ==="
        )

    def log_final_result(self, result: dict[str, object]) -> None:
        if not self.enabled and not self.capture:
            return
        self._write(f"=== Final result: status={result.get('status')} ===\n")

    def to_response(self) -> list[str]:
        with self._lock:
            return list(self.records)

    def _write(self, text: str) -> None:
        with self._lock:
            if self.capture:
                self.records.append(text)
            if self.enabled:
                self.writer.write(f"{text}\n")
                self.writer.flush()


T = TypeVar("T")


def chat_json(
    agent_name: str,
    client: ChatClient,
    tracer: InteractionTracer,
    messages: list[dict[str, Any]],
    parser: Callable[[dict[str, Any]], T],
    expected_schema: str,
) -> T:
    tracer.log_request(agent_name, messages)
    content = client.chat(messages)
    tracer.log_response(agent_name, content)
    try:
        return parser(parse_json_object(content))
    except ContractError:
        retry_messages = messages + [
            {"role": "assistant", "content": content},
            {
                "role": "user",
                "content": (
                    "Your previous response was not valid for this API contract. "
                    "Return only one strict JSON object matching exactly this shape: "
                    f"{expected_schema}. Do not include markdown fences, commentary, "
                    "or any text outside the JSON object. Escape newlines inside "
                    "string values as \\n."
                ),
            },
        ]
        tracer.log_request(f"{agent_name} JSON retry", retry_messages)
        retry_content = client.chat(retry_messages)
        tracer.log_response(f"{agent_name} JSON retry", retry_content)
        try:
            return parser(parse_json_object(retry_content))
        except ContractError as retry_exc:
            raise ModelOutputError(str(retry_exc)) from retry_exc


def _redact_trace_content(content: object) -> object:
    if isinstance(content, list):
        return [_redact_trace_content(item) for item in content]
    if isinstance(content, dict):
        redacted = {}
        for key, value in content.items():
            if key == "data" and isinstance(value, str) and len(value) > 120:
                redacted[key] = f"<base64 image redacted: {len(value)} chars>"
            else:
                redacted[key] = _redact_trace_content(value)
        return redacted
    return content


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
    return f"""You are an expert author of original Japanese university entrance exam {instruction["subject_name"]} questions.
Create one original question and complete solution in Japanese.
Match the requested subject, difficulty level, and topic.
The requested level means entrance-exam difficulty within the {instruction["curriculum_scope"]}, not university-level knowledge.
If the topic is broad, vague, or "free", choose a suitable theme that best matches the requested level.
Use internal reasoning if available, but output only strict JSON.
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
3. Subject and Topic Alignment: the problem matches the requested subject. Treat the requested topic as a loose anchor; do not reject merely because the topic is only a small part of a broader high-quality problem.
4. Level Alignment: the problem difficulty, complexity, and expected reasoning depth match the requested level. This is more important than strict topic centrality. Reject if it is clearly too easy, too hard, overly guided, routine, or calibrated to a different exam level.
5. Text-only Policy: reject any problem that requires a diagram, graph, figure, structural drawing, image, or visual reference to solve.
Before deciding, extract the required knowledge, concepts, laws, formulas, and mathematical methods used in the problem and solution. Judge whether each item is taught within the subject curriculum and prerequisite Japanese high school mathematics. Reject if solving requires any item outside that scope, even when the final answer is correct.
High difficulty means harder combinations of high-school methods, not university-level subject knowledge or mathematics. For elite levels such as University of Tokyo, reject routine textbook exercises, single-formula substitutions, direct applications with excessive hints, or problems whose solution path is essentially immediate.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{{"passed":true|false,"curriculum_compliant":true|false,"issues":["..."]}}"""


def repair_system_prompt(subject: str) -> str:
    instruction = _subject_instruction(subject)
    return f"""You are RepairAgent, an independent {instruction["subject_name"]} revision specialist.
Revise the existing problem and solution using the validator feedback.
Preserve the requested subject, difficulty level, Japanese output, and overall intent. Preserve topic relevance when possible, but prioritize level calibration over keeping the topic central.
Change only what is necessary to resolve the reported issues.
Treat the requested level as a difficulty target within the curriculum scope, never as permission to use knowledge beyond that scope.
Do not introduce university-level subject knowledge or mathematics.
The repaired problem must remain solvable within the {instruction["curriculum_scope"]}.
If the issue is that the problem is too easy for the requested level, increase depth through synthesis, hidden setup, or non-obvious reasoning rather than by adding out-of-curriculum knowledge.
{instruction["constraints"]}
Return exactly this JSON shape:
{{"problem":"...","solution":"..."}}"""


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
        return chat_json(
            "Generator",
            self.client,
            self.tracer,
            messages,
            GeneratedProblem.from_dict,
            '{"problem":"...","solution":"..."}',
        )


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
                        "problem": generated.problem,
                        "solution": generated.solution,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return chat_json(
            "RepairAgent",
            self.client,
            self.tracer,
            messages,
            GeneratedProblem.from_dict,
            '{"problem":"...","solution":"..."}',
        )


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
        validation = chat_json(
            "CriticalValidator",
            self.client,
            self.tracer,
            messages,
            CriticalValidationResult.from_dict,
            '{"passed":true,"solution_accurate":true,"issues":[]}',
        )
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
        validation = chat_json(
            "ComplianceValidator",
            self.client,
            self.tracer,
            messages,
            ComplianceValidationResult.from_dict,
            '{"passed":true,"curriculum_compliant":true,"issues":[]}',
        )
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
        repair_agent: RepairAgent,
        critical_validator: CriticalValidatorAgent,
        compliance_validator: ComplianceValidatorAgent,
        max_repair_attempts: int | None = None,
        tracer: InteractionTracer | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if max_repair_attempts is not None and max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be at least 0")
        self.generator = generator
        self.repair_agent = repair_agent
        self.critical_validator = critical_validator
        self.compliance_validator = compliance_validator
        self.max_repair_attempts = max_repair_attempts
        self.tracer = tracer or InteractionTracer()
        self.progress = progress

    def generate_verified(self, request: GenerateRequest) -> dict[str, object]:
        last_validation: ValidationResult | None = None

        self.tracer.log_attempt(1, request)
        self._progress("Generating problem and solution.")
        generated = self.generator.generate(request)

        repair_count = 0
        while True:
            self._progress("Validating correctness and curriculum compliance.")
            critical_validation, compliance_validation = self._validate_parallel(
                request, generated
            )
            if not critical_validation.passed:
                last_validation = ValidationResult(
                    passed=False,
                    solution_accurate=critical_validation.solution_accurate,
                    curriculum_compliant=True,
                    issues=critical_validation.issues,
                )
                generated = self._repair_or_finish(
                    request,
                    generated,
                    "critical",
                    last_validation.issues,
                    repair_count,
                )
                if generated is None:
                    break
                repair_count += 1
                continue

            if compliance_validation.passed:
                result = generated.to_success_response()
                self._progress("Generation completed.")
                self.tracer.log_final_result(result)
                return result
            last_validation = ValidationResult(
                passed=False,
                solution_accurate=True,
                curriculum_compliant=compliance_validation.curriculum_compliant,
                issues=compliance_validation.issues,
            )
            generated = self._repair_or_finish(
                request,
                generated,
                "compliance",
                last_validation.issues,
                repair_count,
            )
            if generated is None:
                break
            repair_count += 1

        assert last_validation is not None
        result = last_validation.to_failure_response()
        self._progress("Generation failed validation.")
        self.tracer.log_final_result(result)
        return result

    def _validate_parallel(
        self, request: GenerateRequest, generated: GeneratedProblem
    ) -> tuple[CriticalValidationResult, ComplianceValidationResult]:
        with ThreadPoolExecutor(max_workers=2) as executor:
            critical_future = executor.submit(
                self.critical_validator.validate, request, generated
            )
            compliance_future = executor.submit(
                self.compliance_validator.validate, request, generated
            )
            return critical_future.result(), compliance_future.result()

    def _repair_or_finish(
        self,
        request: GenerateRequest,
        generated: GeneratedProblem,
        validation_stage: str,
        issues: tuple[str, ...],
        repair_count: int,
    ) -> GeneratedProblem | None:
        if self.max_repair_attempts is not None and repair_count >= self.max_repair_attempts:
            return None
        next_attempt = repair_count + 1
        self._progress(f"Repairing candidate after {validation_stage} validation.")
        self.tracer.log_repair_attempt(next_attempt, validation_stage, issues)
        return self.repair_agent.repair(
            request, generated, validation_stage, issues
        )

    def _progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)
