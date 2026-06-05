"""Generator and validator agent orchestration."""

from __future__ import annotations

import json
import sys
from typing import Protocol

from .json_utils import parse_json_object
from .models import GenerateRequest, GeneratedProblem, ValidationResult


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

    def log_validation_result(self, validation: ValidationResult) -> None:
        if not self.enabled:
            return
        self._write(
            "--- Validator decision ---\n"
            f"passed={validation.passed}, "
            f"solution_accurate={validation.solution_accurate}, "
            f"curriculum_compliant={validation.curriculum_compliant}, "
            f"issues={list(validation.issues)}"
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
            "Use only text; do not generate diagrams, figures, image references, or "
            "questions that require a diagram to solve. Describe any physical setup "
            "completely in words. Mathematical and physical expressions must be "
            "written in LaTeX using $...$ or $$...$$."
        ),
        "rejection_scope": (
            "physics flaws, incorrect units, missing assumptions, insufficient "
            "conditions, diagram-dependent wording, redundant contradictory "
            "conditions, or out-of-scope methods"
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
{{"problem":"...","solution":"..."}}"""


def validator_system_prompt(subject: str) -> str:
    instruction = _subject_instruction(subject)
    return f"""You are Validator, an independent {instruction["subject_name"]} reviewer.
Evaluate the generated problem and solution using this mandatory checklist:
1. Accuracy of Solution: {instruction["accuracy_scope"]}.
2. Curriculum Compliance: solvable without knowledge beyond the {instruction["curriculum_scope"]}.
3. Japanese Output: the final problem and solution text are written in Japanese, with only mathematical notation, symbols, variable names, units, and standard terms left as needed.
4. Text-only Requirement: for physics, reject any problem that requires a diagram, figure, image, or visual reference to solve.
Before deciding, extract the required knowledge, concepts, laws, formulas, and mathematical methods used in the problem and solution. Judge whether each item is taught within the subject curriculum and prerequisite Japanese high school mathematics. Reject if solving requires any item outside that scope, even when the final answer is correct.
High difficulty means harder combinations of high-school methods, not university-level physics or mathematics.
Reject problems with {instruction["rejection_scope"]}.
Use DeepSeek thinking internally, but output only strict JSON.
Return exactly this JSON shape:
{{"passed":true|false,"solution_accurate":true|false,"curriculum_compliant":true|false,"issues":["..."]}}"""


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


class ValidatorAgent:
    def __init__(
        self, client: ChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def validate(
        self, request: GenerateRequest, generated: GeneratedProblem
    ) -> ValidationResult:
        messages = [
            {"role": "system", "content": validator_system_prompt(request.subject)},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "subject": request.subject,
                        "level": request.level,
                        "topic": request.topic,
                        "problem": generated.problem,
                        "solution": generated.solution,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        self.tracer.log_request("Validator", messages)
        content = self.client.chat(messages)
        self.tracer.log_response("Validator", content)
        validation = ValidationResult.from_dict(parse_json_object(content))
        self.tracer.log_validation_result(validation)
        return validation


class MathGenerationService:
    def __init__(
        self,
        generator: GeneratorAgent,
        validator: ValidatorAgent,
        max_attempts: int = 2,
        tracer: InteractionTracer | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.generator = generator
        self.validator = validator
        self.max_attempts = max_attempts
        self.tracer = tracer or InteractionTracer()

    def generate_verified(self, request: GenerateRequest) -> dict[str, object]:
        last_validation: ValidationResult | None = None

        for attempt in range(1, self.max_attempts + 1):
            self.tracer.log_attempt(attempt, request)
            generated = self.generator.generate(request)
            validation = self.validator.validate(request, generated)
            if validation.passed:
                result = generated.to_success_response()
                self.tracer.log_final_result(result)
                return result
            last_validation = validation

        assert last_validation is not None
        result = last_validation.to_failure_response()
        self.tracer.log_final_result(result)
        return result
