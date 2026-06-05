"""Generator and validator agent orchestration."""

from __future__ import annotations

import json
from typing import Protocol

from .json_utils import parse_json_object
from .models import GenerateRequest, GeneratedProblem, ValidationResult


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant content for the provided chat messages."""


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
    def __init__(self, client: ChatClient) -> None:
        self.client = client

    def generate(self, request: GenerateRequest) -> GeneratedProblem:
        content = self.client.chat(
            [
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
        )
        return GeneratedProblem.from_dict(parse_json_object(content))


class ValidatorAgent:
    def __init__(self, client: ChatClient) -> None:
        self.client = client

    def validate(
        self, request: GenerateRequest, generated: GeneratedProblem
    ) -> ValidationResult:
        content = self.client.chat(
            [
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
        )
        return ValidationResult.from_dict(parse_json_object(content))


class MathGenerationService:
    def __init__(
        self,
        generator: GeneratorAgent,
        validator: ValidatorAgent,
        max_attempts: int = 2,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.generator = generator
        self.validator = validator
        self.max_attempts = max_attempts

    def generate_verified(self, request: GenerateRequest) -> dict[str, object]:
        last_validation: ValidationResult | None = None

        for _ in range(self.max_attempts):
            generated = self.generator.generate(request)
            validation = self.validator.validate(request, generated)
            if validation.passed:
                return generated.to_success_response()
            last_validation = validation

        assert last_validation is not None
        return last_validation.to_failure_response()
