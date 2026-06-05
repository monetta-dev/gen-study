"""Generator and validator agent orchestration."""

from __future__ import annotations

import json
from typing import Protocol

from .json_utils import parse_json_object
from .models import GenerateRequest, GeneratedProblem, ValidationResult


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]]) -> str:
        """Return the assistant content for the provided chat messages."""


GENERATOR_SYSTEM_PROMPT = """You are Generator, an independent mathematics exam author.
Create one original Japanese university entrance exam mathematics question.
Use DeepSeek thinking internally, but output only strict JSON.
Write both the problem and solution in Japanese.
The problem must be solvable within the Japanese high school mathematics curriculum.
Mathematical expressions must be written in LaTeX using $...$ or $$...$$.
Return exactly this JSON shape:
{"problem":"...","solution":"..."}"""


VALIDATOR_SYSTEM_PROMPT = """You are Validator, an independent mathematics reviewer.
Evaluate the generated problem and solution using this mandatory checklist:
1. Accuracy of Solution: formulas, calculations, logical steps, and existence of a valid solution.
2. Curriculum Compliance: solvable without knowledge beyond Japanese high school mathematics curriculum guidelines.
3. Japanese Output: the final problem and solution text are written in Japanese, with only mathematical notation, symbols, variable names, and standard terms left as needed.
Reject problems with mathematical flaws, insufficient conditions, redundant contradictory conditions, or out-of-scope methods.
Use DeepSeek thinking internally, but output only strict JSON.
Return exactly this JSON shape:
{"passed":true|false,"solution_accurate":true|false,"curriculum_compliant":true|false,"issues":["..."]}"""


class GeneratorAgent:
    def __init__(self, client: ChatClient) -> None:
        self.client = client

    def generate(self, request: GenerateRequest) -> GeneratedProblem:
        content = self.client.chat(
            [
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Generate one entrance exam math question with solution.\n"
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
                {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
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
