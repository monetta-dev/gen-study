"""Visual asset generation and validation orchestration."""

from __future__ import annotations

import json
from typing import Protocol

from .agents import InteractionTracer, chat_json
from .models import (
    ContractError,
    VisualGenerateRequest,
    VisualInstruction,
    VisualValidationResult,
)


class VisualChatClient(Protocol):
    def chat(self, messages: list[dict[str, object]]) -> str:
        """Return assistant content for text or multimodal messages."""


class ImageClient(Protocol):
    def generate_png_base64(self, prompt: str) -> str:
        """Generate one PNG image and return base64 content."""


def visual_instruction_system_prompt() -> str:
    return """You are VisualInstructionAgent, an academic visual editor.
Read the completed Japanese entrance-exam problem and solution.
Decide whether a diagram, graph, chemical structure, table-like figure, or other visual asset is useful.
If a visual is needed, write the exact final prompt to send to an image generation model.
The prompt must be detailed enough to generate a precise Japanese university entrance exam style figure.
Prefer monochrome, clean line drawings for physics and mathematics unless the content clearly requires otherwise.
For chemistry, preserve formulas, labels, and experimental details exactly.
Do not create copyrighted textbook-style references or mention existing books.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{"needed":true|false,"placement":"problem|solution|explanation|none","purpose":"diagram|graph|chemical_structure|other|none","prompt":"...","must_include":["..."],"must_not_include":["..."]}"""


def visual_validator_system_prompt() -> str:
    return """You are VisualValidatorAgent, a conservative vision reviewer for academic figures.
Compare the generated image with the original problem, solution, and image prompt.
Return pass if the image is acceptable for a Japanese university entrance exam.
Return repair_required only when there is a concrete visible mismatch that would mislead a student or violate an explicit prompt requirement.
Return uncertain when the issue is ambiguous, low-confidence, or not clearly visible.
Do not demand repair for minor aesthetics when the diagram is scientifically and mathematically usable.
Every repair_required finding must include specific visible evidence.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{"status":"pass|repair_required|uncertain","confidence":0.0,"findings":[{"issue":"...","evidence":"...","severity":"minor|major"}]}"""


def visual_repair_system_prompt() -> str:
    return """You are VisualRepairAgent, an academic image prompt reviser.
Rewrite the image prompt to fix only the validator-confirmed visual issues.
Preserve the original problem, solution, placement, purpose, and all correct constraints.
Make the new prompt more explicit and less ambiguous.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{"prompt":"..."}"""


class VisualInstructionAgent:
    def __init__(
        self, client: VisualChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def create_instruction(self, request: VisualGenerateRequest) -> VisualInstruction:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": visual_instruction_system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "subject": request.subject,
                        "level": request.level,
                        "topic": request.topic,
                        "placement_hint": request.placement_hint,
                        "problem": request.problem,
                        "solution": request.solution,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return chat_json(
            "VisualInstructionAgent",
            self.client,
            self.tracer,
            messages,
            VisualInstruction.from_dict,
            (
                '{"needed":true,"placement":"problem|solution|explanation|none",'
                '"purpose":"diagram|graph|chemical_structure|other|none",'
                '"prompt":"...","must_include":[],"must_not_include":[]}'
            ),
        )


class VisualValidatorAgent:
    def __init__(
        self, client: VisualChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def validate(
        self,
        request: VisualGenerateRequest,
        instruction: VisualInstruction,
        image_base64: str,
    ) -> VisualValidationResult:
        text_payload = json.dumps(
            {
                "subject": request.subject,
                "level": request.level,
                "topic": request.topic,
                "placement": instruction.placement,
                "purpose": instruction.purpose,
                "prompt": instruction.prompt,
                "must_include": list(instruction.must_include),
                "must_not_include": list(instruction.must_not_include),
                "problem": request.problem,
                "solution": request.solution,
            },
            ensure_ascii=False,
        )
        messages: list[dict[str, object]] = [
            {"role": "system", "content": visual_validator_system_prompt()},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_payload},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_base64,
                        },
                    },
                ],
            },
        ]
        return chat_json(
            "VisualValidatorAgent",
            self.client,
            self.tracer,
            messages,
            VisualValidationResult.from_dict,
            (
                '{"status":"pass|repair_required|uncertain","confidence":0.0,'
                '"findings":[{"issue":"...","evidence":"...","severity":"minor|major"}]}'
            ),
        )


class VisualRepairAgent:
    def __init__(
        self, client: VisualChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def repair_prompt(
        self,
        request: VisualGenerateRequest,
        instruction: VisualInstruction,
        validation: VisualValidationResult,
    ) -> VisualInstruction:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": visual_repair_system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "subject": request.subject,
                        "level": request.level,
                        "topic": request.topic,
                        "placement": instruction.placement,
                        "purpose": instruction.purpose,
                        "original_prompt": instruction.prompt,
                        "must_include": list(instruction.must_include),
                        "must_not_include": list(instruction.must_not_include),
                        "validator_status": validation.status,
                        "validator_confidence": validation.confidence,
                        "validator_findings": [
                            finding.to_dict() for finding in validation.findings
                        ],
                        "problem": request.problem,
                        "solution": request.solution,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        result = chat_json(
            "VisualRepairAgent",
            self.client,
            self.tracer,
            messages,
            _repaired_prompt_from_dict,
            '{"prompt":"..."}',
        )
        return VisualInstruction(
            needed=instruction.needed,
            placement=instruction.placement,
            purpose=instruction.purpose,
            prompt=result,
            must_include=instruction.must_include,
            must_not_include=instruction.must_not_include,
        )


class VisualGenerationService:
    def __init__(
        self,
        instruction_agent: VisualInstructionAgent,
        image_client: ImageClient,
        validator: VisualValidatorAgent,
        repair_agent: VisualRepairAgent,
        max_repair_attempts: int = 2,
        tracer: InteractionTracer | None = None,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be at least 0")
        self.instruction_agent = instruction_agent
        self.image_client = image_client
        self.validator = validator
        self.repair_agent = repair_agent
        self.max_repair_attempts = max_repair_attempts
        self.tracer = tracer or InteractionTracer()

    def generate_visual(self, request: VisualGenerateRequest) -> dict[str, object]:
        instruction = self.instruction_agent.create_instruction(request)
        if not instruction.needed:
            return {
                "status": "success",
                "visual_needed": False,
                "visual_assets": [],
                "instruction": instruction.to_dict(),
            }

        attempts: list[dict[str, object]] = []
        repair_count = 0
        while True:
            image_base64 = self.image_client.generate_png_base64(instruction.prompt)
            validation = self.validator.validate(request, instruction, image_base64)
            attempts.append(
                {
                    "prompt": instruction.prompt,
                    "validation": validation.to_dict(),
                }
            )

            if validation.status in {"pass", "uncertain"}:
                return {
                    "status": "success",
                    "visual_needed": True,
                    "visual_validation_status": validation.status,
                    "visual_assets": [
                        {
                            "mime_type": "image/png",
                            "base64": image_base64,
                            "placement": instruction.placement,
                            "purpose": instruction.purpose,
                            "prompt": instruction.prompt,
                        }
                    ],
                    "instruction": instruction.to_dict(),
                    "validation": validation.to_dict(),
                    "attempts": attempts,
                }

            if repair_count >= self.max_repair_attempts:
                return {
                    "status": "failed",
                    "reason": "visual_validator_rejected",
                    "visual_needed": True,
                    "visual_assets": [
                        {
                            "mime_type": "image/png",
                            "base64": image_base64,
                            "placement": instruction.placement,
                            "purpose": instruction.purpose,
                            "prompt": instruction.prompt,
                        }
                    ],
                    "instruction": instruction.to_dict(),
                    "validation": validation.to_dict(),
                    "attempts": attempts,
                }

            instruction = self.repair_agent.repair_prompt(
                request, instruction, validation
            )
            repair_count += 1


def _repaired_prompt_from_dict(data: dict[str, object]) -> str:
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ContractError("`prompt` must be a non-empty string")
    return prompt.strip()
