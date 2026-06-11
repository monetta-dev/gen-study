"""Visual asset generation and validation orchestration."""

from __future__ import annotations

import json
from typing import Protocol

from .agents import InteractionTracer, chat_json
from .models import (
    ContractError,
    VisualGenerateRequest,
    VisualInstruction,
    VisualInstructionSet,
    VisualValidationResult,
)


class VisualChatClient(Protocol):
    def chat(self, messages: list[dict[str, object]]) -> str:
        """Return assistant content for text or multimodal messages."""


class ImageClient(Protocol):
    def generate_png_base64(self, prompt: str) -> str:
        """Generate one PNG image and return base64 content."""

    def edit_png_base64(self, image_base64: str, prompt: str) -> str:
        """Repair an existing PNG image and return base64 content."""


def visual_instruction_system_prompt() -> str:
    return """You are VisualInstructionAgent, an academic visual editor.
Read the completed Japanese entrance-exam problem and solution.
Decide whether a diagram, graph, chemical structure, table-like figure, or other visual asset is useful.
If visuals are needed, write one exact final prompt for each image generation model call.
The prompt must be detailed enough to generate a precise Japanese university entrance exam style figure.
Prefer monochrome, clean line drawings for physics and mathematics unless the content clearly requires otherwise.
For chemistry, preserve formulas, labels, and experimental details exactly.
Create multiple visual items when the problem or solution naturally needs multiple separate diagrams, graphs, chemical structures, or explanatory figures.
Do not create copyrighted textbook-style references or mention existing books.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{"needed":true|false,"visuals":[{"id":"visual_1","placement":"problem|solution|explanation","purpose":"diagram|graph|chemical_structure|other","prompt":"...","must_include":["..."],"must_not_include":["..."]}]}"""


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
    return """You are VisualRepairAgent, an academic image edit prompt reviser.
Write an image edit prompt to repair the previous generated image.
Fix only the validator-confirmed visual issues.
Preserve the original problem, solution, placement, purpose, and all correct constraints.
Make the edit prompt explicit about what should remain unchanged and what should be corrected.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{"prompt":"..."}"""


class VisualInstructionAgent:
    def __init__(
        self, client: VisualChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def create_instruction(self, request: VisualGenerateRequest) -> VisualInstructionSet:
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
            VisualInstructionSet.from_dict,
            (
                '{"needed":true,"visuals":[{"id":"visual_1",'
                '"placement":"problem|solution|explanation",'
                '"purpose":"diagram|graph|chemical_structure|other",'
                '"prompt":"...","must_include":[],"must_not_include":[]}]}'
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
            visual_id=instruction.visual_id,
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
        instruction_set = self.instruction_agent.create_instruction(request)
        if not instruction_set.needed:
            return {
                "status": "success",
                "visual_needed": False,
                "visual_assets": [],
                "instructions": instruction_set.to_dict(),
            }

        visual_assets: list[dict[str, object]] = []
        attempts: list[dict[str, object]] = []
        validations: list[dict[str, object]] = []
        validation_statuses: list[str] = []
        for instruction in instruction_set.visuals:
            result = self._generate_one_visual(request, instruction)
            attempts.extend(result["attempts"])
            validations.append(result["validation"])
            visual_assets.append(result["asset"])
            validation_statuses.append(str(result["validation_status"]))
            if result["status"] == "failed":
                return {
                    "status": "failed",
                    "reason": "visual_validator_rejected",
                    "visual_needed": True,
                    "visual_assets": visual_assets,
                    "instructions": instruction_set.to_dict(),
                    "validations": validations,
                    "attempts": attempts,
                }

        return {
            "status": "success",
            "visual_needed": True,
            "visual_validation_status": _aggregate_status(validation_statuses),
            "visual_assets": visual_assets,
            "instructions": instruction_set.to_dict(),
            "validations": validations,
            "attempts": attempts,
        }

    def _generate_one_visual(
        self, request: VisualGenerateRequest, instruction: VisualInstruction
    ) -> dict[str, object]:
        image_base64 = self.image_client.generate_png_base64(instruction.prompt)
        attempts: list[dict[str, object]] = []
        repair_count = 0
        action = "generate"
        while True:
            validation = self.validator.validate(request, instruction, image_base64)
            attempts.append(
                {
                    "visual_id": instruction.visual_id,
                    "action": action,
                    "prompt": instruction.prompt,
                    "validation": validation.to_dict(),
                }
            )

            if validation.status in {"pass", "uncertain"}:
                return {
                    "status": "success",
                    "asset": _visual_asset(instruction, image_base64),
                    "validation": validation.to_dict(),
                    "attempts": attempts,
                    "validation_status": validation.status,
                }

            if repair_count >= self.max_repair_attempts:
                return {
                    "status": "failed",
                    "asset": _visual_asset(instruction, image_base64),
                    "validation": validation.to_dict(),
                    "attempts": attempts,
                    "validation_status": validation.status,
                }

            instruction = self.repair_agent.repair_prompt(
                request, instruction, validation
            )
            image_base64 = self.image_client.edit_png_base64(
                image_base64, instruction.prompt
            )
            repair_count += 1
            action = "edit"


def _repaired_prompt_from_dict(data: dict[str, object]) -> str:
    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ContractError("`prompt` must be a non-empty string")
    return prompt.strip()


def _visual_asset(
    instruction: VisualInstruction, image_base64: str
) -> dict[str, object]:
    return {
        "id": instruction.visual_id,
        "mime_type": "image/png",
        "base64": image_base64,
        "placement": instruction.placement,
        "purpose": instruction.purpose,
        "prompt": instruction.prompt,
    }


def _aggregate_status(statuses: list[str]) -> str:
    if any(status == "uncertain" for status in statuses):
        return "uncertain"
    return "pass"
