"""TikZ visual asset generation orchestration."""

from __future__ import annotations

import json
from typing import Protocol

from .agents import InteractionTracer, chat_json
from .models import (
    ContractError,
    VisualGenerateRequest,
    VisualInstruction,
    VisualInstructionSet,
)


class VisualChatClient(Protocol):
    def chat(self, messages: list[dict[str, object]]) -> str:
        """Return assistant content for text or multimodal messages."""


def visual_instruction_system_prompt() -> str:
    return """You are VisualInstructionAgent, an academic visual editor.
Read the completed Japanese entrance-exam problem and solution.
Decide whether a diagram, graph, chemical structure, table-like figure, or other visual asset is useful.
If visuals are needed, write one exact final prompt for each TikZ generation call.
The prompt must be detailed enough to generate a precise Japanese university entrance exam style figure.
Prefer monochrome, clean line drawings for physics and mathematics unless the content clearly requires otherwise.
For chemistry, preserve formulas, labels, and experimental details exactly.
Create multiple visual items when the problem or solution naturally needs multiple separate diagrams, graphs, chemical structures, or explanatory figures.
Do not create copyrighted textbook-style references or mention existing books.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{"needed":true|false,"visuals":[{"id":"visual_1","placement":"problem|solution|explanation","purpose":"diagram|graph|chemical_structure|other","prompt":"...","must_include":["..."],"must_not_include":["..."]}]}"""


def tikz_generation_system_prompt() -> str:
    return """You are TikZGenerationAgent, an academic diagram author.
Create precise TikZ source for one Japanese university entrance-exam visual.
Use the visual instruction, problem, and solution as constraints.
Return only TikZ code inside a JSON object.
The TikZ source must be a single complete tikzpicture environment, not a full LaTeX document.
Do not include \\documentclass, preamble commands, markdown fences, or explanatory text.
Use monochrome line-art style suitable for an exam booklet.
Use clear mathematical labels and avoid unnecessary decoration.
Prefer simple TikZ primitives, coordinates, nodes, arrows, dashed auxiliary lines, and plots when appropriate.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{"tikz_source":"\\begin{tikzpicture}...\\end{tikzpicture}"}"""


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


class TikZGenerationAgent:
    def __init__(
        self, client: VisualChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def generate_tikz(
        self, request: VisualGenerateRequest, instruction: VisualInstruction
    ) -> str:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": tikz_generation_system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "subject": request.subject,
                        "level": request.level,
                        "topic": request.topic,
                        "placement": instruction.placement,
                        "purpose": instruction.purpose,
                        "visual_prompt": instruction.prompt,
                        "must_include": list(instruction.must_include),
                        "must_not_include": list(instruction.must_not_include),
                        "problem": request.problem,
                        "solution": request.solution,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        return chat_json(
            "TikZGenerationAgent",
            self.client,
            self.tracer,
            messages,
            _tikz_source_from_dict,
            '{"tikz_source":"\\\\begin{tikzpicture}...\\\\end{tikzpicture}"}',
        )


class VisualGenerationService:
    def __init__(
        self,
        instruction_agent: VisualInstructionAgent,
        tikz_agent: TikZGenerationAgent,
        tracer: InteractionTracer | None = None,
    ) -> None:
        self.instruction_agent = instruction_agent
        self.tikz_agent = tikz_agent
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
            result = self._generate_one_tikz_visual(request, instruction)
            attempts.extend(result["attempts"])
            validations.append(result["validation"])
            visual_assets.append(result["asset"])
            validation_statuses.append(str(result["validation_status"]))

        return {
            "status": "success",
            "visual_needed": True,
            "visual_backend": "tikz",
            "visual_validation_status": _aggregate_status(validation_statuses),
            "visual_assets": visual_assets,
            "instructions": instruction_set.to_dict(),
            "validations": validations,
            "attempts": attempts,
        }

    def _generate_one_tikz_visual(
        self, request: VisualGenerateRequest, instruction: VisualInstruction
    ) -> dict[str, object]:
        tikz_source = self.tikz_agent.generate_tikz(request, instruction)
        return {
            "status": "success",
            "asset": _tikz_asset(instruction, tikz_source),
            "validation": {
                "status": "not_run",
                "confidence": 0.0,
                "findings": [],
            },
            "attempts": [
                {
                    "visual_id": instruction.visual_id,
                    "action": "generate_tikz",
                    "prompt": instruction.prompt,
                    "tikz_source": tikz_source,
                }
            ],
            "validation_status": "not_run",
        }


def _tikz_source_from_dict(data: dict[str, object]) -> str:
    tikz_source = data.get("tikz_source")
    if not isinstance(tikz_source, str) or not tikz_source.strip():
        raise ContractError("`tikz_source` must be a non-empty string")
    if (
        "\\begin{tikzpicture}" not in tikz_source
        or "\\end{tikzpicture}" not in tikz_source
    ):
        raise ContractError("`tikz_source` must contain a tikzpicture environment")
    return tikz_source.strip()


def _tikz_asset(
    instruction: VisualInstruction, tikz_source: str
) -> dict[str, object]:
    return {
        "id": instruction.visual_id,
        "mime_type": "text/x-tex",
        "tikz_source": tikz_source,
        "placement": instruction.placement,
        "purpose": instruction.purpose,
        "prompt": instruction.prompt,
    }


def _aggregate_status(statuses: list[str]) -> str:
    if statuses and all(status == "not_run" for status in statuses):
        return "not_run"
    if any(status == "uncertain" for status in statuses):
        return "uncertain"
    return "pass"
