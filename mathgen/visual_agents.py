"""TikZ visual asset generation orchestration."""

from __future__ import annotations

import json
from typing import Callable, Protocol

from .agents import InteractionTracer, chat_json
from .models import (
    ContractError,
    TikZValidationResult,
    VisualGenerateRequest,
    VisualInstruction,
    VisualInstructionSet,
)


class VisualChatClient(Protocol):
    def chat(self, messages: list[dict[str, object]]) -> str:
        """Return assistant content for text or multimodal messages."""


class TikZRendererClient(Protocol):
    def render_png(self, tikz_source: str):
        """Compile TikZ and return a rendered PNG result."""


def visual_instruction_system_prompt() -> str:
    return """You are VisualInstructionAgent, an academic visual editor.
Read the completed Japanese entrance-exam problem and solution.
Decide whether a diagram, graph, chemical structure, table-like figure, or other visual asset is useful.
If visuals are needed, write one exact final prompt for each TikZ generation call.
For each visual, also create a frozen visual_contract. This contract is local to that visual and must define what later agents should treat as required, what interpretation choices are fixed, and what should not be treated as a reason for repair.
Do not use a global template. Infer the contract from the specific problem, solution, visual purpose, and placement.
The prompt must be detailed enough to generate a precise Japanese university entrance exam style figure.
Prefer monochrome, clean line drawings for physics and mathematics unless the content clearly requires otherwise.
For chemistry, preserve formulas, labels, and experimental details exactly.
Create multiple visual items when the problem or solution naturally needs multiple separate diagrams, graphs, chemical structures, or explanatory figures.
For visuals with placement "problem", include only information explicitly given in the problem statement.
Problem-side visuals must not depict the conclusion to be proved, the final answer, a derived graph or shape, hidden symmetry, auxiliary construction introduced only in the solution, proof-guiding relation, intermediate result, solved state, product, mechanism, trajectory, force decomposition, or any relation students are expected to discover.
If a useful visual would need to show derived or answer-revealing information, set placement to "solution" or "explanation" instead.
When uncertain whether a visual would lower the intended difficulty, prefer placement "solution" over "problem".
For problem-side visual_contract values, explicitly list any conclusion or derived information that must not be drawn in non_goals.
Do not create copyrighted textbook-style references or mention existing books.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{"needed":true|false,"visuals":[{"id":"visual_1","placement":"problem|solution|explanation","purpose":"diagram|graph|chemical_structure|other","prompt":"...","must_include":["..."],"must_not_include":["..."],"visual_contract":{"visual_intent":"...","diagram_type":"...","required_elements":["..."],"style_constraints":["..."],"interpretation_choices":["..."],"acceptance_criteria":["..."],"non_goals":["..."]}}]}"""


def tikz_generation_system_prompt() -> str:
    return """You are TikZGenerationAgent, an academic diagram author.
Create precise TikZ source for one Japanese university entrance-exam visual.
Use the visual instruction, frozen visual_contract, problem, and solution as constraints.
Return only TikZ code inside a JSON object.
The TikZ source must be a single complete tikzpicture environment, not a full LaTeX document.
Do not include \\documentclass, preamble commands, markdown fences, or explanatory text.
Use monochrome line-art style suitable for an exam booklet.
Use clear mathematical labels and avoid unnecessary decoration.
Prefer simple TikZ primitives, coordinates, nodes, arrows, dashed auxiliary lines, and plots when appropriate.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{"tikz_source":"\\begin{tikzpicture}...\\end{tikzpicture}"}"""


def tikz_validation_system_prompt() -> str:
    return """You are TikZValidatorAgent, a conservative acceptance gate for academic TikZ figures.
Review the TikZ source against the frozen visual_contract, visual instruction, problem, and solution.
The visual_contract is fixed. Do not invent new criteria, alternative projection conventions, optional preferences, or new non-goals.
Return pass unless there is a blocking violation of the frozen contract, a required element is missing, a required label/symbol is wrong, or the source is likely not compilable.
During repair validation, evaluate only unresolved previous issues and severe new regressions that make the visual unusable.
Do not reject for minor aesthetics, debatable visibility conventions, exact scale, exact 3D projection, or alternative diagram styles unless the frozen contract explicitly requires them.
Do not require rendering or visual inspection. Judge only from the TikZ source and textual requirements.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{"status":"pass|repair_required","issues":["..."]}"""


def tikz_repair_system_prompt() -> str:
    return """You are TikZRepairAgent, an academic TikZ source reviser.
Repair the current TikZ source using the validator feedback.
Preserve correct parts of the existing source, the frozen visual_contract, labels, placement, purpose, and exam-style monochrome line-art constraints.
Repair only the listed blocking issues. Do not reinterpret the diagram or change the frozen contract.
Return only the repaired TikZ code inside a JSON object.
The TikZ source must be a single complete tikzpicture environment, not a full LaTeX document.
Do not include \\documentclass, preamble commands, markdown fences, or explanatory text.
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
                '"prompt":"...","must_include":[],"must_not_include":[],'
                '"visual_contract":{"visual_intent":"...","diagram_type":"...",'
                '"required_elements":[],"style_constraints":[],'
                '"interpretation_choices":[],"acceptance_criteria":[],'
                '"non_goals":[]}}]}'
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
                        "visual_contract": instruction.visual_contract,
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


class TikZValidatorAgent:
    def __init__(
        self, client: VisualChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def validate_tikz(
        self,
        request: VisualGenerateRequest,
        instruction: VisualInstruction,
        tikz_source: str,
        previous_issues: tuple[str, ...] = (),
    ) -> TikZValidationResult:
        payload = _tikz_context(request, instruction, tikz_source)
        payload["previous_issues"] = list(previous_issues)
        payload["validation_mode"] = (
            "repair_validation" if previous_issues else "initial_validation"
        )
        messages: list[dict[str, object]] = [
            {"role": "system", "content": tikz_validation_system_prompt()},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ]
        return chat_json(
            "TikZValidatorAgent",
            self.client,
            self.tracer,
            messages,
            TikZValidationResult.from_dict,
            '{"status":"pass|repair_required","issues":["..."]}',
        )


class TikZRepairAgent:
    def __init__(
        self, client: VisualChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def repair_tikz(
        self,
        request: VisualGenerateRequest,
        instruction: VisualInstruction,
        tikz_source: str,
        issues: tuple[str, ...],
    ) -> str:
        payload = _tikz_context(request, instruction, tikz_source)
        payload["validator_issues"] = list(issues)
        messages: list[dict[str, object]] = [
            {"role": "system", "content": tikz_repair_system_prompt()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        return chat_json(
            "TikZRepairAgent",
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
        tikz_validator: TikZValidatorAgent,
        tikz_repair_agent: TikZRepairAgent,
        max_repair_attempts: int = 2,
        renderer: TikZRendererClient | None = None,
        tracer: InteractionTracer | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if max_repair_attempts < 0:
            raise ValueError("max_repair_attempts must be at least 0")
        self.instruction_agent = instruction_agent
        self.tikz_agent = tikz_agent
        self.tikz_validator = tikz_validator
        self.tikz_repair_agent = tikz_repair_agent
        self.max_repair_attempts = max_repair_attempts
        self.renderer = renderer
        self.tracer = tracer or InteractionTracer()
        self.progress = progress

    def generate_visual(self, request: VisualGenerateRequest) -> dict[str, object]:
        self._progress("Planning required visuals.")
        instruction_set = self.instruction_agent.create_instruction(request)
        if not instruction_set.needed:
            self._progress("No visual was needed.")
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
            self._progress(f"Generating TikZ for {instruction.visual_id}.")
            result = self._generate_one_tikz_visual(request, instruction)
            attempts.extend(result["attempts"])
            validations.append(result["validation"])
            visual_assets.append(result["asset"])
            validation_statuses.append(str(result["validation_status"]))
            if result["status"] == "failed":
                self._progress(f"TikZ validation failed for {instruction.visual_id}.")
                return {
                    "status": "failed",
                    "reason": "tikz_validator_rejected",
                    "visual_needed": True,
                    "visual_backend": "tikz",
                    "visual_validation_status": "failed",
                    "visual_assets": visual_assets,
                    "instructions": instruction_set.to_dict(),
                    "validations": validations,
                    "attempts": attempts,
                }

        self._progress("Visual generation completed.")
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
        active_issues: tuple[str, ...] = ()
        attempts: list[dict[str, object]] = [
            {
                "visual_id": instruction.visual_id,
                "action": "generate_tikz",
                "prompt": instruction.prompt,
                "tikz_source": tikz_source,
            }
        ]
        repair_count = 0
        while True:
            self._progress(f"Validating TikZ for {instruction.visual_id}.")
            validation = self.tikz_validator.validate_tikz(
                request, instruction, tikz_source, active_issues
            )
            validation_dict = {
                "visual_id": instruction.visual_id,
                **validation.to_dict(),
            }
            if validation.status == "pass":
                render_result = self._render_tikz(instruction.visual_id, tikz_source)
                if render_result is None:
                    asset = _tikz_asset(instruction, tikz_source)
                    validation_status = "pass"
                elif render_result.status == "compiled":
                    asset = _tikz_asset(
                        instruction,
                        tikz_source,
                        render_fields=render_result.to_asset_fields(),
                    )
                    attempts.append(
                        {
                            "visual_id": instruction.visual_id,
                            "action": "compile_tikz",
                            "status": render_result.status,
                            "engine": render_result.engine,
                            "command_count": render_result.command_count,
                        }
                    )
                    validation_status = "compiled"
                else:
                    attempts.append(
                        {
                            "visual_id": instruction.visual_id,
                            "action": "compile_tikz",
                            "status": render_result.status,
                            "issues": list(render_result.issues),
                        }
                    )
                    if repair_count >= self.max_repair_attempts:
                        return {
                            "status": "failed",
                            "asset": _tikz_asset(
                                instruction,
                                tikz_source,
                                render_fields=render_result.to_asset_fields(),
                            ),
                            "validation": {
                                "visual_id": instruction.visual_id,
                                "status": "repair_required",
                                "issues": list(render_result.issues),
                            },
                            "attempts": attempts,
                            "validation_status": "compile_failed",
                        }
                    self._progress(f"Repairing TikZ compile issue for {instruction.visual_id}.")
                    tikz_source = self.tikz_repair_agent.repair_tikz(
                        request,
                        instruction,
                        tikz_source,
                        render_result.issues,
                    )
                    active_issues = render_result.issues
                    repair_count += 1
                    attempts.append(
                        {
                            "visual_id": instruction.visual_id,
                            "action": "repair_tikz_compile",
                            "issues": list(render_result.issues),
                            "tikz_source": tikz_source,
                        }
                    )
                    continue
                return {
                    "status": "success",
                    "asset": asset,
                    "validation": validation_dict,
                    "attempts": attempts,
                    "validation_status": validation_status,
                }

            attempts.append(
                {
                    "visual_id": instruction.visual_id,
                    "action": "validate_tikz",
                    "issues": list(validation.issues),
                }
            )
            if repair_count >= self.max_repair_attempts:
                return {
                    "status": "failed",
                    "asset": _tikz_asset(instruction, tikz_source),
                    "validation": validation_dict,
                    "attempts": attempts,
                    "validation_status": "failed",
                }

            self._progress(f"Repairing TikZ for {instruction.visual_id}.")
            tikz_source = self.tikz_repair_agent.repair_tikz(
                request, instruction, tikz_source, validation.issues
            )
            active_issues = validation.issues
            repair_count += 1
            attempts.append(
                {
                    "visual_id": instruction.visual_id,
                    "action": "repair_tikz",
                    "issues": list(validation.issues),
                    "tikz_source": tikz_source,
                }
            )

    def _progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def _render_tikz(self, visual_id: str, tikz_source: str):
        if self.renderer is None:
            return None
        self._progress(f"Compiling TikZ for {visual_id}.")
        return self.renderer.render_png(tikz_source)


def _tikz_context(
    request: VisualGenerateRequest,
    instruction: VisualInstruction,
    tikz_source: str,
) -> dict[str, object]:
    return {
        "subject": request.subject,
        "level": request.level,
        "topic": request.topic,
        "placement": instruction.placement,
        "purpose": instruction.purpose,
        "visual_prompt": instruction.prompt,
        "visual_contract": instruction.visual_contract,
        "must_include": list(instruction.must_include),
        "must_not_include": list(instruction.must_not_include),
        "problem": request.problem,
        "solution": request.solution,
        "tikz_source": tikz_source,
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
    instruction: VisualInstruction,
    tikz_source: str,
    render_fields: dict[str, object] | None = None,
) -> dict[str, object]:
    asset = {
        "id": instruction.visual_id,
        "mime_type": "text/x-tex",
        "tikz_source": tikz_source,
        "placement": instruction.placement,
        "purpose": instruction.purpose,
        "prompt": instruction.prompt,
    }
    if render_fields:
        asset.update(render_fields)
    return asset


def _aggregate_status(statuses: list[str]) -> str:
    if any(status == "failed" for status in statuses):
        return "failed"
    if statuses and all(status == "pass" for status in statuses):
        return "pass"
    return "pass"
