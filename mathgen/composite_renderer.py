"""Development-only composition of rendered problem, solution, and visuals."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from typing import Any, Protocol

from .agents import InteractionTracer, chat_json
from .models import ContractError


class CompositeRenderError(RuntimeError):
    """Raised when composite PNG rendering fails."""


class CompositeChatClient(Protocol):
    def chat(self, messages: list[dict[str, object]]) -> str:
        """Return assistant content for layout decisions."""


@dataclass(frozen=True)
class CompositeAsset:
    source_id: str
    kind: str
    title: str
    placement: str
    purpose: str
    png_base64: str

    def metadata(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "title": self.title,
            "placement": self.placement,
            "purpose": self.purpose,
        }


@dataclass(frozen=True)
class CompositePlacement:
    source_id: str
    target: str
    reason: str
    reveal_risk: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompositePlacement":
        source_id = data.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip():
            raise ContractError("`source_id` must be a non-empty string")
        target = data.get("target")
        if target not in {"problem", "solution", "both", "unused"}:
            raise ContractError("`target` must be problem, solution, both, or unused")
        reason = data.get("reason", "")
        if not isinstance(reason, str):
            raise ContractError("`reason` must be a string")
        reveal_risk = data.get("reveal_risk", "safe_given_condition")
        if reveal_risk not in {
            "safe_given_condition",
            "minor_hint",
            "answer_revealing",
            "unknown",
        }:
            raise ContractError("`reveal_risk` has an unsupported value")
        return cls(
            source_id=source_id.strip(),
            target=target,
            reason=reason.strip(),
            reveal_risk=reveal_risk,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "source_id": self.source_id,
            "target": self.target,
            "reason": self.reason,
            "reveal_risk": self.reveal_risk,
        }


@dataclass(frozen=True)
class CompositeLayoutDecision:
    placements: tuple[CompositePlacement, ...]
    notes: str
    status: str = "agent_decided"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CompositeLayoutDecision":
        raw_placements = data.get("placements")
        if not isinstance(raw_placements, list):
            raise ContractError("`placements` must be an array")
        notes = data.get("notes", "")
        if not isinstance(notes, str):
            raise ContractError("`notes` must be a string")
        placements = tuple(
            CompositePlacement.from_dict(item)
            for item in raw_placements
            if isinstance(item, dict)
        )
        if len(placements) != len(raw_placements):
            raise ContractError("each `placements` item must be an object")
        return cls(placements=placements, notes=notes.strip())

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "placements": [placement.to_dict() for placement in self.placements],
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CompositeOutputAsset:
    kind: str
    mime_type: str
    png_base64: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "mime_type": self.mime_type,
            "png_base64": self.png_base64,
        }


def composite_layout_system_prompt() -> str:
    return """You are CompositeLayoutAgent, a conservative exam layout editor.
Decide whether each rendered PNG asset belongs on the problem side, the solution side, both, or unused.
Your most important responsibility is to preserve entrance-exam difficulty.
Never place a visual on the problem side if it reveals the solution method, an intermediate result, the final graph shape that students are meant to derive, a numerical answer, a proof idea, or a relation that is not explicitly given in the problem statement.
When uncertain, place the visual on the solution side.
Rendered problem text pages must go on the problem side.
Rendered solution text pages must go on the solution side.
Only place diagrams/graphs on the problem side when they merely communicate given conditions and do not lower the intended difficulty.
Use internal reasoning if available, but output only strict JSON.
Return exactly this JSON shape:
{"placements":[{"source_id":"...","target":"problem|solution|both|unused","reason":"...","reveal_risk":"safe_given_condition|minor_hint|answer_revealing|unknown"}],"notes":"..."}"""


class CompositeLayoutAgent:
    def __init__(
        self, client: CompositeChatClient, tracer: InteractionTracer | None = None
    ) -> None:
        self.client = client
        self.tracer = tracer or InteractionTracer()

    def decide_layout(
        self,
        problem: str,
        solution: str,
        assets: tuple[CompositeAsset, ...],
    ) -> CompositeLayoutDecision:
        messages: list[dict[str, object]] = [
            {"role": "system", "content": composite_layout_system_prompt()},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "problem": problem,
                        "solution": solution,
                        "assets": [asset.metadata() for asset in assets],
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        decision = chat_json(
            "CompositeLayoutAgent",
            self.client,
            self.tracer,
            messages,
            CompositeLayoutDecision.from_dict,
            (
                '{"placements":[{"source_id":"...","target":"problem|solution|both|unused",'
                '"reason":"...","reveal_risk":"safe_given_condition|minor_hint|answer_revealing|unknown"}],'
                '"notes":"..."}'
            ),
        )
        return _sanitize_layout(decision, assets)


class CompositeRenderService:
    def __init__(
        self,
        layout_agent: CompositeLayoutAgent,
        tracer: InteractionTracer | None = None,
    ) -> None:
        self.layout_agent = layout_agent
        self.tracer = tracer or InteractionTracer()

    def render_composite(self, payload: dict[str, Any]) -> dict[str, object]:
        problem = _string_value(payload, "problem")
        solution = _string_value(payload, "solution")
        assets = _collect_assets(payload)
        if not any(asset.kind == "problem_page" for asset in assets):
            raise ContractError("`rendered_output.assets` must include a problem PNG")
        if not any(asset.kind == "solution_page" for asset in assets):
            raise ContractError("`rendered_output.assets` must include a solution PNG")
        try:
            decision = self.layout_agent.decide_layout(problem, solution, assets)
        except Exception:
            decision = _fallback_layout(assets)
        problem_assets, solution_assets = _split_assets(assets, decision)
        return {
            "status": "success",
            "composite_assets": [
                _compose_png("problem", problem_assets).to_dict(),
                _compose_png("solution", solution_assets).to_dict(),
            ],
            "layout_decision": decision.to_dict(),
        }


def _sanitize_layout(
    decision: CompositeLayoutDecision, assets: tuple[CompositeAsset, ...]
) -> CompositeLayoutDecision:
    known_ids = {asset.source_id for asset in assets}
    sanitized: list[CompositePlacement] = []
    seen: set[str] = set()
    for placement in decision.placements:
        if placement.source_id not in known_ids or placement.source_id in seen:
            continue
        seen.add(placement.source_id)
        target = placement.target
        if placement.reveal_risk in {"minor_hint", "answer_revealing", "unknown"}:
            if target in {"problem", "both"}:
                target = "solution"
        asset = next(asset for asset in assets if asset.source_id == placement.source_id)
        if asset.kind == "problem_page":
            target = "problem"
        elif asset.kind == "solution_page":
            target = "solution"
        sanitized.append(
            CompositePlacement(
                source_id=placement.source_id,
                target=target,
                reason=placement.reason,
                reveal_risk=placement.reveal_risk,
            )
        )
    placed_ids = {placement.source_id for placement in sanitized}
    for asset in assets:
        if asset.source_id not in placed_ids:
            sanitized.append(_fallback_placement(asset))
    return CompositeLayoutDecision(
        placements=tuple(sanitized),
        notes=decision.notes,
        status=decision.status,
    )


def _fallback_layout(assets: tuple[CompositeAsset, ...]) -> CompositeLayoutDecision:
    return CompositeLayoutDecision(
        placements=tuple(_fallback_placement(asset) for asset in assets),
        notes="Fallback layout used. Visuals are kept off the problem side unless they are rendered problem text.",
        status="fallback",
    )


def _fallback_placement(asset: CompositeAsset) -> CompositePlacement:
    if asset.kind == "problem_page":
        target = "problem"
        risk = "safe_given_condition"
    elif asset.kind == "solution_page":
        target = "solution"
        risk = "safe_given_condition"
    else:
        target = "solution"
        risk = "unknown"
    return CompositePlacement(
        source_id=asset.source_id,
        target=target,
        reason="Conservative fallback placement.",
        reveal_risk=risk,
    )


def _split_assets(
    assets: tuple[CompositeAsset, ...], decision: CompositeLayoutDecision
) -> tuple[list[CompositeAsset], list[CompositeAsset]]:
    asset_by_id = {asset.source_id: asset for asset in assets}
    problem_assets: list[CompositeAsset] = []
    solution_assets: list[CompositeAsset] = []
    for placement in decision.placements:
        asset = asset_by_id.get(placement.source_id)
        if asset is None:
            continue
        if placement.target in {"problem", "both"}:
            problem_assets.append(asset)
        if placement.target in {"solution", "both"}:
            solution_assets.append(asset)
    return problem_assets, solution_assets


def _collect_assets(payload: dict[str, Any]) -> tuple[CompositeAsset, ...]:
    assets: list[CompositeAsset] = []
    rendered_output = payload.get("rendered_output")
    if not isinstance(rendered_output, dict):
        raise ContractError("`rendered_output` must be an object")
    rendered_assets = rendered_output.get("assets")
    if not isinstance(rendered_assets, list):
        raise ContractError("`rendered_output.assets` must be an array")
    for item in rendered_assets:
        if not isinstance(item, dict):
            raise ContractError("each rendered asset must be an object")
        kind = _string_value(item, "kind")
        source_kind = "solution_page" if kind == "solution" else "problem_page"
        page = item.get("page", len(assets) + 1)
        assets.append(
            CompositeAsset(
                source_id=f"{kind}-page-{page}",
                kind=source_kind,
                title=_optional_string(item, "title", kind),
                placement=kind,
                purpose="rendered_text",
                png_base64=_string_value(item, "png_base64"),
            )
        )
    visual_assets = payload.get("visual_assets", [])
    if visual_assets is None:
        visual_assets = []
    if not isinstance(visual_assets, list):
        raise ContractError("`visual_assets` must be an array")
    for index, item in enumerate(visual_assets, start=1):
        if not isinstance(item, dict):
            raise ContractError("each visual asset must be an object")
        png_base64 = item.get("png_base64")
        if not isinstance(png_base64, str) or not png_base64.strip():
            continue
        source_id = _optional_string(item, "id", f"visual-{index}") or f"visual-{index}"
        assets.append(
            CompositeAsset(
                source_id=source_id,
                kind="visual",
                title=source_id,
                placement=_optional_string(item, "placement", "auto"),
                purpose=_optional_string(item, "purpose", "visual"),
                png_base64=png_base64.strip(),
            )
        )
    return tuple(assets)


def _compose_png(kind: str, assets: list[CompositeAsset]) -> CompositeOutputAsset:
    try:
        from PIL import Image
    except ImportError as exc:
        raise CompositeRenderError("Pillow is required for composite PNG rendering") from exc

    if not assets:
        raise CompositeRenderError(f"No assets were selected for {kind} composite")
    images = [_decode_png(asset.png_base64, Image) for asset in assets]
    max_width = 1400
    margin = 60
    gap = 36
    content_width = max_width - margin * 2
    normalized = [_fit_width(image, content_width) for image in images]
    total_height = margin * 2 + sum(image.height for image in normalized)
    total_height += gap * max(0, len(normalized) - 1)
    canvas = Image.new("RGB", (max_width, total_height), "white")
    y = margin
    for image in normalized:
        x = (max_width - image.width) // 2
        canvas.paste(image, (x, y))
        y += image.height + gap
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return CompositeOutputAsset(
        kind=kind,
        mime_type="image/png",
        png_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
    )


def _decode_png(png_base64: str, image_module):
    try:
        raw = base64.b64decode(png_base64, validate=True)
        image = image_module.open(io.BytesIO(raw))
        return image.convert("RGB")
    except Exception as exc:
        raise CompositeRenderError("Invalid PNG asset") from exc


def _fit_width(image, max_width: int):
    if image.width <= max_width:
        return image
    height = max(1, round(image.height * (max_width / image.width)))
    return image.resize((max_width, height))


def _string_value(data: dict[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"`{field}` must be a non-empty string")
    return value.strip()


def _optional_string(data: dict[str, Any], field: str, default: str = "") -> str:
    value = data.get(field, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ContractError(f"`{field}` must be a string")
    return value.strip()
