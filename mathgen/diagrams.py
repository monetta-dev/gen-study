"""Deterministic SVG rendering for structured physics diagram specs."""

from __future__ import annotations

import html
from typing import Any

from .models import ContractError


SUPPORTED_DIAGRAM_KINDS = {"inclined_plane", "force_diagram", "circuit"}


def render_diagram(diagram_spec: dict[str, Any] | None) -> dict[str, Any] | None:
    if diagram_spec is None:
        return None

    kind = _string(diagram_spec, "kind")
    if kind == "inclined_plane":
        svg = _render_inclined_plane(diagram_spec)
    elif kind == "force_diagram":
        svg = _render_force_diagram(diagram_spec)
    elif kind == "circuit":
        svg = _render_circuit(diagram_spec)
    else:
        raise ContractError(
            "`diagram_spec.kind` must be one of: "
            + ", ".join(sorted(SUPPORTED_DIAGRAM_KINDS))
        )

    return {
        "type": "svg",
        "description": _optional_string(diagram_spec, "description"),
        "spec": diagram_spec,
        "svg": svg,
    }


def _render_inclined_plane(spec: dict[str, Any]) -> str:
    angle = _number(spec, "angle_deg", default=30)
    block_label = _optional_string(spec, "block_label", "m")
    forces = _string_list(spec.get("forces", ["mg", "N"]))
    force_labels = " ".join(
        f'<text x="{20 + index * 45}" y="190" class="label">{_escape(label)}</text>'
        for index, label in enumerate(forces[:4])
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="360" height="220" viewBox="0 0 360 220" role="img" aria-label="{_escape(_optional_string(spec, "description"))}">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#222"/>
    </marker>
    <style>.line{{stroke:#222;stroke-width:3;fill:none}}.thin{{stroke:#555;stroke-width:2;fill:none}}.label{{font:16px sans-serif;fill:#111}}</style>
  </defs>
  <path class="line" d="M40 170 L320 170 L320 70 Z"/>
  <g transform="translate(190 110) rotate(-{angle})">
    <rect x="-26" y="-18" width="52" height="36" fill="#f5f5f5" stroke="#222" stroke-width="2"/>
    <text x="-7" y="6" class="label">{_escape(block_label)}</text>
  </g>
  <line x1="190" y1="110" x2="190" y2="158" class="thin" marker-end="url(#arrow)"/>
  <text x="198" y="150" class="label">mg</text>
  <line x1="190" y1="110" x2="160" y2="80" class="thin" marker-end="url(#arrow)"/>
  <text x="142" y="78" class="label">N</text>
  <path class="thin" d="M265 170 A45 45 0 0 0 249 137"/>
  <text x="252" y="160" class="label">{angle:g}°</text>
  {force_labels}
</svg>'''


def _render_force_diagram(spec: dict[str, Any]) -> str:
    vectors = spec.get("vectors", [])
    if not isinstance(vectors, list) or not vectors:
        vectors = [{"label": "F", "direction": "right"}]

    directions = {
        "up": (0, -60),
        "down": (0, 60),
        "left": (-70, 0),
        "right": (70, 0),
        "up_right": (50, -50),
        "up_left": (-50, -50),
        "down_right": (50, 50),
        "down_left": (-50, 50),
    }
    arrows = []
    for vector in vectors[:8]:
        if not isinstance(vector, dict):
            continue
        label = _optional_string(vector, "label", "F")
        direction = _optional_string(vector, "direction", "right")
        dx, dy = directions.get(direction, directions["right"])
        label_x = 180 + dx + (8 if dx >= 0 else -28)
        label_y = 110 + dy + (5 if dy >= 0 else -8)
        arrows.append(
            f'<line x1="180" y1="110" x2="{180 + dx}" y2="{110 + dy}" class="line" marker-end="url(#arrow)"/>'
            f'<text x="{label_x}" y="{label_y}" class="label">{_escape(label)}</text>'
        )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="360" height="220" viewBox="0 0 360 220" role="img" aria-label="{_escape(_optional_string(spec, "description"))}">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#222"/>
    </marker>
    <style>.line{{stroke:#222;stroke-width:3;fill:none}}.label{{font:16px sans-serif;fill:#111}}</style>
  </defs>
  <circle cx="180" cy="110" r="18" fill="#f5f5f5" stroke="#222" stroke-width="2"/>
  {''.join(arrows)}
</svg>'''


def _render_circuit(spec: dict[str, Any]) -> str:
    components = spec.get("components", [])
    if not isinstance(components, list) or not components:
        components = [{"type": "battery", "label": "E"}, {"type": "resistor", "label": "R"}]
    labels = [
        _optional_string(component, "label", f"C{index + 1}")
        for index, component in enumerate(components[:4])
        if isinstance(component, dict)
    ]
    while len(labels) < 2:
        labels.append(f"C{len(labels) + 1}")
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="360" height="220" viewBox="0 0 360 220" role="img" aria-label="{_escape(_optional_string(spec, "description"))}">
  <style>.line{{stroke:#222;stroke-width:3;fill:none}}.label{{font:16px sans-serif;fill:#111}}</style>
  <path class="line" d="M70 70 H290 V170 H70 Z"/>
  <line class="line" x1="110" y1="55" x2="110" y2="85"/>
  <line class="line" x1="125" y1="62" x2="125" y2="78"/>
  <text x="92" y="48" class="label">{_escape(labels[0])}</text>
  <path class="line" d="M185 70 l8 -12 l8 24 l8 -24 l8 24 l8 -12"/>
  <text x="192" y="48" class="label">{_escape(labels[1])}</text>
  <text x="155" y="195" class="label">{_escape(_optional_string(spec, "label", "回路"))}</text>
</svg>'''


def _string(spec: dict[str, Any], field: str) -> str:
    value = spec.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"`diagram_spec.{field}` must be a non-empty string")
    return value.strip()


def _optional_string(
    spec: dict[str, Any], field: str, default: str = ""
) -> str:
    value = spec.get(field, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ContractError(f"`diagram_spec.{field}` must be a string")
    return value.strip()


def _number(spec: dict[str, Any], field: str, default: float) -> float:
    value = spec.get(field, default)
    if not isinstance(value, int | float):
        raise ContractError(f"`diagram_spec.{field}` must be a number")
    return float(value)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ContractError("`diagram_spec.forces` must be an array of strings")
    if not all(isinstance(item, str) for item in value):
        raise ContractError("`diagram_spec.forces` must be an array of strings")
    return [item for item in value if item.strip()]


def _escape(value: str) -> str:
    return html.escape(value, quote=True)
