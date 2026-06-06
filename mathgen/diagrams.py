"""Deterministic SVG rendering for structured physics diagram specs."""

from __future__ import annotations

import html
import math
from typing import Any

from .models import ContractError


SUPPORTED_DIAGRAM_KINDS = {"scene", "inclined_plane", "force_diagram", "circuit"}
SUPPORTED_SCENE_OBJECTS = {
    "line",
    "arrow",
    "rect",
    "circle",
    "arc",
    "text",
    "polyline",
    "path",
}


def render_diagram(diagram_spec: dict[str, Any] | None) -> dict[str, Any] | None:
    if diagram_spec is None:
        return None

    kind = _string(diagram_spec, "kind")
    if kind == "scene":
        svg = _render_scene(diagram_spec)
    elif kind == "inclined_plane":
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


def _render_scene(spec: dict[str, Any]) -> str:
    canvas = spec.get("canvas", {})
    if canvas is None:
        canvas = {}
    if not isinstance(canvas, dict):
        raise ContractError("`diagram_spec.canvas` must be an object")
    width = _bounded_number(canvas, "width", default=640, minimum=120, maximum=1600)
    height = _bounded_number(canvas, "height", default=420, minimum=120, maximum=1200)
    objects = spec.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ContractError("`diagram_spec.objects` must be a non-empty array")
    if len(objects) > 80:
        raise ContractError("`diagram_spec.objects` must contain at most 80 objects")

    rendered = []
    for index, item in enumerate(objects):
        if not isinstance(item, dict):
            raise ContractError(f"`diagram_spec.objects[{index}]` must be an object")
        rendered.append(_render_scene_object(item, index))

    description = _escape(_optional_string(spec, "description"))
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" height="{height:g}" viewBox="0 0 {width:g} {height:g}" role="img" aria-label="{description}">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#222"/>
    </marker>
    <style>.line{{stroke:#222;stroke-width:2.5;fill:none;stroke-linecap:round;stroke-linejoin:round}}.shape{{stroke:#222;stroke-width:2;fill:#f7f9fb}}.label{{font:16px sans-serif;fill:#111}}.muted{{stroke:#66727f;fill:none}}</style>
  </defs>
  {''.join(rendered)}
</svg>'''


def _render_scene_object(item: dict[str, Any], index: int) -> str:
    object_type = _string(item, "type")
    if object_type not in SUPPORTED_SCENE_OBJECTS:
        raise ContractError(
            f"`diagram_spec.objects[{index}].type` must be one of: "
            + ", ".join(sorted(SUPPORTED_SCENE_OBJECTS))
        )
    if object_type == "line":
        start = _point(item, "from", index)
        end = _point(item, "to", index)
        label = _object_label(item, end[0] + 6, end[1] - 6)
        return f'<line class="line" x1="{start[0]:g}" y1="{start[1]:g}" x2="{end[0]:g}" y2="{end[1]:g}"/>{label}'
    if object_type == "arrow":
        start = _point(item, "from", index)
        end = _point(item, "to", index)
        label = _object_label(item, end[0] + 6, end[1] - 6)
        return f'<line class="line" x1="{start[0]:g}" y1="{start[1]:g}" x2="{end[0]:g}" y2="{end[1]:g}" marker-end="url(#arrow)"/>{label}'
    if object_type == "rect":
        center = _point(item, "center", index)
        size = _size(item, "size", index)
        rotation = _bounded_number(item, "rotation_deg", default=0, minimum=-360, maximum=360)
        x = center[0] - size[0] / 2
        y = center[1] - size[1] / 2
        label = _object_label(item, center[0] - 6, center[1] + 5)
        return f'<g transform="rotate({rotation:g} {center[0]:g} {center[1]:g})"><rect class="shape" x="{x:g}" y="{y:g}" width="{size[0]:g}" height="{size[1]:g}"/>{label}</g>'
    if object_type == "circle":
        center = _point(item, "center", index)
        radius = _bounded_number(item, "radius", default=20, minimum=1, maximum=500)
        label = _object_label(item, center[0] + radius + 6, center[1] + 5)
        return f'<circle class="shape" cx="{center[0]:g}" cy="{center[1]:g}" r="{radius:g}"/>{label}'
    if object_type == "arc":
        center = _point(item, "center", index)
        radius = _bounded_number(item, "radius", default=30, minimum=1, maximum=500)
        start_deg = _bounded_number(item, "start_deg", default=0, minimum=-720, maximum=720)
        end_deg = _bounded_number(item, "end_deg", default=90, minimum=-720, maximum=720)
        start = _polar(center, radius, start_deg)
        end = _polar(center, radius, end_deg)
        large_arc = 1 if abs(end_deg - start_deg) > 180 else 0
        sweep = 1 if end_deg >= start_deg else 0
        label_point = _polar(center, radius + 14, (start_deg + end_deg) / 2)
        label = _object_label(item, label_point[0], label_point[1])
        return f'<path class="line muted" d="M {start[0]:g} {start[1]:g} A {radius:g} {radius:g} 0 {large_arc} {sweep} {end[0]:g} {end[1]:g}"/>{label}'
    if object_type == "text":
        position = _point(item, "position", index)
        text = _string(item, "text")
        return f'<text class="label" x="{position[0]:g}" y="{position[1]:g}">{_escape(text)}</text>'
    if object_type == "polyline":
        points = _points(item, "points", index)
        label = _object_label(item, points[-1][0] + 6, points[-1][1] - 6)
        point_text = " ".join(f"{x:g},{y:g}" for x, y in points)
        return f'<polyline class="line" points="{point_text}"/>{label}'
    path_data = _string(item, "d")
    label = _object_label(item, 0, 0)
    return f'<path class="line" d="{_escape(path_data)}"/>{label}'


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


def _bounded_number(
    spec: dict[str, Any],
    field: str,
    default: float,
    minimum: float,
    maximum: float,
) -> float:
    value = _number(spec, field, default)
    if not minimum <= value <= maximum:
        raise ContractError(
            f"`diagram_spec.{field}` must be between {minimum:g} and {maximum:g}"
        )
    return value


def _point(spec: dict[str, Any], field: str, index: int) -> tuple[float, float]:
    value = spec.get(field)
    if (
        not isinstance(value, list | tuple)
        or len(value) != 2
        or not all(isinstance(item, int | float) for item in value)
    ):
        raise ContractError(
            f"`diagram_spec.objects[{index}].{field}` must be a two-number array"
        )
    return float(value[0]), float(value[1])


def _size(spec: dict[str, Any], field: str, index: int) -> tuple[float, float]:
    width, height = _point(spec, field, index)
    if width <= 0 or height <= 0:
        raise ContractError(
            f"`diagram_spec.objects[{index}].{field}` values must be positive"
        )
    return width, height


def _points(spec: dict[str, Any], field: str, index: int) -> list[tuple[float, float]]:
    value = spec.get(field)
    if not isinstance(value, list) or len(value) < 2:
        raise ContractError(
            f"`diagram_spec.objects[{index}].{field}` must contain at least two points"
        )
    return [_point({field: point}, field, index) for point in value]


def _object_label(item: dict[str, Any], x: float, y: float) -> str:
    label = _optional_string(item, "label")
    if not label:
        return ""
    return f'<text class="label" x="{x:g}" y="{y:g}">{_escape(label)}</text>'


def _polar(center: tuple[float, float], radius: float, degrees: float) -> tuple[float, float]:
    radians = math.radians(degrees)
    return center[0] + radius * math.cos(radians), center[1] + radius * math.sin(radians)


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ContractError("`diagram_spec.forces` must be an array of strings")
    if not all(isinstance(item, str) for item in value):
        raise ContractError("`diagram_spec.forces` must be an array of strings")
    return [item for item in value if item.strip()]


def _escape(value: str) -> str:
    return html.escape(value, quote=True)
