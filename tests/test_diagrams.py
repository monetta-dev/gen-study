import unittest

from mathgen.diagrams import render_diagram
from mathgen.models import GeneratedProblem


class DiagramTests(unittest.TestCase):
    def test_render_inclined_plane_returns_svg_diagram_payload(self):
        diagram = render_diagram(
            {
                "kind": "inclined_plane",
                "description": "傾角30度の斜面上の物体",
                "angle_deg": 30,
                "block_label": "m",
                "forces": ["mg", "N"],
            }
        )

        self.assertIsNotNone(diagram)
        assert diagram is not None
        self.assertEqual(diagram["type"], "svg")
        self.assertEqual(diagram["description"], "傾角30度の斜面上の物体")
        self.assertEqual(diagram["spec"]["kind"], "inclined_plane")
        self.assertIn("<svg", diagram["svg"])
        self.assertIn("30", diagram["svg"])
        self.assertIn("mg", diagram["svg"])

    def test_render_rejects_unknown_diagram_kind(self):
        with self.assertRaisesRegex(ValueError, "diagram_spec.kind"):
            render_diagram({"kind": "unknown"})

    def test_generated_problem_treats_empty_diagram_spec_as_none(self):
        generated = GeneratedProblem.from_dict(
            {"problem": "問", "solution": "解", "diagram_spec": {}}
        )

        self.assertIsNone(generated.diagram_spec)

    def test_generated_problem_preserves_non_empty_diagram_spec_without_kind(self):
        generated = GeneratedProblem.from_dict(
            {
                "problem": "問",
                "solution": "解",
                "diagram_spec": {"objects": []},
            }
        )

        self.assertEqual(generated.diagram_spec, {"objects": []})

    def test_render_scene_returns_svg_for_general_primitives(self):
        diagram = render_diagram(
            {
                "kind": "scene",
                "description": "斜面上の物体にはたらく力",
                "canvas": {"width": 640, "height": 420},
                "objects": [
                    {"type": "line", "from": [80, 340], "to": [540, 160], "label": "斜面"},
                    {
                        "type": "rect",
                        "center": [300, 230],
                        "size": [70, 45],
                        "rotation_deg": -25,
                        "label": "m",
                    },
                    {"type": "arrow", "from": [300, 230], "to": [300, 320], "label": "mg"},
                    {"type": "circle", "center": [120, 120], "radius": 24, "label": "O"},
                    {
                        "type": "arc",
                        "center": [500, 340],
                        "radius": 45,
                        "start_deg": 180,
                        "end_deg": 155,
                        "label": "30^\\circ",
                    },
                    {"type": "text", "position": [40, 40], "text": "$x$ 軸"},
                    {"type": "polyline", "points": [[40, 370], [80, 350], [120, 370]]},
                    {"type": "path", "d": "M 450 80 q 30 -40 60 0", "label": "軌跡"},
                ],
            }
        )

        self.assertIsNotNone(diagram)
        assert diagram is not None
        self.assertEqual(diagram["spec"]["kind"], "scene")
        self.assertIn('width="640"', diagram["svg"])
        self.assertIn('marker-end="url(#arrow)"', diagram["svg"])
        self.assertIn("斜面", diagram["svg"])
        self.assertIn("30^\\circ", diagram["svg"])

    def test_render_scene_rejects_unknown_object_type(self):
        with self.assertRaisesRegex(ValueError, "objects\\[0\\].type"):
            render_diagram(
                {
                    "kind": "scene",
                    "objects": [{"type": "unsupported", "from": [0, 0], "to": [10, 10]}],
                }
            )
