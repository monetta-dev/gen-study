import unittest

from mathgen.diagrams import render_diagram


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
