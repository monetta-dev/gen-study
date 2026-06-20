import base64
import io
import unittest

from PIL import Image

from mathgen.composite_renderer import (
    CompositeAsset,
    CompositeLayoutAgent,
    CompositeRenderService,
)


class FakeClient:
    def __init__(self, response):
        self.response = response

    def chat(self, messages):
        return self.response


def tiny_png_base64(color="white"):
    image = Image.new("RGB", (12, 8), color)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class CompositeRendererTests(unittest.TestCase):
    def test_answer_revealing_visual_is_moved_off_problem_side(self):
        agent = CompositeLayoutAgent(
            FakeClient(
                """
                {
                  "placements": [
                    {
                      "source_id": "problem-page-1",
                      "target": "problem",
                      "reason": "problem text",
                      "reveal_risk": "safe_given_condition"
                    },
                    {
                      "source_id": "solution-page-1",
                      "target": "solution",
                      "reason": "solution text",
                      "reveal_risk": "safe_given_condition"
                    },
                    {
                      "source_id": "visual_1",
                      "target": "problem",
                      "reason": "shows final graph",
                      "reveal_risk": "answer_revealing"
                    }
                  ],
                  "notes": "test"
                }
                """
            )
        )

        decision = agent.decide_layout(
            "problem",
            "solution",
            (
                CompositeAsset(
                    "problem-page-1",
                    "problem_page",
                    "問題",
                    "problem",
                    "rendered_text",
                    tiny_png_base64(),
                ),
                CompositeAsset(
                    "solution-page-1",
                    "solution_page",
                    "解答",
                    "solution",
                    "rendered_text",
                    tiny_png_base64(),
                ),
                CompositeAsset(
                    "visual_1",
                    "visual",
                    "visual_1",
                    "problem",
                    "graph",
                    tiny_png_base64(),
                ),
            ),
        )

        visual = [
            placement
            for placement in decision.placements
            if placement.source_id == "visual_1"
        ][0]
        self.assertEqual(visual.target, "solution")

    def test_service_composes_problem_and_solution_pngs(self):
        service = CompositeRenderService(
            CompositeLayoutAgent(
                FakeClient(
                    """
                    {
                      "placements": [
                        {"source_id":"problem-page-1","target":"problem","reason":"","reveal_risk":"safe_given_condition"},
                        {"source_id":"solution-page-1","target":"solution","reason":"","reveal_risk":"safe_given_condition"}
                      ],
                      "notes":""
                    }
                    """
                )
            )
        )

        result = service.render_composite(
            {
                "problem": "問",
                "solution": "解",
                "rendered_output": {
                    "assets": [
                        {
                            "kind": "problem",
                            "page": 1,
                            "title": "問題",
                            "mime_type": "image/png",
                            "png_base64": tiny_png_base64(),
                        },
                        {
                            "kind": "solution",
                            "page": 1,
                            "title": "解答",
                            "mime_type": "image/png",
                            "png_base64": tiny_png_base64(),
                        },
                    ]
                },
                "visual_assets": [],
            }
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["composite_assets"]), 2)
        self.assertEqual(result["composite_assets"][0]["kind"], "problem")
        self.assertEqual(result["composite_assets"][1]["kind"], "solution")


if __name__ == "__main__":
    unittest.main()
