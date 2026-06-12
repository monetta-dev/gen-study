import json
import unittest

from mathgen.models import ModelOutputError, VisualGenerateRequest
from mathgen.visual_agents import (
    TikZGenerationAgent,
    VisualGenerationService,
    VisualInstructionAgent,
)


class FakeChatClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def chat(self, messages):
        self.messages.append(messages)
        return self.responses.pop(0)


class VisualGenerationServiceTests(unittest.TestCase):
    def request(self):
        return VisualGenerateRequest.from_dict(
            {
                "subject": "physics",
                "level": "University of Tokyo",
                "topic": "Mechanics",
                "problem": "台車と棒の問題。",
                "solution": "力学的エネルギーを用いる。",
            }
        )

    def test_returns_no_asset_when_visual_not_needed(self):
        instruction_client = FakeChatClient([json.dumps({"needed": False})])
        tikz_client = FakeChatClient([])
        service = VisualGenerationService(
            VisualInstructionAgent(instruction_client),
            TikZGenerationAgent(tikz_client),
        )

        result = service.generate_visual(self.request())

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["visual_needed"])
        self.assertEqual(result["visual_assets"], [])
        self.assertEqual(tikz_client.messages, [])

    def test_generates_tikz_asset(self):
        instruction_client = FakeChatClient(
            [
                json.dumps(
                    {
                        "needed": True,
                        "visuals": [
                            {
                                "id": "figure_1",
                                "placement": "problem",
                                "purpose": "diagram",
                                "prompt": "Draw a cart setup as TikZ.",
                                "must_include": ["cart"],
                                "must_not_include": ["wheels"],
                            }
                        ],
                    }
                )
            ]
        )
        tikz_client = FakeChatClient(
            [
                json.dumps(
                    {
                        "tikz_source": "\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}"
                    }
                )
            ]
        )
        service = VisualGenerationService(
            VisualInstructionAgent(instruction_client),
            TikZGenerationAgent(tikz_client),
        )

        result = service.generate_visual(self.request())

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["visual_backend"], "tikz")
        self.assertEqual(result["visual_validation_status"], "not_run")
        self.assertEqual(result["visual_assets"][0]["id"], "figure_1")
        self.assertEqual(result["visual_assets"][0]["mime_type"], "text/x-tex")
        self.assertIn("\\begin{tikzpicture}", result["visual_assets"][0]["tikz_source"])
        self.assertEqual(result["attempts"][0]["action"], "generate_tikz")
        tikz_payload = json.loads(tikz_client.messages[0][1]["content"])
        self.assertEqual(tikz_payload["visual_prompt"], "Draw a cart setup as TikZ.")
        self.assertEqual(tikz_payload["must_not_include"], ["wheels"])

    def test_generates_multiple_tikz_assets(self):
        instruction_client = FakeChatClient(
            [
                json.dumps(
                    {
                        "needed": True,
                        "visuals": [
                            {
                                "id": "figure_1",
                                "placement": "problem",
                                "purpose": "diagram",
                                "prompt": "Draw the cart setup.",
                                "must_include": ["cart"],
                                "must_not_include": [],
                            },
                            {
                                "id": "figure_2",
                                "placement": "solution",
                                "purpose": "graph",
                                "prompt": "Draw the potential graph.",
                                "must_include": ["graph"],
                                "must_not_include": [],
                            },
                        ],
                    }
                )
            ]
        )
        tikz_client = FakeChatClient(
            [
                json.dumps(
                    {
                        "tikz_source": "\\begin{tikzpicture}\\draw (0,0) rectangle (1,1);\\end{tikzpicture}"
                    }
                ),
                json.dumps(
                    {
                        "tikz_source": "\\begin{tikzpicture}\\draw[->] (0,0)--(2,0);\\end{tikzpicture}"
                    }
                ),
            ]
        )
        service = VisualGenerationService(
            VisualInstructionAgent(instruction_client),
            TikZGenerationAgent(tikz_client),
        )

        result = service.generate_visual(self.request())

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            [asset["id"] for asset in result["visual_assets"]],
            ["figure_1", "figure_2"],
        )
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(len(tikz_client.messages), 2)

    def test_rejects_non_tikz_source(self):
        instruction_client = FakeChatClient(
            [
                json.dumps(
                    {
                        "needed": True,
                        "visuals": [
                            {
                                "id": "figure_1",
                                "placement": "problem",
                                "purpose": "diagram",
                                "prompt": "Draw a cart setup.",
                                "must_include": [],
                                "must_not_include": [],
                            }
                        ],
                    }
                )
            ]
        )
        tikz_client = FakeChatClient(
            [
                json.dumps({"tikz_source": "\\draw (0,0)--(1,0);"}),
                json.dumps({"tikz_source": "\\draw (0,0)--(1,0);"}),
            ]
        )
        service = VisualGenerationService(
            VisualInstructionAgent(instruction_client),
            TikZGenerationAgent(tikz_client),
        )

        with self.assertRaisesRegex(ModelOutputError, "tikzpicture"):
            service.generate_visual(self.request())
