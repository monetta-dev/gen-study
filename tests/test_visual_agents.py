import json
import unittest

from mathgen.models import VisualGenerateRequest
from mathgen.visual_agents import (
    VisualGenerationService,
    VisualInstructionAgent,
    VisualRepairAgent,
    VisualValidatorAgent,
)


class FakeChatClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def chat(self, messages):
        self.messages.append(messages)
        return self.responses.pop(0)


class FakeImageClient:
    def __init__(self, images):
        self.images = list(images)
        self.prompts = []
        self.edits = []

    def generate_png_base64(self, prompt):
        self.prompts.append(prompt)
        return self.images.pop(0)

    def edit_png_base64(self, image_base64, prompt):
        self.edits.append((image_base64, prompt))
        return self.images.pop(0)


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
        instruction_client = FakeChatClient(
            [
                json.dumps(
                    {
                        "needed": False,
                        "placement": "none",
                        "purpose": "none",
                        "prompt": "",
                        "must_include": [],
                        "must_not_include": [],
                    }
                )
            ]
        )
        service = VisualGenerationService(
            VisualInstructionAgent(instruction_client),
            FakeImageClient([]),
            VisualValidatorAgent(FakeChatClient([])),
            VisualRepairAgent(FakeChatClient([])),
        )

        result = service.generate_visual(self.request())

        self.assertEqual(result["status"], "success")
        self.assertFalse(result["visual_needed"])
        self.assertEqual(result["visual_assets"], [])

    def test_generates_and_validates_image(self):
        instruction_client = FakeChatClient(
            [
                json.dumps(
                    {
                        "needed": True,
                        "placement": "problem",
                        "purpose": "diagram",
                        "prompt": "Draw a cart and rod.",
                        "must_include": ["cart", "rod"],
                        "must_not_include": ["wheels"],
                    }
                )
            ]
        )
        validator_client = FakeChatClient(
            [
                json.dumps(
                    {
                        "status": "pass",
                        "confidence": 0.93,
                        "findings": [],
                    }
                )
            ]
        )
        image_client = FakeImageClient(["png-base64"])
        service = VisualGenerationService(
            VisualInstructionAgent(instruction_client),
            image_client,
            VisualValidatorAgent(validator_client),
            VisualRepairAgent(FakeChatClient([])),
        )

        result = service.generate_visual(self.request())

        self.assertEqual(result["status"], "success")
        self.assertTrue(result["visual_needed"])
        self.assertEqual(result["visual_validation_status"], "pass")
        self.assertEqual(result["visual_assets"][0]["base64"], "png-base64")
        self.assertEqual(result["visual_assets"][0]["id"], "visual_1")
        self.assertEqual(image_client.prompts, ["Draw a cart and rod."])
        validator_content = validator_client.messages[0][1]["content"]
        self.assertEqual(validator_content[1]["source"]["data"], "png-base64")

    def test_generates_multiple_visuals(self):
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
        image_client = FakeImageClient(["image-1", "image-2"])
        service = VisualGenerationService(
            VisualInstructionAgent(instruction_client),
            image_client,
            VisualValidatorAgent(
                FakeChatClient(
                    [
                        json.dumps(
                            {"status": "pass", "confidence": 0.9, "findings": []}
                        ),
                        json.dumps(
                            {"status": "pass", "confidence": 0.9, "findings": []}
                        ),
                    ]
                )
            ),
            VisualRepairAgent(FakeChatClient([])),
        )

        result = service.generate_visual(self.request())

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["visual_validation_status"], "pass")
        self.assertEqual([asset["id"] for asset in result["visual_assets"]], ["figure_1", "figure_2"])
        self.assertEqual(image_client.prompts, ["Draw the cart setup.", "Draw the potential graph."])

    def test_uncertain_validator_preserves_image_without_repair(self):
        service = VisualGenerationService(
            VisualInstructionAgent(
                FakeChatClient(
                    [
                        json.dumps(
                            {
                                "needed": True,
                                "placement": "problem",
                                "purpose": "diagram",
                                "prompt": "Draw a graph.",
                                "must_include": [],
                                "must_not_include": [],
                            }
                        )
                    ]
                )
            ),
            FakeImageClient(["image-1"]),
            VisualValidatorAgent(
                FakeChatClient(
                    [
                        json.dumps(
                            {
                                "status": "uncertain",
                                "confidence": 0.45,
                                "findings": [],
                            }
                        )
                    ]
                )
            ),
            VisualRepairAgent(FakeChatClient([])),
        )

        result = service.generate_visual(self.request())

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["visual_validation_status"], "uncertain")
        self.assertEqual(len(result["attempts"]), 1)

    def test_repairs_existing_image_after_repair_required(self):
        image_client = FakeImageClient(["bad-image", "good-image"])
        repair_client = FakeChatClient([json.dumps({"prompt": "Draw a thinner rod."})])
        service = VisualGenerationService(
            VisualInstructionAgent(
                FakeChatClient(
                    [
                        json.dumps(
                            {
                                "needed": True,
                                "placement": "problem",
                                "purpose": "diagram",
                                "prompt": "Draw a rod.",
                                "must_include": ["rod"],
                                "must_not_include": [],
                            }
                        )
                    ]
                )
            ),
            image_client,
            VisualValidatorAgent(
                FakeChatClient(
                    [
                        json.dumps(
                            {
                                "status": "repair_required",
                                "confidence": 0.9,
                                "findings": [
                                    {
                                        "issue": "rod is too thick",
                                        "evidence": "visible rectangle",
                                        "severity": "major",
                                    }
                                ],
                            }
                        ),
                        json.dumps(
                            {
                                "status": "pass",
                                "confidence": 0.95,
                                "findings": [],
                            }
                        ),
                    ]
                )
            ),
            VisualRepairAgent(repair_client),
            max_repair_attempts=1,
        )

        result = service.generate_visual(self.request())

        self.assertEqual(result["status"], "success")
        self.assertEqual(image_client.prompts, ["Draw a rod."])
        self.assertEqual(image_client.edits, [("bad-image", "Draw a thinner rod.")])
        self.assertEqual(result["visual_assets"][0]["base64"], "good-image")
        self.assertEqual(result["attempts"][0]["action"], "generate")
        self.assertEqual(result["attempts"][1]["action"], "edit")
        self.assertEqual(len(result["attempts"]), 2)
