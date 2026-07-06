import json
import unittest

from mathgen.models import ModelOutputError, VisualGenerateRequest
from mathgen.visual_agents import (
    CetzConversionAgent,
    CetzRepairAgent,
    CetzValidatorAgent,
    TikZGenerationAgent,
    TikZRepairAgent,
    TikZValidatorAgent,
    VisualGenerationService,
    VisualInstructionAgent,
    visual_instruction_system_prompt,
)


class FakeChatClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def chat(self, messages):
        self.messages.append(messages)
        return self.responses.pop(0)


class FakeRenderResult:
    status = "compiled"
    engine = "lualatex"
    command_count = 1
    issues = ()

    def to_asset_fields(self):
        return {
            "mime_type": "image/png",
            "png_base64": "cG5n",
            "compile_status": "compiled",
            "compile_engine": "lualatex",
            "compile_command_count": 1,
            "compile_transcript": [{"action": "run_command"}],
        }


class FakeRenderer:
    def __init__(self, result=None):
        self.result = result or FakeRenderResult()
        self.sources = []

    def render_png(self, tikz_source):
        self.sources.append(tikz_source)
        return self.result


class FakeCetzConverter:
    def __init__(self, cetz_source="import cetz.draw: *\nline((0,0), (1,0))"):
        self.cetz_source = cetz_source
        self.sources = []

    def convert_tikz_to_cetz(self, tikz_source):
        self.sources.append(tikz_source)
        return type(
            "FakeCetzConversionResult",
            (),
            {
                "cetz_source": self.cetz_source,
                "transcript": (
                    {
                        "action": "run_command",
                        "argv": ["t2l", "tikz", "input.tex", "-o", "output.typ"],
                    },
                ),
            },
        )()


class VisualGenerationServiceTests(unittest.TestCase):
    def test_visual_instruction_prompt_keeps_problem_visuals_given_only(self):
        prompt = visual_instruction_system_prompt()

        self.assertIn('For visuals with placement "problem"', prompt)
        self.assertIn("explicitly given in the problem statement", prompt)
        self.assertIn("conclusion to be proved", prompt)
        self.assertIn("prefer placement \"solution\"", prompt)

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

    def service(
        self,
        instruction_client,
        tikz_client,
        validator_client=None,
        repair_client=None,
        max_repair_attempts=2,
        renderer=None,
        cetz_converter=None,
        cetz_validator_client=None,
        cetz_repair_client=None,
        cetz_renderer=None,
    ):
        return VisualGenerationService(
            VisualInstructionAgent(instruction_client),
            TikZGenerationAgent(tikz_client),
            TikZValidatorAgent(validator_client or FakeChatClient([])),
            TikZRepairAgent(repair_client or FakeChatClient([])),
            max_repair_attempts=max_repair_attempts,
            renderer=renderer,
            cetz_agent=(
                CetzConversionAgent(cetz_converter)
                if cetz_converter is not None
                else None
            ),
            cetz_validator=(
                CetzValidatorAgent(cetz_validator_client)
                if cetz_validator_client is not None
                else None
            ),
            cetz_repair_agent=(
                CetzRepairAgent(cetz_repair_client)
                if cetz_repair_client is not None
                else None
            ),
            cetz_renderer=cetz_renderer,
        )

    def test_returns_no_asset_when_visual_not_needed(self):
        instruction_client = FakeChatClient([json.dumps({"needed": False})])
        tikz_client = FakeChatClient([])
        service = self.service(instruction_client, tikz_client)

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
                                "visual_contract": {
                                    "visual_intent": "show setup",
                                    "diagram_type": "mechanics_setup",
                                    "acceptance_criteria": ["cart is present"],
                                    "non_goals": ["do not require scale"],
                                },
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
        validator_client = FakeChatClient(
            [json.dumps({"status": "pass", "issues": []})]
        )
        service = self.service(instruction_client, tikz_client, validator_client)

        result = service.generate_visual(self.request())

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["visual_backend"], "tikz")
        self.assertEqual(result["visual_validation_status"], "pass")
        self.assertEqual(result["visual_assets"][0]["id"], "figure_1")
        self.assertEqual(result["visual_assets"][0]["mime_type"], "text/x-tex")
        self.assertIn("\\begin{tikzpicture}", result["visual_assets"][0]["tikz_source"])
        self.assertEqual(result["attempts"][0]["action"], "generate_tikz")
        self.assertEqual(result["validations"][0]["status"], "pass")
        tikz_payload = json.loads(tikz_client.messages[0][1]["content"])
        self.assertEqual(tikz_payload["visual_prompt"], "Draw a cart setup as TikZ.")
        self.assertEqual(tikz_payload["must_not_include"], ["wheels"])
        self.assertEqual(
            tikz_payload["visual_contract"]["diagram_type"],
            "mechanics_setup",
        )
        validator_payload = json.loads(validator_client.messages[0][1]["content"])
        self.assertEqual(
            validator_payload["visual_contract"]["acceptance_criteria"],
            ["cart is present"],
        )
        self.assertEqual(validator_payload["validation_mode"], "initial_validation")

    def test_generates_cetz_comparison_asset_without_replacing_tikz(self):
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
                                "must_include": ["cart"],
                                "must_not_include": [],
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
        validator_client = FakeChatClient(
            [json.dumps({"status": "pass", "issues": []})]
        )
        cetz_converter = FakeCetzConverter()
        cetz_validator_client = FakeChatClient(
            [json.dumps({"status": "pass", "issues": []})]
        )
        service = self.service(
            instruction_client,
            tikz_client,
            validator_client,
            cetz_converter=cetz_converter,
            cetz_validator_client=cetz_validator_client,
            cetz_repair_client=FakeChatClient([]),
        )

        result = service.generate_visual(self.request())

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["visual_backend"], "tikz")
        self.assertIn("tikz_source", result["visual_assets"][0])
        self.assertEqual(result["comparison_backend"], "cetz")
        self.assertEqual(result["comparison_visual_validation_status"], "pass")
        self.assertIn("cetz_source", result["comparison_visual_assets"][0])
        self.assertEqual(
            result["comparison_attempts"][0]["action"],
            "convert_tikz_to_cetz",
        )
        self.assertEqual(result["comparison_attempts"][0]["engine"], "tylax")
        self.assertEqual(
            cetz_converter.sources,
            ["\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}"],
        )

    def test_generates_rendered_png_asset_when_renderer_is_available(self):
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
                                "must_not_include": [],
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
        validator_client = FakeChatClient(
            [json.dumps({"status": "pass", "issues": []})]
        )
        renderer = FakeRenderer()
        service = self.service(
            instruction_client,
            tikz_client,
            validator_client,
            renderer=renderer,
        )

        result = service.generate_visual(self.request())

        asset = result["visual_assets"][0]
        self.assertEqual(asset["mime_type"], "image/png")
        self.assertEqual(asset["png_base64"], "cG5n")
        self.assertEqual(asset["compile_status"], "compiled")
        self.assertEqual(asset["compile_engine"], "lualatex")
        self.assertEqual(result["attempts"][1]["action"], "compile_tikz")
        self.assertEqual(renderer.sources, [asset["tikz_source"]])

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
        validator_client = FakeChatClient(
            [
                json.dumps({"status": "pass", "issues": []}),
                json.dumps({"status": "pass", "issues": []}),
            ]
        )
        service = self.service(instruction_client, tikz_client, validator_client)

        result = service.generate_visual(self.request())

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            [asset["id"] for asset in result["visual_assets"]],
            ["figure_1", "figure_2"],
        )
        self.assertEqual(result["visual_validation_status"], "pass")
        self.assertEqual(len(result["attempts"]), 2)
        self.assertEqual(len(tikz_client.messages), 2)

    def test_repairs_tikz_then_passes_validation(self):
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
                                "prompt": "Draw a rod.",
                                "must_include": ["thin rod"],
                                "must_not_include": [],
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
                        "tikz_source": "\\begin{tikzpicture}\\draw[line width=3pt] (0,0)--(1,0);\\end{tikzpicture}"
                    }
                )
            ]
        )
        validator_client = FakeChatClient(
            [
                json.dumps(
                    {
                        "status": "repair_required",
                        "issues": ["rod is too thick"],
                    }
                ),
                json.dumps({"status": "pass", "issues": []}),
            ]
        )
        repair_client = FakeChatClient(
            [
                json.dumps(
                    {
                        "tikz_source": "\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}"
                    }
                )
            ]
        )
        service = self.service(
            instruction_client, tikz_client, validator_client, repair_client
        )

        result = service.generate_visual(self.request())

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["visual_validation_status"], "pass")
        self.assertIn("\\draw (0,0)--(1,0)", result["visual_assets"][0]["tikz_source"])
        self.assertEqual(
            [attempt["action"] for attempt in result["attempts"]],
            ["generate_tikz", "validate_tikz", "repair_tikz"],
        )
        repair_payload = json.loads(repair_client.messages[0][1]["content"])
        self.assertEqual(repair_payload["validator_issues"], ["rod is too thick"])
        self.assertIn("line width=3pt", repair_payload["tikz_source"])
        second_validation_payload = json.loads(validator_client.messages[1][1]["content"])
        self.assertEqual(second_validation_payload["validation_mode"], "repair_validation")
        self.assertEqual(second_validation_payload["previous_issues"], ["rod is too thick"])

    def test_fails_after_tikz_repair_limit(self):
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
                json.dumps(
                    {
                        "tikz_source": "\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}"
                    }
                )
            ]
        )
        validator_client = FakeChatClient(
            [
                json.dumps({"status": "repair_required", "issues": ["missing label"]}),
                json.dumps({"status": "repair_required", "issues": ["still missing label"]}),
            ]
        )
        repair_client = FakeChatClient(
            [
                json.dumps(
                    {
                        "tikz_source": "\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}"
                    }
                )
            ]
        )
        service = self.service(
            instruction_client,
            tikz_client,
            validator_client,
            repair_client,
            max_repair_attempts=1,
        )

        result = service.generate_visual(self.request())

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "tikz_validator_rejected")
        self.assertEqual(result["visual_validation_status"], "failed")
        self.assertEqual(result["validations"][0]["issues"], ["still missing label"])
        self.assertEqual(
            [attempt["action"] for attempt in result["attempts"]],
            ["generate_tikz", "validate_tikz", "repair_tikz", "validate_tikz"],
        )

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
        service = self.service(instruction_client, tikz_client)

        with self.assertRaisesRegex(ModelOutputError, "tikzpicture"):
            service.generate_visual(self.request())
