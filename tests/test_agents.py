from io import StringIO
import unittest

from mathgen.agents import (
    GeneratorAgent,
    InteractionTracer,
    MathGenerationService,
    ValidatorAgent,
)
from mathgen.models import GenerateRequest


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.messages = []

    def chat(self, messages):
        self.messages.append(messages)
        if not self.responses:
            raise AssertionError("No fake responses left")
        return self.responses.pop(0)


class AgentTests(unittest.TestCase):
    def test_service_returns_success_after_validator_passes(self):
        generator_client = FakeClient(
            [
                '{"problem":"$a_n$を求めよ。","solution":"漸化式より $a_n=n$。"}',
            ]
        )
        validator_client = FakeClient(
            [
                '{"passed":true,"solution_accurate":true,"curriculum_compliant":true,"issues":[]}',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            ValidatorAgent(validator_client),
        )

        result = service.generate_verified(
            GenerateRequest(level="University of Tokyo", topic="Probability Recurrence Relations")
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("problem", result)
        self.assertIn("solution", result)
        generator_system_prompt = generator_client.messages[0][0]["content"]
        validator_system_prompt = validator_client.messages[0][0]["content"]
        self.assertIn("Write both the problem and solution in Japanese", generator_system_prompt)
        self.assertIn("Japanese Output", validator_system_prompt)
        self.assertIn("mathematics", generator_system_prompt)
        self.assertIn("Subject: mathematics", generator_client.messages[0][1]["content"])

    def test_service_retries_then_returns_validator_failure(self):
        generator_client = FakeClient(
            [
                '{"problem":"bad","solution":"bad"}',
                '{"problem":"still bad","solution":"still bad"}',
            ]
        )
        validator_client = FakeClient(
            [
                '{"passed":false,"solution_accurate":false,"curriculum_compliant":true,"issues":["calculation error"]}',
                '{"passed":false,"solution_accurate":false,"curriculum_compliant":false,"issues":["out of scope"]}',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            ValidatorAgent(validator_client),
            max_attempts=2,
        )

        result = service.generate_verified(GenerateRequest(level="Nitto-Koma-Sen", topic="Calculus"))

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "validator_rejected")
        self.assertEqual(result["issues"], ["out of scope"])

    def test_request_requires_level_and_topic(self):
        with self.assertRaisesRegex(ValueError, "level"):
            GenerateRequest.from_dict({"level": "", "topic": "Probability"})

        with self.assertRaisesRegex(ValueError, "topic"):
            GenerateRequest.from_dict({"level": "University of Tokyo"})

    def test_physics_generation_uses_physics_text_only_prompts(self):
        generator_client = FakeClient(
            [
                '{"problem":"水平面上を動く小物体について答えよ。","solution":"運動方程式より求める。"}',
            ]
        )
        validator_client = FakeClient(
            [
                '{"passed":true,"solution_accurate":true,"curriculum_compliant":true,"issues":[]}',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            ValidatorAgent(validator_client),
        )

        result = service.generate_verified(
            GenerateRequest.from_dict(
                {
                    "subject": "physics",
                    "level": "University of Tokyo",
                    "topic": "Mechanics",
                }
            )
        )

        self.assertEqual(result["status"], "success")
        generator_system_prompt = generator_client.messages[0][0]["content"]
        validator_system_prompt = validator_client.messages[0][0]["content"]
        validator_payload = validator_client.messages[0][1]["content"]
        self.assertIn("physics", generator_system_prompt)
        self.assertIn("Do not output raw diagrams", generator_system_prompt)
        self.assertIn("allow_diagram is false", generator_system_prompt)
        self.assertIn("Japanese high school physics curriculum guidelines", generator_system_prompt)
        self.assertIn("Text-only Requirement", validator_system_prompt)
        self.assertIn("Diagram Consistency", validator_system_prompt)
        self.assertIn("extract the required knowledge", validator_system_prompt)
        self.assertIn("Judge whether each item is taught", validator_system_prompt)
        self.assertIn("High difficulty means harder combinations", validator_system_prompt)
        self.assertIn("not university-level subject knowledge or mathematics", validator_system_prompt)
        self.assertIn('"subject": "physics"', validator_payload)
        self.assertIn('"allow_diagram": false', validator_payload)

    def test_physics_generation_with_diagram_renders_svg_response(self):
        generator_client = FakeClient(
            [
                (
                    '{"problem":"傾角 $30^\\\\circ$ の斜面上の物体を考える。",'
                    '"solution":"力を斜面方向に分解する。",'
                    '"diagram_spec":{"kind":"inclined_plane","description":"斜面上の物体",'
                    '"angle_deg":30,"block_label":"m","forces":["mg","N"]}}'
                ),
            ]
        )
        validator_client = FakeClient(
            [
                '{"passed":true,"solution_accurate":true,"curriculum_compliant":true,"issues":[]}',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            ValidatorAgent(validator_client),
        )

        result = service.generate_verified(
            GenerateRequest.from_dict(
                {
                    "subject": "physics",
                    "level": "University of Tokyo",
                    "topic": "Mechanics",
                    "allow_diagram": True,
                }
            )
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("diagram", result)
        self.assertEqual(result["diagram"]["type"], "svg")
        self.assertEqual(result["diagram"]["spec"]["kind"], "inclined_plane")
        self.assertIn("<svg", result["diagram"]["svg"])
        self.assertIn("allow_diagram: true", generator_client.messages[0][1]["content"])
        self.assertIn('"diagram_spec"', validator_client.messages[0][1]["content"])

    def test_chemistry_generation_uses_chemistry_curriculum_prompts(self):
        generator_client = FakeClient(
            [
                '{"problem":"酢酸水溶液の中和滴定について答えよ。","solution":"物質量の関係より求める。"}',
            ]
        )
        validator_client = FakeClient(
            [
                '{"passed":true,"solution_accurate":true,"curriculum_compliant":true,"issues":[]}',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            ValidatorAgent(validator_client),
        )

        result = service.generate_verified(
            GenerateRequest.from_dict(
                {
                    "subject": "chemistry",
                    "level": "University of Tokyo",
                    "topic": "Acid-Base Titration",
                }
            )
        )

        self.assertEqual(result["status"], "success")
        generator_system_prompt = generator_client.messages[0][0]["content"]
        validator_system_prompt = validator_client.messages[0][0]["content"]
        validator_payload = validator_client.messages[0][1]["content"]
        self.assertIn("chemistry", generator_system_prompt)
        self.assertIn("Japanese high school chemistry curriculum guidelines", generator_system_prompt)
        self.assertIn("Use only text", generator_system_prompt)
        self.assertIn("do not generate diagrams, structural drawings", generator_system_prompt)
        self.assertIn("reaction equations", generator_system_prompt)
        self.assertIn("Text-only Requirement", validator_system_prompt)
        self.assertIn("extract the required knowledge", validator_system_prompt)
        self.assertIn("not university-level subject knowledge or mathematics", validator_system_prompt)
        self.assertIn("chemistry flaws", validator_system_prompt)
        self.assertIn('"subject": "chemistry"', validator_payload)

    def test_request_accepts_subject_aliases_and_rejects_unknown_subjects(self):
        self.assertEqual(
            GenerateRequest.from_dict(
                {"subject": "物理", "level": "Standard", "topic": "Waves"}
            ).subject,
            "physics",
        )
        self.assertEqual(
            GenerateRequest.from_dict(
                {"subject": "math", "level": "Standard", "topic": "Calculus"}
            ).subject,
            "mathematics",
        )
        self.assertEqual(
            GenerateRequest.from_dict(
                {"subject": "化学", "level": "Standard", "topic": "Chemical Equilibrium"}
            ).subject,
            "chemistry",
        )

        with self.assertRaisesRegex(ValueError, "subject"):
            GenerateRequest.from_dict(
                {"subject": "biology", "level": "Standard", "topic": "Cells"}
            )

    def test_interaction_tracer_logs_generation_and_validation_flow(self):
        trace_output = StringIO()
        tracer = InteractionTracer(enabled=True, writer=trace_output)
        generator_client = FakeClient(
            [
                '{"problem":"磁場中の荷電粒子について答えよ。","solution":"ローレンツ力より求める。"}',
            ]
        )
        validator_client = FakeClient(
            [
                '{"passed":true,"solution_accurate":true,"curriculum_compliant":true,"issues":[]}',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client, tracer=tracer),
            ValidatorAgent(validator_client, tracer=tracer),
            tracer=tracer,
        )

        service.generate_verified(
            GenerateRequest.from_dict(
                {
                    "subject": "physics",
                    "level": "University of Tokyo",
                    "topic": "Magnetic Fields",
                }
            )
        )

        trace = trace_output.getvalue()
        self.assertIn("=== Generation attempt 1", trace)
        self.assertIn("--- Generator request ---", trace)
        self.assertIn("Subject: physics", trace)
        self.assertIn("--- Generator response ---", trace)
        self.assertIn("--- Validator request ---", trace)
        self.assertIn("--- Validator response ---", trace)
        self.assertIn("--- Validator decision ---", trace)
        self.assertIn("passed=True", trace)
        self.assertIn("=== Final result: status=success ===", trace)

    def test_request_rejects_diagrams_for_non_physics_subjects(self):
        with self.assertRaisesRegex(ValueError, "allow_diagram"):
            GenerateRequest.from_dict(
                {
                    "subject": "chemistry",
                    "level": "Standard",
                    "topic": "Acid-Base Titration",
                    "allow_diagram": True,
                }
            )
