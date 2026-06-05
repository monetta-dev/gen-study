import unittest

from mathgen.agents import GeneratorAgent, MathGenerationService, ValidatorAgent
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
        self.assertIn("Use only text", generator_system_prompt)
        self.assertIn("do not generate diagrams", generator_system_prompt)
        self.assertIn("Japanese high school physics curriculum guidelines", generator_system_prompt)
        self.assertIn("Text-only Requirement", validator_system_prompt)
        self.assertIn('"subject": "physics"', validator_payload)

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

        with self.assertRaisesRegex(ValueError, "subject"):
            GenerateRequest.from_dict(
                {"subject": "chemistry", "level": "Standard", "topic": "Moles"}
            )
