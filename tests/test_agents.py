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
