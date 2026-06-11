from io import StringIO
import unittest

from mathgen.agents import (
    AdjudicatorAgent,
    GeneratorAgent,
    InteractionTracer,
    MathGenerationService,
    ComplianceValidatorAgent,
    CriticalValidatorAgent,
    RepairAgent,
)
from mathgen.models import GenerateRequest, ModelOutputError


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
        critical_validator_client = FakeClient(
            [
                '{"passed":true,"solution_accurate":true,"issues":[]}',
            ]
        )
        compliance_validator_client = FakeClient(
            [
                '{"passed":true,"curriculum_compliant":true,"issues":[]}',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            AdjudicatorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(critical_validator_client),
            ComplianceValidatorAgent(compliance_validator_client),
        )

        result = service.generate_verified(
            GenerateRequest(level="University of Tokyo", topic="Probability Recurrence Relations")
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("problem", result)
        self.assertIn("solution", result)
        self.assertNotIn("diagram", result)
        self.assertNotIn("visual_assets", result)
        generator_system_prompt = generator_client.messages[0][0]["content"]
        compliance_system_prompt = compliance_validator_client.messages[0][0]["content"]
        self.assertIn("Write both the problem and solution in Japanese", generator_system_prompt)
        self.assertIn('{"problem":"...","solution":"..."}', generator_system_prompt)
        self.assertIn("Japanese Output", compliance_system_prompt)
        self.assertIn("Text-only Policy", compliance_system_prompt)
        self.assertIn("Subject: mathematics", generator_client.messages[0][1]["content"])

    def test_generator_retries_once_after_invalid_json(self):
        generator_client = FakeClient(
            [
                '{"problem":"broken"',
                '{"problem":"問","solution":"解"}',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            AdjudicatorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(
                FakeClient(['{"passed":true,"solution_accurate":true,"issues":[]}'])
            ),
            ComplianceValidatorAgent(
                FakeClient(['{"passed":true,"curriculum_compliant":true,"issues":[]}'])
            ),
        )

        result = service.generate_verified(
            GenerateRequest(level="University of Tokyo", topic="Mechanics")
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["problem"], "問")
        self.assertEqual(len(generator_client.messages), 2)
        self.assertIn("previous response was not valid", generator_client.messages[1][-1]["content"])

    def test_generator_raises_model_output_error_after_json_retry_fails(self):
        generator_client = FakeClient(
            [
                '{"problem":"broken"',
                '{"problem":"still broken"',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            AdjudicatorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(FakeClient([])),
            ComplianceValidatorAgent(FakeClient([])),
        )

        with self.assertRaises(ModelOutputError):
            service.generate_verified(
                GenerateRequest(level="University of Tokyo", topic="Mechanics")
            )

    def test_service_repairs_after_validator_failure(self):
        generator_client = FakeClient(
            [
                '{"problem":"bad","solution":"bad"}',
            ]
        )
        repair_client = FakeClient(
            [
                '{"problem":"fixed","solution":"fixed"}',
            ]
        )
        adjudicator_client = FakeClient(
            [
                (
                    '{"decision":"repair_required","reason":"validator finding is valid",'
                    '"repair_focus":"fix the calculation"}'
                ),
            ]
        )
        critical_validator_client = FakeClient(
            [
                '{"passed":false,"solution_accurate":false,"issues":["calculation error"]}',
                '{"passed":true,"solution_accurate":true,"issues":[]}',
            ]
        )
        compliance_validator_client = FakeClient(
            [
                '{"passed":true,"curriculum_compliant":true,"issues":[]}',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            AdjudicatorAgent(adjudicator_client),
            RepairAgent(repair_client),
            CriticalValidatorAgent(critical_validator_client),
            ComplianceValidatorAgent(compliance_validator_client),
            max_repair_attempts=1,
        )

        result = service.generate_verified(GenerateRequest(level="Nitto-Koma-Sen", topic="Calculus"))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["problem"], "fixed")
        repair_system_prompt = repair_client.messages[0][0]["content"]
        repair_payload = repair_client.messages[0][1]["content"]
        self.assertIn("RepairAgent", repair_system_prompt)
        self.assertIn("Preserve the requested subject, topic, difficulty level", repair_system_prompt)
        self.assertIn("Change only what is necessary", repair_system_prompt)
        self.assertIn('"validation_stage": "critical"', repair_payload)
        self.assertIn('"adjudicator_decision": "repair_required"', repair_payload)
        self.assertIn('"repair_focus": "fix the calculation"', repair_payload)
        self.assertIn("calculation error", repair_payload)
        self.assertNotIn("diagram_spec", repair_payload)

    def test_service_returns_failure_after_repair_limit(self):
        generator_client = FakeClient(
            [
                '{"problem":"bad","solution":"bad"}',
            ]
        )
        repair_client = FakeClient(
            [
                '{"problem":"still bad","solution":"still bad"}',
            ]
        )
        adjudicator_client = FakeClient(
            [
                (
                    '{"decision":"repair_required","reason":"validator finding is valid",'
                    '"repair_focus":"fix the calculation"}'
                ),
                (
                    '{"decision":"repair_required","reason":"validator finding is still valid",'
                    '"repair_focus":"fix the remaining error"}'
                ),
            ]
        )
        critical_validator_client = FakeClient(
            [
                '{"passed":false,"solution_accurate":false,"issues":["calculation error"]}',
                '{"passed":false,"solution_accurate":false,"issues":["still wrong"]}',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            AdjudicatorAgent(adjudicator_client),
            RepairAgent(repair_client),
            CriticalValidatorAgent(critical_validator_client),
            ComplianceValidatorAgent(FakeClient([])),
            max_repair_attempts=1,
        )

        result = service.generate_verified(GenerateRequest(level="Nitto-Koma-Sen", topic="Calculus"))

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["reason"], "validator_rejected")
        self.assertEqual(result["issues"], ["still wrong"])
        self.assertEqual(len(generator_client.messages), 1)
        self.assertEqual(len(repair_client.messages), 1)

    def test_service_ignores_critical_validator_false_positive(self):
        generator_client = FakeClient(
            [
                '{"problem":"valid","solution":"valid"}',
            ]
        )
        adjudicator_client = FakeClient(
            [
                (
                    '{"decision":"validator_false_positive",'
                    '"reason":"validator misunderstood the solution",'
                    '"repair_focus":""}'
                ),
            ]
        )
        critical_validator_client = FakeClient(
            [
                '{"passed":false,"solution_accurate":false,"issues":["not actually wrong"]}',
            ]
        )
        compliance_validator_client = FakeClient(
            [
                '{"passed":true,"curriculum_compliant":true,"issues":[]}',
            ]
        )
        repair_client = FakeClient([])
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            AdjudicatorAgent(adjudicator_client),
            RepairAgent(repair_client),
            CriticalValidatorAgent(critical_validator_client),
            ComplianceValidatorAgent(compliance_validator_client),
            max_repair_attempts=1,
        )

        result = service.generate_verified(GenerateRequest(level="Standard", topic="Calculus"))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["problem"], "valid")
        self.assertEqual(len(repair_client.messages), 0)

    def test_service_repairs_when_adjudicator_requests_clarification(self):
        generator_client = FakeClient(
            [
                '{"problem":"ambiguous","solution":"ambiguous"}',
            ]
        )
        adjudicator_client = FakeClient(
            [
                (
                    '{"decision":"clarify_definition_required",'
                    '"reason":"sign convention is ambiguous",'
                    '"repair_focus":"clarify the sign convention"}'
                ),
            ]
        )
        repair_client = FakeClient(
            [
                '{"problem":"clarified","solution":"clarified"}',
            ]
        )
        critical_validator_client = FakeClient(
            [
                '{"passed":true,"solution_accurate":true,"issues":[]}',
                '{"passed":true,"solution_accurate":true,"issues":[]}',
            ]
        )
        compliance_validator_client = FakeClient(
            [
                '{"passed":false,"curriculum_compliant":true,"issues":["ambiguous convention"]}',
                '{"passed":true,"curriculum_compliant":true,"issues":[]}',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            AdjudicatorAgent(adjudicator_client),
            RepairAgent(repair_client),
            CriticalValidatorAgent(critical_validator_client),
            ComplianceValidatorAgent(compliance_validator_client),
            max_repair_attempts=1,
        )

        result = service.generate_verified(GenerateRequest(level="Standard", topic="Mechanics"))

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["problem"], "clarified")
        repair_payload = repair_client.messages[0][1]["content"]
        self.assertIn('"adjudicator_decision": "clarify_definition_required"', repair_payload)
        self.assertIn("clarify the sign convention", repair_payload)

    def test_request_requires_level_and_topic(self):
        with self.assertRaisesRegex(ValueError, "level"):
            GenerateRequest.from_dict({"level": "", "topic": "Probability"})

        with self.assertRaisesRegex(ValueError, "topic"):
            GenerateRequest.from_dict({"level": "University of Tokyo"})

    def test_request_rejects_visual_generation_flags(self):
        with self.assertRaisesRegex(ValueError, "allow_visuals"):
            GenerateRequest.from_dict(
                {
                    "subject": "physics",
                    "level": "Standard",
                    "topic": "Mechanics",
                    "allow_visuals": True,
                }
            )

        with self.assertRaisesRegex(ValueError, "allow_diagram"):
            GenerateRequest.from_dict(
                {
                    "subject": "physics",
                    "level": "Standard",
                    "topic": "Mechanics",
                    "allow_diagram": True,
                }
            )

    def test_physics_generation_uses_physics_text_only_prompts(self):
        generator_client = FakeClient(
            [
                '{"problem":"水平面上を動く小物体について答えよ。","solution":"運動方程式より求める。"}',
            ]
        )
        critical_validator_client = FakeClient(
            [
                '{"passed":true,"solution_accurate":true,"issues":[]}',
            ]
        )
        compliance_validator_client = FakeClient(
            [
                '{"passed":true,"curriculum_compliant":true,"issues":[]}',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            AdjudicatorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(critical_validator_client),
            ComplianceValidatorAgent(compliance_validator_client),
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
        critical_system_prompt = critical_validator_client.messages[0][0]["content"]
        compliance_system_prompt = compliance_validator_client.messages[0][0]["content"]
        compliance_payload = compliance_validator_client.messages[0][1]["content"]
        self.assertIn("physics", generator_system_prompt)
        self.assertIn("Use only text", generator_system_prompt)
        self.assertIn("Describe setups", generator_system_prompt)
        self.assertIn("Japanese high school physics curriculum guidelines", generator_system_prompt)
        self.assertIn("genuine errors or inconsistencies", critical_system_prompt)
        self.assertIn("Level Alignment", compliance_system_prompt)
        self.assertIn("Text-only Policy", compliance_system_prompt)
        self.assertIn("extract the required knowledge", compliance_system_prompt)
        self.assertIn("not university-level subject knowledge or mathematics", compliance_system_prompt)
        self.assertIn('"subject": "physics"', compliance_payload)
        self.assertNotIn("allow_visuals", compliance_payload)
        self.assertNotIn("diagram_spec", compliance_payload)

    def test_chemistry_generation_uses_chemistry_curriculum_prompts(self):
        generator_client = FakeClient(
            [
                '{"problem":"酢酸水溶液の中和滴定について答えよ。","solution":"物質量の関係より求める。"}',
            ]
        )
        critical_validator_client = FakeClient(
            [
                '{"passed":true,"solution_accurate":true,"issues":[]}',
            ]
        )
        compliance_validator_client = FakeClient(
            [
                '{"passed":true,"curriculum_compliant":true,"issues":[]}',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            AdjudicatorAgent(FakeClient([])),
            RepairAgent(FakeClient([])),
            CriticalValidatorAgent(critical_validator_client),
            ComplianceValidatorAgent(compliance_validator_client),
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
        critical_system_prompt = critical_validator_client.messages[0][0]["content"]
        compliance_system_prompt = compliance_validator_client.messages[0][0]["content"]
        compliance_payload = compliance_validator_client.messages[0][1]["content"]
        self.assertIn("chemistry", generator_system_prompt)
        self.assertIn("Japanese high school chemistry curriculum guidelines", generator_system_prompt)
        self.assertIn("Use only text", generator_system_prompt)
        self.assertIn("reaction equations", generator_system_prompt)
        self.assertIn("genuine errors or inconsistencies", critical_system_prompt)
        self.assertIn("chemistry flaws", critical_system_prompt)
        self.assertIn("Level Alignment", compliance_system_prompt)
        self.assertIn("Text-only Policy", compliance_system_prompt)
        self.assertIn("extract the required knowledge", compliance_system_prompt)
        self.assertIn("not university-level subject knowledge or mathematics", compliance_system_prompt)
        self.assertIn('"subject": "chemistry"', compliance_payload)

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
        critical_validator_client = FakeClient(
            [
                '{"passed":true,"solution_accurate":true,"issues":[]}',
            ]
        )
        compliance_validator_client = FakeClient(
            [
                '{"passed":true,"curriculum_compliant":true,"issues":[]}',
            ]
        )
        service = MathGenerationService(
            GeneratorAgent(generator_client, tracer=tracer),
            AdjudicatorAgent(FakeClient([]), tracer=tracer),
            RepairAgent(FakeClient([]), tracer=tracer),
            CriticalValidatorAgent(critical_validator_client, tracer=tracer),
            ComplianceValidatorAgent(compliance_validator_client, tracer=tracer),
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
        self.assertIn("=== Initial generation 1", trace)
        self.assertIn("--- Generator request ---", trace)
        self.assertIn("Subject: physics", trace)
        self.assertIn("--- Generator response ---", trace)
        self.assertIn("--- CriticalValidator request ---", trace)
        self.assertIn("--- CriticalValidator response ---", trace)
        self.assertIn("--- CriticalValidator decision ---", trace)
        self.assertIn("--- ComplianceValidator request ---", trace)
        self.assertIn("--- ComplianceValidator response ---", trace)
        self.assertIn("--- ComplianceValidator decision ---", trace)
        self.assertIn("passed=True", trace)
        self.assertIn("=== Final result: status=success ===", trace)
