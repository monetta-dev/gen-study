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
        critical_validator_client = FakeClient(
            [
                '{"passed":true,"solution_accurate":true,"issues":[]}',
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
        generator_system_prompt = generator_client.messages[0][0]["content"]
        compliance_system_prompt = compliance_validator_client.messages[0][0]["content"]
        self.assertIn("Write both the problem and solution in Japanese", generator_system_prompt)
        self.assertIn("Japanese Output", compliance_system_prompt)
        self.assertIn("mathematics", generator_system_prompt)
        self.assertIn("Subject: mathematics", generator_client.messages[0][1]["content"])

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
        self.assertEqual(len(generator_client.messages), 1)
        self.assertEqual(len(repair_client.messages), 1)
        repair_system_prompt = repair_client.messages[0][0]["content"]
        repair_payload = repair_client.messages[0][1]["content"]
        self.assertIn("RepairAgent", repair_system_prompt)
        self.assertIn("Preserve the requested subject, topic, difficulty level", repair_system_prompt)
        self.assertIn("Change only what is necessary", repair_system_prompt)
        self.assertIn('"validation_stage": "critical"', repair_payload)
        self.assertIn('"adjudicator_decision": "repair_required"', repair_payload)
        self.assertIn('"repair_focus": "fix the calculation"', repair_payload)
        self.assertIn("calculation error", repair_payload)

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
        compliance_validator_client = FakeClient([])
        service = MathGenerationService(
            GeneratorAgent(generator_client),
            AdjudicatorAgent(adjudicator_client),
            RepairAgent(repair_client),
            CriticalValidatorAgent(critical_validator_client),
            ComplianceValidatorAgent(compliance_validator_client),
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
        repair_client = FakeClient([])
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

    def test_empty_diagram_spec_is_treated_as_no_diagram(self):
        generator_client = FakeClient(
            [
                '{"problem":"問","solution":"解","diagram_spec":{}}',
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
                    "level": "Standard",
                    "topic": "Mechanics",
                    "allow_diagram": True,
                }
            )
        )

        self.assertEqual(result["status"], "success")
        self.assertNotIn("diagram", result)

    def test_malformed_scene_diagram_is_repaired_with_coordinate_feedback(self):
        generator_client = FakeClient(
            [
                (
                    '{"problem":"図を用いる問題","solution":"解",'
                    '"diagram_spec":{"kind":"scene","objects":['
                    '{"type":"circle","center":{"x":100,"y":100},"radius":20}]}}'
                ),
            ]
        )
        repair_client = FakeClient(
            [
                (
                    '{"problem":"図を用いる問題","solution":"解",'
                    '"diagram_spec":{"kind":"scene","objects":['
                    '{"type":"circle","center":[100,100],"radius":20}]}}'
                ),
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
            RepairAgent(repair_client),
            CriticalValidatorAgent(critical_validator_client),
            ComplianceValidatorAgent(compliance_validator_client),
            max_repair_attempts=1,
        )

        result = service.generate_verified(
            GenerateRequest.from_dict(
                {
                    "subject": "physics",
                    "level": "Standard",
                    "topic": "Mechanics",
                    "allow_diagram": True,
                }
            )
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("diagram", result)
        repair_payload = repair_client.messages[0][1]["content"]
        self.assertIn("diagram_rendering", repair_payload)
        self.assertIn("center", repair_payload)
        self.assertIn("two-number arrays like [300, 230]", repair_payload)

    def test_non_empty_diagram_without_kind_is_repaired(self):
        generator_client = FakeClient(
            [
                (
                    '{"problem":"図を用いる問題","solution":"解",'
                    '"diagram_spec":{"objects":['
                    '{"type":"circle","center":[100,100],"radius":20}]}}'
                ),
            ]
        )
        repair_client = FakeClient(
            [
                (
                    '{"problem":"図を用いる問題","solution":"解",'
                    '"diagram_spec":{"kind":"scene","objects":['
                    '{"type":"circle","center":[100,100],"radius":20}]}}'
                ),
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
            RepairAgent(repair_client),
            CriticalValidatorAgent(critical_validator_client),
            ComplianceValidatorAgent(compliance_validator_client),
            max_repair_attempts=1,
        )

        result = service.generate_verified(
            GenerateRequest.from_dict(
                {
                    "subject": "physics",
                    "level": "Standard",
                    "topic": "Mechanics",
                    "allow_diagram": True,
                }
            )
        )

        self.assertEqual(result["status"], "success")
        self.assertIn("diagram", result)
        self.assertEqual(result["diagram"]["spec"]["kind"], "scene")
        repair_payload = repair_client.messages[0][1]["content"]
        self.assertIn("diagram_rendering", repair_payload)
        self.assertIn("diagram_spec.kind", repair_payload)
        self.assertIn("include a valid diagram_spec.kind such as scene", repair_payload)

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
        self.assertIn("Do not output raw diagrams", generator_system_prompt)
        self.assertIn("allow_diagram is false", generator_system_prompt)
        self.assertIn('Prefer diagram_spec kind "scene"', generator_system_prompt)
        self.assertIn("line, arrow, rect, circle, arc, text, polyline, and path", generator_system_prompt)
        self.assertIn("Japanese high school physics curriculum guidelines", generator_system_prompt)
        self.assertIn("genuine errors or inconsistencies", critical_system_prompt)
        self.assertIn("Level Alignment", compliance_system_prompt)
        self.assertIn("requested level", compliance_system_prompt)
        self.assertIn("too easy, too hard", compliance_system_prompt)
        self.assertIn("Diagram Policy", compliance_system_prompt)
        self.assertIn("extract the required knowledge", compliance_system_prompt)
        self.assertIn("Judge whether each item is taught", compliance_system_prompt)
        self.assertIn("High difficulty means harder combinations", compliance_system_prompt)
        self.assertIn("not university-level subject knowledge or mathematics", compliance_system_prompt)
        self.assertIn('"subject": "physics"', compliance_payload)
        self.assertIn('"allow_diagram": false', compliance_payload)

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
        self.assertIn('"diagram_spec"', critical_validator_client.messages[0][1]["content"])
        self.assertIn('"diagram_spec"', compliance_validator_client.messages[0][1]["content"])

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
        self.assertIn("do not generate diagrams, structural drawings", generator_system_prompt)
        self.assertIn("reaction equations", generator_system_prompt)
        self.assertIn("genuine errors or inconsistencies", critical_system_prompt)
        self.assertIn("chemistry flaws", critical_system_prompt)
        self.assertIn("Level Alignment", compliance_system_prompt)
        self.assertIn("Diagram Policy", compliance_system_prompt)
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
