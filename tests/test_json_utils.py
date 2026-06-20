import unittest

from mathgen.json_utils import parse_json_object
from mathgen.models import ContractError


class JsonUtilsTests(unittest.TestCase):
    def test_parses_fenced_json(self):
        self.assertEqual(
            parse_json_object('```json\n{"prompt":"draw"}\n```'),
            {"prompt": "draw"},
        )

    def test_extracts_json_object_from_surrounding_text(self):
        self.assertEqual(
            parse_json_object('Here is the result:\n{"prompt":"draw"}\nDone.'),
            {"prompt": "draw"},
        )

    def test_accepts_trailing_commas(self):
        self.assertEqual(
            parse_json_object('{"needed": true, "visuals": [],}'),
            {"needed": True, "visuals": []},
        )

    def test_accepts_python_style_dict(self):
        self.assertEqual(
            parse_json_object("{'needed': True, 'visuals': []}"),
            {"needed": True, "visuals": []},
        )

    def test_escapes_raw_newlines_inside_strings(self):
        self.assertEqual(
            parse_json_object('{"prompt":"line 1\nline 2"}'),
            {"prompt": "line 1\nline 2"},
        )

    def test_escapes_raw_latex_backslashes_inside_strings(self):
        parsed = parse_json_object(
            '{"issues":["The formula $\\sum_{k=0}^{\\infty} \\frac{1}{k!}=e$ is outside scope."]}'
        )

        self.assertEqual(
            parsed,
            {
                "issues": [
                    "The formula $\\sum_{k=0}^{\\infty} \\frac{1}{k!}=e$ is outside scope."
                ]
            },
        )

    def test_preserves_valid_json_escapes_inside_strings(self):
        self.assertEqual(
            parse_json_object('{"prompt":"line 1\\nline 2","quote":"\\"x\\""}'),
            {"prompt": "line 1\nline 2", "quote": '"x"'},
        )

    def test_rejects_missing_json_object(self):
        with self.assertRaisesRegex(ContractError, "did not contain"):
            parse_json_object("not json")

    def test_rejects_unrecoverable_invalid_json(self):
        with self.assertRaisesRegex(ContractError, "did not contain"):
            parse_json_object('{"prompt":"broken"')
