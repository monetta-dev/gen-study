import tempfile
import unittest
from pathlib import Path

from mathgen.models import ContractError
from mathgen.tikz_renderer import (
    CompileAction,
    TikZCompileAgent,
    TikZRenderer,
    _write_compile_documents,
)


class FakeCompileClient:
    def __init__(self, response):
        self.response = response
        self.messages = []

    def chat(self, messages):
        self.messages.append(messages)
        return self.response


class TikZRendererTests(unittest.TestCase):
    def test_compile_agent_parses_run_command_action(self):
        client = FakeCompileClient(
            '{"action":"run_command","argv":["lualatex","-halt-on-error","main-lualatex.tex"],"reason":"Use LuaLaTeX."}'
        )
        agent = TikZCompileAgent(client)

        action = agent.next_action({"available_commands": ["lualatex"]})

        self.assertEqual(action.action, "run_command")
        self.assertEqual(action.argv[0], "lualatex")
        self.assertEqual(action.reason, "Use LuaLaTeX.")

    def test_compile_action_requires_issues_for_tikz_repair(self):
        with self.assertRaisesRegex(ContractError, "issues"):
            CompileAction.from_dict({"action": "request_tikz_repair"})

    def test_renderer_records_disallowed_agent_command_without_running_it(self):
        client = FakeCompileClient(
            '{"action":"run_command","argv":["lualatex","main.tex;rm -rf ."],"reason":"bad"}'
        )
        renderer = TikZRenderer(TikZCompileAgent(client), max_actions=1)

        result = renderer.render_png(
            "\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}"
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.compile_transcript[1]["observation"]["returncode"], 126)
        self.assertIn(
            "shell syntax",
            result.compile_transcript[1]["observation"]["stderr"],
        )

    def test_renderer_records_unknown_agent_command_without_running_it(self):
        client = FakeCompileClient(
            '{"action":"run_command","argv":["cat","main-lualatex.tex"],"reason":"inspect"}'
        )
        renderer = TikZRenderer(TikZCompileAgent(client), max_actions=1)

        result = renderer.render_png(
            "\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}"
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.compile_transcript[1]["observation"]["returncode"], 126)
        self.assertIn(
            "not an allowed compile command",
            result.compile_transcript[1]["observation"]["stderr"],
        )

    def test_compile_documents_load_intersections_library(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workdir = Path(tmpdir)
            _write_compile_documents(
                workdir,
                "\\begin{tikzpicture}\\end{tikzpicture}",
            )

            document = (workdir / "main-lualatex.tex").read_text(encoding="utf-8")

        self.assertIn("intersections", document)


if __name__ == "__main__":
    unittest.main()
