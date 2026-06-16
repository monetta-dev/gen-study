import unittest

from mathgen.models import ContractError
from mathgen.tikz_renderer import CompileAction, TikZCompileAgent, TikZRenderer


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

    def test_renderer_rejects_shell_syntax_in_agent_command(self):
        client = FakeCompileClient(
            '{"action":"run_command","argv":["lualatex","main.tex;rm -rf ."],"reason":"bad"}'
        )
        renderer = TikZRenderer(TikZCompileAgent(client), max_actions=1)

        with self.assertRaisesRegex(ContractError, "shell syntax"):
            renderer.render_png(
                "\\begin{tikzpicture}\\draw (0,0)--(1,0);\\end{tikzpicture}"
            )


if __name__ == "__main__":
    unittest.main()
