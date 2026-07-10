"""Tests for the isolated Codex CLI chess player."""

import json
import unittest

from llm.codex_subagent_client import CodexSubagentPlayer


class CodexSubagentPlayerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.player = CodexSubagentPlayer(
            player_id="gpt-5.6-sol (low)",
            model_name="openai/gpt-5.6-sol",
            reasoning_effort="low",
            max_concurrent=1,
        )

    def test_builds_isolated_ephemeral_command(self) -> None:
        command = self.player._command("/tmp/move.txt", "choose a move")

        self.assertEqual(self.player.model_name, "gpt-5.6-sol")
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--skip-git-repo-check", command)
        self.assertIn("--ephemeral", command)
        self.assertIn("--json", command)
        self.assertIn("read-only", command)

    def test_accepts_agent_message_and_reasoning_items(self) -> None:
        stdout = "\n".join(
            [
                json.dumps({"type": "item.completed", "item": {"type": "reasoning"}}),
                json.dumps({"type": "item.completed", "item": {"type": "agent_message"}}),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10}}),
            ]
        )

        self.assertEqual(self.player._disallowed_item_types(stdout), [])

    def test_rejects_command_and_other_tool_items(self) -> None:
        stdout = "\n".join(
            [
                json.dumps({"type": "item.started", "item": {"type": "command_execution"}}),
                json.dumps({"type": "item.completed", "item": {"type": "command_execution"}}),
                json.dumps({"type": "item.completed", "item": {"type": "web_search"}}),
            ]
        )

        self.assertEqual(
            self.player._disallowed_item_types(stdout),
            ["command_execution", "web_search"],
        )


if __name__ == "__main__":
    unittest.main()
