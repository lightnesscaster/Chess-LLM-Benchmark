"""Tests for the isolated Codex CLI chess player."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

import chess

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

    def test_permanent_chatgpt_model_failure_is_not_retryable(self) -> None:
        stdout = (
            "The 'gpt-5.2' model is not supported when using Codex "
            "with a ChatGPT account."
        )

        self.assertTrue(self.player._is_permanent_model_failure(stdout))
        self.assertFalse(
            self.player._is_permanent_model_failure("temporary upstream timeout")
        )

    def test_preflight_response_is_reused_as_first_move(self) -> None:
        board = chess.Board()
        usage = {"prompt_tokens": 10, "completion_tokens": 2}

        with patch.object(
            self.player,
            "_run_codex",
            new=AsyncMock(return_value=("MOVE: e2e4", usage)),
        ) as run_codex:
            async def exercise() -> str:
                await self.player.preflight(board)
                self.player.reset_token_usage()
                self.player.reset_timing()
                return await self.player.select_move(board)

            move = asyncio.run(exercise())

        self.assertEqual(move, "e2e4")
        run_codex.assert_awaited_once()
        self.assertEqual(self.player.get_token_usage()["total_tokens"], 12)


if __name__ == "__main__":
    unittest.main()
