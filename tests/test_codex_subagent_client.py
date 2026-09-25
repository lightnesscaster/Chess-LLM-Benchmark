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
            subscription_only=True,
        )

    def test_builds_isolated_ephemeral_command(self) -> None:
        command = self.player._command("/tmp/move.txt", "choose a move")

        self.assertEqual(self.player.model_name, "gpt-5.6-sol")
        self.assertIn("--ignore-user-config", command)
        self.assertIn("--skip-git-repo-check", command)
        self.assertIn("--ephemeral", command)
        self.assertIn("--json", command)
        self.assertIn("read-only", command)
        self.assertIn("features.shell_tool=false", command)
        self.assertIn('web_search="disabled"', command)
        self.assertIn("shell_environment_policy.inherit=none", command)

    def test_subscription_environment_is_allowlisted_and_ignores_billing_keys(self) -> None:
        environment = self.player._subprocess_environment({
            "OPENAI_API_KEY": "usage-billed-key",
            "CODEX_API_KEY": "alternate-usage-billed-key",
            "OPENROUTER_API_KEY": "unrelated-secret",
            "CLAUDE_CODE_OAUTH_TOKEN": "unrelated-secret",
            "FIREBASE_WEB_API_KEY": "unrelated-secret",
            "CODEX_HOME": "/var/data/codex",
            "PATH": "/usr/bin",
            "HOME": "/opt/render/project",
            "LANG": "C.UTF-8",
        })

        self.assertEqual(
            environment,
            {
                "CODEX_HOME": "/var/data/codex",
                "PATH": "/usr/bin",
                "HOME": "/opt/render/project",
                "LANG": "C.UTF-8",
            },
        )

    def test_accepts_agent_message_and_reasoning_items(self) -> None:
        stdout = "\n".join(
            [
                json.dumps({"type": "item.completed", "item": {"type": "reasoning"}}),
                json.dumps({"type": "item.completed", "item": {"type": "agent_message"}}),
                json.dumps({"type": "turn.completed", "usage": {"input_tokens": 10}}),
            ]
        )

        self.assertEqual(self.player._disallowed_item_types(stdout), [])

    def test_accepts_codex_diagnostic_error_items(self) -> None:
        stdout = "\n".join(
            [
                json.dumps({"type": "item.completed", "item": {
                    "type": "error",
                    "message": "Codex is ignoring 1 unrecognized configuration setting.",
                }}),
                json.dumps({"type": "item.completed", "item": {"type": "agent_message"}}),
            ]
        )

        self.assertEqual(self.player._disallowed_item_types(stdout), [])

    def test_turn_failure_message_reports_codex_error(self) -> None:
        stdout = "\n".join(
            [
                json.dumps({"type": "error", "message": "Reconnecting... 1/5"}),
                json.dumps({"type": "turn.failed", "error": {
                    "message": "unexpected status 401 Unauthorized",
                }}),
            ]
        )

        self.assertEqual(
            self.player._turn_failure_message(stdout),
            "unexpected status 401 Unauthorized",
        )

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

    def test_expired_login_stops_without_retrying_or_exposing_raw_output(self) -> None:
        from llm.openrouter_client import TransientAPIError

        process = AsyncMock()
        process.returncode = 1
        process.communicate.return_value = (
            b'Your refresh token was already used. Please log out and sign in again. secret-value',
            None,
        )
        with patch("asyncio.create_subprocess_exec", new=AsyncMock(return_value=process)) as spawn:
            with self.assertRaises(TransientAPIError) as caught:
                asyncio.run(self.player._run_codex("choose a move"))
        self.assertEqual(type(caught.exception).__name__, "CodexAuthenticationError")
        self.assertNotIn("secret-value", str(caught.exception))
        self.assertEqual(spawn.await_count, 1)

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
