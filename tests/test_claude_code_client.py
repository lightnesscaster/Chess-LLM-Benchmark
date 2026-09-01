"""Tests for the tool-free Claude Code chess player."""

import asyncio
from unittest.mock import AsyncMock

import chess
import pytest

from llm.claude_code_client import ClaudeCodePlayer
from llm.openrouter_client import TransientAPIError


@pytest.fixture
def player():
    return ClaudeCodePlayer(
        player_id="claude-code-opus",
        model_name="anthropic/claude-opus-4.7",
        reasoning_effort="high",
        timeout=30,
    )


def test_command_disables_tools_and_session_persistence(player):
    command = player._command("choose a move")

    assert player.model_name == "claude-opus-4.7"
    assert command[-1] == "choose a move"
    assert command[command.index("--output-format") + 1] == "json"
    assert command[command.index("--model") + 1] == "claude-opus-4.7"
    assert command[command.index("--effort") + 1] == "high"
    assert command[command.index("--tools") + 1] == ""
    assert command[command.index("--disallowedTools") + 1] == "mcp__*"
    assert "--safe-mode" in command
    assert "--no-session-persistence" in command
    assert "--no-chrome" in command


def test_subscription_environment_ignores_api_billing_credentials(player):
    environment = player._subscription_environment({
        "CLAUDE_CODE_OAUTH_TOKEN": "subscription-token",
        "ANTHROPIC_API_KEY": "usage-billed-key",
        "ANTHROPIC_AUTH_TOKEN": "gateway-token",
        "PATH": "/usr/bin",
    })

    assert environment == {
        "CLAUDE_CODE_OAUTH_TOKEN": "subscription-token",
        "PATH": "/usr/bin",
    }


def test_select_move_parses_result_and_tracks_usage(player, monkeypatch):
    monkeypatch.setattr(
        player,
        "_run_cli",
        AsyncMock(return_value=(
            "MOVE: e2e4",
            {"prompt_tokens": 12, "completion_tokens": 3},
        )),
        raising=False,
    )

    move = asyncio.run(player.select_move(chess.Board()))

    assert move == "e2e4"
    assert player.get_token_usage() == {
        "prompt_tokens": 12,
        "completion_tokens": 3,
        "total_tokens": 15,
    }


def test_cli_failure_is_reported_as_transient_provider_error(player, monkeypatch):
    monkeypatch.setattr(
        player,
        "_run_cli",
        AsyncMock(side_effect=RuntimeError("process exited 1")),
        raising=False,
    )

    with pytest.raises(TransientAPIError, match="Claude Code call failed"):
        asyncio.run(player.select_move(chess.Board()))
