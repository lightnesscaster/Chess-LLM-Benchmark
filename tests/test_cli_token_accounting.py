"""Regression tests for CLI raw usage and bounded chess input accounting."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import chess
import pytest

from llm.codex_subagent_client import CodexSubagentPlayer
from llm.claude_code_client import ClaudeCodePlayer


def test_codex_retains_cache_and_separates_runtime_input():
    player = CodexSubagentPlayer("codex")
    player.last_prompt = "hello"
    usage = player._parse_usage(json.dumps({"type": "turn.completed", "usage": {
        "input_tokens": 100, "output_tokens": 25, "cached_input_tokens": 100,
    }}))
    player._track_usage(usage)
    result = player.get_token_usage()
    assert result.get("cached_input_tokens") == 100
    assert result["chess_prompt_tokens"] == 1
    assert result["runtime_prompt_tokens"] == 99
    assert result["chess_cached_input_tokens"] == 1
    assert result["completion_tokens"] == 25
    assert result["total_tokens"] == 125
    assert result["cache_accounting_known"] is True
    assert "estimated" in result["input_accounting_method"]
    assert "o200k_base" in result["input_accounting_method"]


@pytest.mark.parametrize("cached,want", [(0, 0), (99, 0), (100, 1)])
def test_codex_allocates_cache_to_runtime_prefix_first(cached, want):
    player = CodexSubagentPlayer("codex")
    player.last_prompt = "hello"
    player._track_usage(player._parse_usage(json.dumps({
        "type": "turn.completed", "usage": {
            "input_tokens": 100, "output_tokens": 4, "cached_input_tokens": cached,
        },
    })))
    assert player.get_token_usage().get("chess_cached_input_tokens") == want


def test_accounting_unknown_cache_accumulates_and_clone_isolates():
    player = CodexSubagentPlayer("codex")
    player.last_prompt = "hello"
    player._track_usage({"prompt_tokens": 100, "completion_tokens": 4})
    player._track_usage(player._parse_usage(json.dumps({
        "type": "turn.completed", "usage": {
            "input_tokens": 100, "output_tokens": 6, "cached_input_tokens": 0,
        },
    })))
    result = player.get_token_usage()
    assert result.get("chess_prompt_tokens") == 2
    assert result["cache_accounting_known"] is False
    assert len(result["input_accounting_requests"]) == 2
    clone = player.clone_for_game()
    clone.last_prompt = "hello"
    clone._track_usage({"prompt_tokens": 1, "completion_tokens": 1})
    assert len(player.get_token_usage()["input_accounting_requests"]) == 2
    result["input_accounting_requests"].clear()
    assert len(player.get_token_usage()["input_accounting_requests"]) == 2
    player.reset_token_usage()
    assert player.get_token_usage()["prompt_tokens"] == 0
    assert not player.get_token_usage().get("input_accounting_requests")


def test_claude_normalizes_disjoint_input_buckets_through_real_cli(monkeypatch):
    payload = {"subtype": "success", "result": "e2e4", "usage": {
        "input_tokens": 20, "output_tokens": 50,
        "cache_read_input_tokens": 4000, "cache_creation_input_tokens": 500,
        "cache_creation": {"ephemeral_5m_input_tokens": 300,
                           "ephemeral_1h_input_tokens": 200},
    }}
    process = SimpleNamespace(returncode=0, communicate=AsyncMock(
        return_value=(json.dumps(payload).encode(), b"")))
    monkeypatch.setattr(asyncio, "create_subprocess_exec", AsyncMock(return_value=process))
    player = ClaudeCodePlayer("claude", "claude-opus-4.7")
    assert asyncio.run(player.select_move(chess.Board())) == "e2e4"
    result = player.get_token_usage()
    assert result["prompt_tokens"] == 4520
    assert result["total_tokens"] == 4570
    assert result.get("cached_input_tokens") == 4000
    assert result["cache_creation_input_tokens"] == 500
    assert result["cache_creation_5m_input_tokens"] == 300
    assert result["cache_creation_1h_input_tokens"] == 200
    assert result["chess_prompt_tokens"] > 0
    assert result["runtime_prompt_tokens"] + result["chess_prompt_tokens"] == 4520
    assert result["cache_accounting_known"] is True


def test_missing_codex_cache_is_unknown_and_estimate_cannot_exceed_raw_input():
    player = CodexSubagentPlayer("codex")
    player.last_prompt = "a chess position with many words"
    player._track_usage(player._parse_usage(json.dumps({
        "type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1},
    })))
    usage = player.get_token_usage()
    assert usage.get("cache_accounting_known") is False
    assert usage["chess_prompt_tokens"] == 1
    assert usage["runtime_prompt_tokens"] == 0


def test_cache_writes_follow_reads_and_keep_ttl_split():
    player = CodexSubagentPlayer("codex")
    player.last_prompt = "hello world"
    player._track_usage({
        "prompt_tokens": 100, "completion_tokens": 2,
        "cached_input_tokens": 97, "cache_creation_input_tokens": 3,
        "cache_creation_5m_input_tokens": 1, "cache_creation_1h_input_tokens": 2,
    })
    usage = player.get_token_usage()
    assert usage["chess_prompt_tokens"] == 2
    assert usage["chess_cached_input_tokens"] == 0
    assert usage["chess_cache_creation_input_tokens"] == 2
    assert usage["chess_cache_creation_5m_input_tokens"] == 0
    assert usage["chess_cache_creation_1h_input_tokens"] == 2


def test_direct_api_input_uses_measured_prompt_without_runtime_estimate():
    player = ClaudeCodePlayer("claude", "claude-opus-4.7")
    player.record_chess_usage({
        "prompt_tokens": 120, "completion_tokens": 25,
        "cached_input_tokens": 100, "cache_creation_input_tokens": 5,
    }, "hello", runtime=False)
    usage = player.get_token_usage()
    assert usage["chess_prompt_tokens"] == 120
    assert usage["chess_cached_input_tokens"] == 100
    assert usage["chess_cache_creation_input_tokens"] == 5
    assert usage["runtime_prompt_tokens"] == 0
    assert usage["input_accounting_method"] == "provider_reported_chess_input"


def test_preflight_accounts_for_prompt_actually_sent(monkeypatch):
    player = CodexSubagentPlayer("codex")
    board = chess.Board()
    player._prefetched_response = {
        "fen": board.fen(), "prompt": "hello", "response_text": "e2e4",
        "usage": {"prompt_tokens": 100, "completion_tokens": 4}, "elapsed": 1,
    }
    assert asyncio.run(player.select_move(board)) == "e2e4"
    usage = player.get_token_usage()
    assert usage["chess_prompt_tokens"] == 1
    assert len(usage["input_accounting_requests"]) == 1


def test_tokenizer_unavailable_preserves_legal_move_and_labeled_usage(monkeypatch):
    from llm import token_accounting

    def unavailable():
        raise OSError("Tokenizer vocabulary download unavailable")

    monkeypatch.setattr(token_accounting, "_encoding", unavailable)
    player = CodexSubagentPlayer("codex")
    board = chess.Board()
    player._prefetched_response = {
        "fen": board.fen(), "prompt": "hello", "response_text": "e2e4",
        "usage": {"prompt_tokens": 100, "completion_tokens": 4,
                  "cached_input_tokens": 100, "cache_accounting_known": True},
        "elapsed": 1,
    }
    assert asyncio.run(player.select_move(board)) == "e2e4"
    usage = player.get_token_usage()
    assert usage["prompt_tokens"] == 100
    assert usage["completion_tokens"] == 4
    assert usage["cached_input_tokens"] == 100
    assert usage["chess_prompt_tokens"] == 2
    assert usage["runtime_prompt_tokens"] == 98
    assert usage["chess_cached_input_tokens"] == 2
    assert usage["input_accounting_method"] == "estimated_utf8_quarter_runtime_prefix_cache"
    assert usage["input_accounting_requests"][0]["input_accounting_method"] == usage["input_accounting_method"]
