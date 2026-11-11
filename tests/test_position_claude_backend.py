"""Explicit subscription benchmarks use the game client and bounded concurrency."""

import asyncio

import chess
import pytest

from position_benchmark import run_benchmark as benchmark


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("effort", ["high", None])
def test_claude_positions_use_requested_effort_and_run_serially(monkeypatch, configured, effort):
    constructed, active, peak = [], 0, 0

    class Player:
        def __init__(self, **kwargs):
            constructed.append((type(self).__name__, kwargs))

        async def close(self):
            pass

    class Claude(Player):
        pass

    monkeypatch.setattr(benchmark, "ClaudeCodePlayer", Claude, raising=False)
    monkeypatch.setattr(benchmark, "OpenRouterPlayer", Player)

    async def evaluate(player, position, stockfish, depth):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return benchmark.PositionResult(
            position_idx=0, fen=position["fen"], model_move="e2e4", model_move_san="e4",
            best_move="e2e4", best_move_san="e4", blunder_move="", cpl=0,
            is_legal=True, is_best=True, avoided_blunder=True,
            eval_model=0, eval_best=0, eval_before=0,
        )

    monkeypatch.setattr(benchmark, "test_llm_on_position", evaluate)
    config = {"model_name": "anthropic/claude-fable-5-2", "reasoning_effort": effort}
    if configured:
        config["api"] = "claude_code"
    result = asyncio.run(benchmark.run_benchmark(
        "claude-fable-5.2 (high)", [{"fen": chess.STARTING_FEN}] * 2,
        None, config, depth=30, api_backend="openrouter" if configured else "claude_code",
    ))
    assert result["summary"]["legal_moves"] == 2
    assert [kind for kind, _ in constructed] == ["Claude", "Claude"]
    assert all(kwargs["reasoning_effort"] == (effort or "medium") for _, kwargs in constructed)
    assert all(kwargs["model_name"] == "anthropic/claude-fable-5-2" for _, kwargs in constructed)
    assert peak == 1


@pytest.mark.parametrize("settings", [{"reasoning": False}, {"reasoning_effort": "minimal"}])
def test_claude_rejects_unsupported_reasoning_before_starting_positions(settings):
    with pytest.raises(ValueError, match="Claude Code"):
        asyncio.run(benchmark.run_benchmark(
            "claude-fable-5.2", [], None, {"model_name": "claude-fable-5-2", **settings},
            depth=30, api_backend="claude_code",
        ))


def test_subscription_effort_is_available_to_position_and_continuation_commands():
    configs = benchmark.load_player_configs()
    pid = "claude-fable-5.2 (high)"
    assert pid in configs
    assert configs[pid]["api"] == "claude_code"
    assert configs[pid]["reasoning_effort"] == "high"
    # Retain the provider-qualified identifier used by pricing and metadata;
    # ClaudeCodePlayer normalizes its CLI argument itself.
    assert configs[pid]["model_name"] == "anthropic/claude-fable-5-2"


@pytest.mark.parametrize("configured", [False, True])
def test_continuations_use_same_claude_surface_and_effort(monkeypatch, configured):
    from llm.claude_code_client import ClaudeCodePlayer
    from scripts.run_stability_probe import create_llm_player
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only")
    config = {"model_name": "anthropic/claude-fable-5-2", "reasoning_effort": "high"}
    if configured:
        config["api"] = "claude_code"
    player = create_llm_player("claude-fable-5.2 (high)", config,
                               api_backend="openrouter" if configured else "claude_code",
                               respect_config_api=True)
    try:
        assert isinstance(player, ClaudeCodePlayer)
        assert player.model_name == "claude-fable-5-2"
        assert player.reasoning_effort == "high"
    finally:
        asyncio.run(player.close())
