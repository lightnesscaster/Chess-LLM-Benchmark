"""Official entrypoints retain private resume state without paid test calls."""

import asyncio
from pathlib import Path

import pytest

import cli
from engines.random_engine import RandomEngine
from game import live_games
from game.match_scheduler import GameTask, MatchScheduler
from game.pgn_logger import PGNLogger
from game.stats_collector import StatsCollector
from rating.rating_store import RatingStore


@pytest.fixture(autouse=True)
def no_provider_credentials(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)


def manual_args(monkeypatch, *flags):
    captured = []
    original = cli.run_manual_game

    async def capture(args):
        captured.append(args)

    with monkeypatch.context() as patch:
        patch.setattr(cli, "run_manual_game", capture)
        patch.setattr("sys.argv", ["cli.py", "manual", *flags])
        cli.main()
    return original, captured[0]


@pytest.mark.parametrize("extra,expected", [
    (["--games", "2"], "requires --games 1"),
    (["--moves", "e2e4"], "cannot be combined with --moves"),
    (["--no-save"], "cannot be combined with --no-save"),
])
def test_invalid_resume_flags_stop_before_clients(monkeypatch, tmp_path, capsys, extra, expected):
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text("{}")
    run, args = manual_args(monkeypatch, "--resume-checkpoint", str(checkpoint), *extra)
    assert asyncio.run(run(args)) == 1
    assert expected in capsys.readouterr().out


@pytest.mark.parametrize("contents", [None, "not json", "[]"])
def test_unreadable_resume_checkpoint_stops_before_clients(monkeypatch, tmp_path, capsys, contents):
    checkpoint = tmp_path / "checkpoint.json"
    if contents is not None:
        checkpoint.write_text(contents)
    run, args = manual_args(monkeypatch, "--resume-checkpoint", str(checkpoint))
    assert asyncio.run(run(args)) == 1
    assert "cannot read resume checkpoint" in capsys.readouterr().out


def test_clocked_engine_moves_cannot_silently_reset_clock(monkeypatch, capsys):
    run, args = manual_args(monkeypatch, "--black-engine", "--engine-type", "eubos", "--moves", "e2e4")
    assert asyncio.run(run(args)) == 1
    assert "requires --resume-checkpoint" in capsys.readouterr().out


@pytest.mark.parametrize("saved", [False, True])
def test_official_manual_random_game_checkpoints_only_saved_runs(monkeypatch, tmp_path, saved):
    monkeypatch.chdir(tmp_path)
    flags = [] if saved else ["--no-save"]
    run, args = manual_args(monkeypatch, "--white-engine", "--black-engine", "--engine-type", "random", "--api", "codex", "--max-moves", "1", "--local-only", *flags)
    assert asyncio.run(run(args)) == 0
    checkpoints = list(Path("data/checkpoints").glob("*.json"))
    assert len(checkpoints) == int(saved)
    assert len(list(Path("data/results").glob("*.json"))) == int(saved)


def test_scheduler_saved_game_writes_checkpoint(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    scheduler = MatchScheduler.__new__(MatchScheduler)
    scheduler._semaphore = asyncio.Semaphore(1)
    scheduler.verbose = False
    scheduler.max_moves = 1
    scheduler.rating_store = RatingStore(path=str(tmp_path / "ratings.json"))
    scheduler.pgn_logger = PGNLogger(use_firestore=False)
    scheduler.stats_collector = StatsCollector()

    async def update(_result):
        pass

    scheduler._update_ratings = update
    monkeypatch.setattr(live_games, "get_live_game_store", lambda: live_games.LiveGameStore(path=tmp_path / "live", use_firestore=False))
    result = asyncio.run(scheduler.run_single_game(GameTask(RandomEngine("white", 400), RandomEngine("black", 400), 1, 1)))
    assert result is not None
    assert (Path("data/checkpoints") / f"{result.game_id}.json").exists()


def test_official_resume_keeps_completed_uuid_without_replaying(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    run, args = manual_args(
        monkeypatch, "--white-engine", "--black-engine", "--engine-type", "random",
        "--api", "codex", "--max-moves", "1", "--local-only",
    )
    assert asyncio.run(run(args)) == 0
    checkpoint, = Path("data/checkpoints").glob("*.json")
    original_result, = Path("data/results").glob("*.json")
    original_text = original_result.read_text()

    def unexpected_move(*_args, **_kwargs):
        raise AssertionError("a completed game must not be replayed")

    monkeypatch.setattr(RandomEngine, "select_move", unexpected_move)
    args.resume_checkpoint = checkpoint
    assert asyncio.run(run(args)) == 0
    assert list(Path("data/results").glob("*.json")) == [original_result]
    assert original_result.read_text() == original_text


def test_llm_only_move_history_continuation_remains_available(monkeypatch, tmp_path, capsys):
    from llm.base_llm import BaseLLMPlayer

    class ResigningPlayer(BaseLLMPlayer):
        def __init__(self, player_id, model_name, reasoning_effort):
            super().__init__(player_id, model_name)

        async def select_move(self, board, is_retry=False, last_move_illegal=None):
            assert board.peek().uci() == "e2e4"
            return "resign"

        async def close(self):
            pass

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "CodexSubagentPlayer", ResigningPlayer)
    run, args = manual_args(
        monkeypatch, "--white-model", "white", "--black-model", "black",
        "--api", "codex", "--no-save", "--moves", "e2e4",
    )
    assert asyncio.run(run(args)) == 0
    output = capsys.readouterr().out
    assert "Result: white (resignation)" in output
    assert "1. e4 1-0" in output
    assert not Path("data/checkpoints").exists()
