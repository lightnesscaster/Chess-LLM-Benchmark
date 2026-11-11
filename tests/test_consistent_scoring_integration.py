"""Regression coverage for the consumers of consistent root scoring."""

import asyncio
from copy import deepcopy

import chess
import chess.engine

from position_benchmark import run_benchmark as benchmark
from scripts import run_stability_probe as probe


class RootEngine:
    id = {"name": "integration-root-engine", "author": "tests"}
    options = {}

    def __init__(self):
        self.calls = []

    def configure(self, options):
        pass

    def analyse(self, board, limit, *, root_moves=None, **kwargs):
        self.calls.append(board.copy(stack=True))
        move = (root_moves or [next(iter(board.legal_moves))])[0]
        return {"score": chess.engine.PovScore(chess.engine.Cp(70), board.turn), "pv": [move]}

    def quit(self):
        pass


def position_after_e4():
    board = chess.Board()
    board.push_uci("e2e4")
    return {"fen": board.fen(), "move_history": ["e2e4"], "best_move": "e7e5", "best_move_san": "e5", "eval_before": 900}


def test_static_engine_replays_history_and_ignores_stale_baseline():
    class Player:
        def select_move(self, board):
            assert [m.uci() for m in board.move_stack] == ["e2e4"]
            return next(iter(board.legal_moves))

    engine = RootEngine()
    result = benchmark.test_engine_on_position(Player(), position_after_e4(), engine)
    assert result.cpl == 0
    assert result.eval_before == result.eval_model == 70
    assert result.is_best
    assert result.scoring_version
    assert result.scoring_profile
    assert all([m.uci() for m in b.move_stack] == ["e2e4"] for b in engine.calls)


def test_static_llm_prepares_reference_before_model_and_preserves_usage():
    engine = RootEngine()

    class Player:
        allow_resignation = False
        rating_context = None
        prompt_tokens = 0
        completion_tokens = 0

        async def select_move(self, board, **kwargs):
            assert engine.calls, "reference must be fixed before evaluating the response"
            self.prompt_tokens += 15
            self.completion_tokens += 4
            return next(iter(board.legal_moves)).uci()

    result = asyncio.run(benchmark.test_llm_on_position(Player(), position_after_e4(), engine))
    assert result.is_best and result.cpl == 0
    assert result.eval_before == 70
    assert result.prompt_tokens == result.initial_prompt_tokens == 15
    assert result.completion_tokens == result.initial_completion_tokens == 4
    assert result.scoring_version and result.scoring_profile


def test_continuation_uses_pgn_setup_and_full_root_history():
    pgn = '[SetUp "1"]\n[FEN "8/8/8/8/8/8/4K3/6k1 w - - 0 1"]\n[Result "*"]\n\n1. Kd3 Kf2 2. Ke4 *\n'
    engine = RootEngine()
    model, opponent = probe.score_continuation_moves_from_pgn(pgn, model_side=chess.WHITE, pre_move_count=1, stockfish=engine, depth=30)
    assert len(model) == len(opponent) == 1
    assert model[0]["continuation_ply"] == 2
    assert opponent[0]["continuation_ply"] == 1
    assert all(score["cpl"] == 0 for score in model + opponent)
    assert all(score["scoring_version"] and score["scoring_profile"] for score in model + opponent)
    assert {len(b.move_stack) for b in engine.calls} == {1, 2}
    assert all(b.root().fen() == '8/8/8/8/8/8/4K3/6k1 w - - 0 1' for b in engine.calls)


def test_scoring_summary_clears_mixed_or_missing_provenance():
    rows = [{"scoring_version": "version-a", "scoring_profile": {"depth": 30}}, {"scoring_version": "version-a", "scoring_profile": {"depth": 30}}]
    summary = {}
    benchmark.stamp_scoring_summary(summary, rows)
    assert summary == rows[0]
    mixed = deepcopy(rows)
    mixed[1].pop("scoring_version")
    benchmark.stamp_scoring_summary(summary, mixed)
    assert "scoring_version" not in summary and "scoring_profile" not in summary


def test_illegal_first_response_uses_current_reference_and_keeps_retry_usage():
    engine = RootEngine()

    class Player:
        allow_resignation = False
        rating_context = None
        prompt_tokens = 0
        completion_tokens = 0
        responses = ["invalid", "e7e5"]

        async def select_move(self, board, **kwargs):
            self.prompt_tokens += 10
            self.completion_tokens += 2
            return self.responses.pop(0)

    result = asyncio.run(benchmark.test_llm_on_position(Player(), position_after_e4(), engine))
    assert not result.is_legal and not result.is_best
    assert result.cpl == 5070
    assert result.eval_before == 70
    assert result.retry_is_legal and result.retry_move == "e7e5"
    assert result.prompt_tokens == 20 and result.retry_prompt_tokens == 10
    assert result.completion_tokens == 4 and result.retry_completion_tokens == 2


def test_saved_continuation_rescore_preserves_responses_and_recomputes_summary(monkeypatch):
    row = {
        "pgn": '[Result "*"]\n\n1. e4 e5 *\n', "model_side": "white", "pre_moves": 0,
        "model_legal_moves": 1, "model_illegal_attempts": 0, "model_forfeited": False,
        "termination": "max_moves", "probe_plies_played": 2, "illegal_move_details": [],
        "raw_response": "e2e4", "prompt_tokens": 70, "completion_tokens": 4,
        "model_move_scores": [{"cpl": 999}], "opponent_move_scores": [],
    }
    results = {"model": {"summary": {"model_avg_cpl": 999}, "results": [deepcopy(row)]}}
    monkeypatch.setattr(probe.chess.engine.SimpleEngine, "popen_uci", lambda path: RootEngine())
    def forbid_model_calls(*args, **kwargs):
        raise AssertionError("rescoring must not create a model client")
    monkeypatch.setattr(probe, "create_llm_player", forbid_model_calls)
    assert probe.rescore_existing_results(results, player_ids={"model"}, stockfish_path="unused", score_depth=30, workers=1) == (1, 1)
    saved = results["model"]["results"][0]
    for field in row.keys() - {"model_move_scores", "opponent_move_scores"}:
        assert saved[field] == row[field]
    assert results["model"]["summary"]["model_avg_cpl"] == 0
    assert results["model"]["summary"]["scoring_version"]
    assert results["model"]["summary"]["scoring_profile"]
    assert results["model"]["summary"]["scoring_engine_profile"]["engine"]["name"] == "integration-root-engine"
    assert "scoring_engine_profile" not in saved


def first_turn_forfeit_row():
    return {
        "pgn": '[Result "0-1"]\n\n0-1\n', "model_side": "white", "pre_moves": 0,
        "model_legal_moves": 0, "model_illegal_attempts": 2, "model_forfeited": True,
        "termination": "forfeit_illegal_move", "probe_plies_played": 0,
        "illegal_move_details": [{"move_number": 1}, {"move_number": 1}],
        "model_move_scores": [], "opponent_move_scores": [],
    }


def test_rescore_all_first_turn_forfeits_stamps_rows_and_summary_without_search(monkeypatch):
    engine = RootEngine()
    monkeypatch.setattr(probe.chess.engine.SimpleEngine, "popen_uci", lambda path: engine)
    rows = [first_turn_forfeit_row() for _ in range(8)]
    results = {"model": {"summary": {}, "results": rows}}
    assert probe.rescore_existing_results(results, player_ids={"model"}, stockfish_path="unused", score_depth=30, workers=1) == (1, 8)
    assert engine.calls == []
    summary = results["model"]["summary"]
    assert summary["model_forfeit_pct"] == 100
    assert summary["model_scored_moves"] == 0
    assert summary["scoring_version"]
    assert summary["scoring_profile"]["depth"] == 30
    assert all(row["scoring_version"] == summary["scoring_version"] for row in rows)
    assert all(row["scoring_profile"] == summary["scoring_profile"] for row in rows)


def test_new_all_forfeit_probe_records_provenance_without_search(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    game_result = SimpleNamespace(
        termination="forfeit_illegal_move", winner="black", moves=0,
        illegal_moves_white=2, illegal_moves_black=0,
        total_moves_white=0, total_moves_black=0,
        tokens_white={}, tokens_black={},
        illegal_move_details=[{"side": "white", "move_number": 1}, {"side": "white", "move_number": 1}],
    )
    monkeypatch.setattr(probe, "create_llm_player", lambda *args, **kwargs: SimpleNamespace(close=AsyncMock()))
    monkeypatch.setattr(probe.GameRunner, "play_game", AsyncMock(return_value=(game_result, '[Result "0-1"]\n\n0-1\n')))
    engine = RootEngine()
    record = asyncio.run(probe.run_probe_for_player(
        "model", {}, [(i, {"fen": chess.STARTING_FEN}) for i in range(8)],
        probe_plies=8, api_backend="openrouter", random_seed=1, verbose=False,
        respect_config_api=False, stockfish=engine, score_depth=30,
    ))
    assert engine.calls == []
    assert record["summary"]["model_forfeit_pct"] == 100
    assert record["summary"]["scoring_version"]
    assert all(row["scoring_profile"]["depth"] == 30 for row in record["results"])
    assert record["summary"]["scoring_engine_profile"]["engine"]["name"] == "integration-root-engine"
    assert all("scoring_engine_profile" not in row for row in record["results"])


def test_static_summary_keeps_full_engine_audit_out_of_rows():
    record = asyncio.run(benchmark.run_benchmark(
        "random", [position_after_e4()], RootEngine(), {"type": "random"}, depth=30,
    ))
    audit = record["summary"]["scoring_engine_profile"]
    assert audit["engine"]["name"] == "integration-root-engine"
    assert audit["options"] == {}
    assert record["summary"]["scoring_profile"]["engine"] == "integration-root-engine"
    assert all("scoring_engine_profile" not in row for row in record["results"])


def test_full_engine_audit_is_removed_when_summary_profile_disagrees():
    evaluator = benchmark.ConsistentEvaluator(RootEngine(), depth=30)
    summary = {"scoring_profile": {"sha256": "different", "engine": "other", "depth": 30}, "scoring_engine_profile": {"stale": True}}
    benchmark.stamp_scoring_engine_profile(summary, evaluator.engine_profile)
    assert "scoring_engine_profile" not in summary
