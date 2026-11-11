"""Regression checks for history-preserving, same-root benchmark evaluation."""

from concurrent.futures import ThreadPoolExecutor
import json

import chess
import chess.engine
import pytest

from position_benchmark.evaluation import ConsistentEvaluator, SCORING_VERSION
from position_benchmark.scoring import ILLEGAL_MOVE_EVAL, illegal_move_cpl


class RootEngine:
    """An engine whose child search intentionally disagrees with its root score."""

    id = {"name": "Root test engine", "author": "tests"}
    options = {
        name: chess.engine.Option(name, kind, default, None, None, None)
        for name, kind, default in (
            ("Threads", "spin", 8), ("Hash", "spin", 512),
            ("Skill Level", "spin", 3), ("UCI_LimitStrength", "check", True),
            ("EvalFile", "string", "embedded-test.nnue"),
            ("Clear Hash", "button", None),
        )
    }

    def __init__(self, baseline=100, selected=25):
        self.baseline = baseline
        self.selected = selected
        self.calls = []
        self.configurations = []

    def configure(self, options):
        self.configurations.append(dict(options))

    def analyse(self, board, limit, *, root_moves=None, game=None):
        self.calls.append((board.copy(stack=True), limit, root_moves, game))
        best = next(iter(board.legal_moves))
        value = self.selected if root_moves else self.baseline
        score = value if isinstance(value, chess.engine.Score) else chess.engine.Cp(value)
        return {"score": chess.engine.PovScore(score, board.turn), "pv": [best]}


def alternative(board):
    return list(board.legal_moves)[1]


def test_selected_move_is_scored_at_same_root_depth_and_history():
    board = chess.Board()
    board.push_uci("e2e4")
    original = board.copy(stack=True)
    engine = RootEngine()
    result = ConsistentEvaluator(engine).score(board, alternative(board))
    assert result["cpl"] == 75
    assert result["eval_before"] == result["eval_best"] == 100
    assert result["eval_model"] == 25
    assert result["scoring_version"] == SCORING_VERSION
    json.dumps(result["scoring_profile"])
    assert [call[0].fen() for call in engine.calls] == [original.fen()] * 2
    assert [call[0].move_stack for call in engine.calls] == [original.move_stack] * 2
    assert [call[1].depth for call in engine.calls] == [30, 30]
    assert engine.calls[1][2] == [alternative(board)]
    assert engine.calls[0][3] is not engine.calls[1][3]
    assert board == original and board.move_stack == original.move_stack


def test_reference_best_move_reuses_baseline_even_if_forced_search_would_disagree():
    board = chess.Board()
    engine = RootEngine(baseline=150, selected=-20)
    result = ConsistentEvaluator(engine).score(board, next(iter(board.legal_moves)))
    assert result["cpl"] == 0
    assert result["is_best"] is True
    assert result["eval_model"] == 150
    assert len(engine.calls) == 1
    assert result["best_move_san"] == board.san(next(iter(board.legal_moves)))


def test_repeated_calls_and_new_evaluators_reuse_persistent_cache(tmp_path):
    board = chess.Board()
    path = tmp_path / "scores.sqlite3"
    engine = RootEngine()
    first = ConsistentEvaluator(engine, cache_path=path).score(board, alternative(board))
    second_engine = RootEngine(baseline=-200, selected=-300)
    evaluator = ConsistentEvaluator(second_engine, cache_path=path)
    assert evaluator.score(board, alternative(board)) == first
    assert evaluator.score(board, alternative(board)) == first
    assert not second_engine.calls


def test_history_and_depth_are_separate_cache_keys(tmp_path):
    board = chess.Board()
    for move in ["g1f3", "g8f6", "f3g1", "f6g8"]:
        board.push_uci(move)
    same_fen_without_history = chess.Board(board.fen())
    engine = RootEngine()
    path = tmp_path / "scores.sqlite3"
    evaluator = ConsistentEvaluator(engine, cache_path=path)
    evaluator.score(board, None)
    evaluator.score(same_fen_without_history, None)
    ConsistentEvaluator(engine, depth=29, cache_path=path).score(board, None)
    assert len(engine.calls) == 3


def test_engine_profile_and_nnue_contents_separate_cache_keys(tmp_path):
    nnue = tmp_path / "weights.nnue"
    nnue.write_bytes(b"first")
    path = tmp_path / "scores.sqlite3"
    engine = RootEngine()
    engine.options = dict(engine.options)
    engine.options["EvalFile"] = chess.engine.Option("EvalFile", "string", str(nnue), None, None, None)
    first = ConsistentEvaluator(engine, cache_path=path).score(chess.Board(), None)
    nnue.write_bytes(b"second")
    second = ConsistentEvaluator(engine, cache_path=path).score(chess.Board(), None)
    assert len(engine.calls) == 2
    assert first["scoring_profile"] != second["scoring_profile"]


@pytest.mark.parametrize("value,expected", [(chess.engine.Mate(3), 9970), (chess.engine.Mate(-4), -9960), (chess.engine.Cp(-125), -125)])
def test_black_perspective_and_existing_mate_conversion(value, expected):
    board = chess.Board()
    board.push_uci("e2e4")
    engine = RootEngine(baseline=value)
    result = ConsistentEvaluator(engine).score(board, None)
    assert result["eval_before"] == expected
    assert result["eval_model"] == ILLEGAL_MOVE_EVAL
    assert result["cpl"] == illegal_move_cpl(expected)


def test_full_strength_and_clean_state_for_each_cache_miss():
    engine = RootEngine()
    evaluator = ConsistentEvaluator(engine)
    board = chess.Board()
    evaluator.score(board, alternative(board))
    config = {key: value for call in engine.configurations for key, value in call.items()}
    assert config["Threads"] == 1
    assert config["Hash"] == 64
    assert config["UCI_LimitStrength"] is False
    assert config["Skill Level"] == 20
    assert "UCI_Elo" not in config
    assert sum("Clear Hash" in call for call in engine.configurations) == 2


def test_sqlite_cache_accepts_concurrent_evaluators(tmp_path):
    path = tmp_path / "scores.sqlite3"
    board = chess.Board()
    def evaluate(_):
        return ConsistentEvaluator(RootEngine(), cache_path=path).score(board, alternative(board))
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(evaluate, range(8)))
    assert all(result == results[0] for result in results)


def test_compact_profile_keeps_full_engine_profile_for_audit():
    import hashlib
    import position_benchmark.evaluation as evaluation

    evaluator = ConsistentEvaluator(RootEngine())
    full = evaluator.engine_profile
    expected_hash = hashlib.sha256(json.dumps(full, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    expected = {"sha256": expected_hash, "engine": "Root test engine", "depth": 30}
    assert full["options"]["Threads"] == 1
    assert full["options"]["Hash"] == 64
    assert full["scoring_version"] == SCORING_VERSION
    assert evaluator.scoring_profile == expected
    assert evaluator.score(chess.Board(), None)["scoring_profile"] == expected
    assert evaluation.scoring_profile_identity(full) == expected
    assert evaluation.scoring_profile_identity(expected) == expected
    assert len(json.dumps(expected)) < len(json.dumps(full))


def test_compact_profile_reuses_cache_keys_written_with_original_full_profile(tmp_path):
    import hashlib
    import sqlite3

    engine = RootEngine()
    cache = tmp_path / "old-cache.sqlite3"
    evaluator = ConsistentEvaluator(engine, cache_path=cache)
    board = chess.Board()
    board.push_uci("e2e4")
    best = next(iter(board.legal_moves))
    selected = alternative(board)
    with sqlite3.connect(cache) as connection:
        for move, value in ((None, 123), (selected, 20)):
            # This is the original version's on-disk key contract, unchanged.
            old_payload = {
                "profile": evaluator._profile(),
                "root_fen": board.root().fen(en_passant="fen"),
                "moves": [played.uci() for played in board.move_stack],
                "fen": board.fen(en_passant="fen"),
                "chess960": board.chess960,
                "variant": type(board).__qualname__,
                "move": move.uci() if move is not None else None,
            }
            old_key = hashlib.sha256(json.dumps(old_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            connection.execute("INSERT INTO position_scores VALUES (?, ?)", (old_key, json.dumps({"eval": value, "best_move": best.uci()})))
    result = evaluator.score(board, selected)
    assert result["eval_before"] == 123
    assert result["cpl"] == 103
    assert result["scoring_profile"]["sha256"]
    assert not engine.calls
