"""Saved-response rescoring keeps response evidence immutable and resumes safely."""

import copy
import json

import chess
import pytest

from scripts import rescore_position_results as module


class FakeEvaluator:
    scoring_profile = {"engine": "fake", "depth": 30}

    def __init__(self, calls):
        self.calls = calls

    def score(self, board, move):
        self.calls.append((len(board.move_stack), move))
        best = chess.Move.from_uci("e2e4")
        return {
            "best_move": best.uci(), "best_move_san": "e4",
            "eval_before": 20, "eval_best": 20,
            "eval_model": 20 if move == best else -5000,
            "cpl": 0 if move == best else 5020, "is_best": move == best,
            "scoring_version": module.SCORING_VERSION,
            "scoring_profile": self.scoring_profile,
        }


def fixture_data(count=50):
    positions = [
        {"position_id": f"test-{i}", "fen": chess.STARTING_FEN, "type": "equal"}
        for i in range(count)
    ]
    rows = [
        {
            "position_idx": i, "position_id": p["position_id"], "fen": p["fen"],
            "model_move": "e2e4", "model_move_san": "e4", "is_legal": True,
            "is_best": False, "avoided_blunder": True, "cpl": 150,
            "position_benchmark_version": "history-replay-v2",
            "prompt_history_replay": True, "stockfish_depth": 30,
            "raw_response": "e2e4\n", "prompt_tokens": 15, "completion_tokens": 3,
            "retry_attempted": False, "move_rank": 2, "top3": True,
            "top5": True, "reciprocal_rank": 0.5,
        }
        for i, p in enumerate(positions)
    ]
    rows[0].update(model_move="bad", model_move_san="ILLEGAL", is_legal=False,
                   retry_attempted=True, retry_move="e2e4", retry_is_legal=True,
                   retry_prompt_tokens=24, raw_response="bad")
    record = {"results": rows, "summary": {"run_actual_api_cost": 1.25,
              "equal": {"avg_cpl": 150, "top3_pct": 100}},
              "token_usage": {"prompt": 750, "completion": 150}, "extra": "keep"}
    legacy = copy.deepcopy(record)
    legacy["results"][0].pop("prompt_history_replay")
    return positions, {"ready": record, "legacy": legacy}


def run(tmp_path, monkeypatch, positions, data, calls, **kwargs):
    source = tmp_path / "input.json"
    source.write_text(json.dumps(data))
    return module.rescore_file(
        panel="core" if len(positions) == 50 else "game_like",
        input_path=source, output_path=tmp_path / "corrected.json", positions=positions,
        evaluator_factory=lambda: FakeEvaluator(calls), workers=2, **kwargs,
    )


@pytest.mark.parametrize("count", [50, 48])
def test_rescores_ready_only_preserving_response_evidence(tmp_path, monkeypatch, count):
    positions, data = fixture_data(count)
    original = copy.deepcopy(data)
    calls = []
    result = run(tmp_path, monkeypatch, positions, data, calls)
    assert data == original
    assert result["legacy"] == original["legacy"]
    assert len(calls) == count
    assert calls[0][1] is None
    for before, after in zip(original["ready"]["results"], result["ready"]["results"]):
        for field in before.keys() - module.SCORING_FIELDS:
            assert after[field] == before[field]
        assert after["rank_metadata_source"] == "screening-multipv"
    summary = result["ready"]["summary"]
    assert summary["avg_cpl"] == pytest.approx(5020 / count)
    assert summary["avg_cpl_legal"] == summary["median_cpl"] == 0
    assert summary["best_pct"] == pytest.approx((count - 1) / count * 100)
    assert summary["equal"]["avg_cpl"] == summary["avg_cpl"]
    assert summary["equal"]["top3_pct"] == 100
    assert summary["run_actual_api_cost"] == 1.25
    assert summary["scoring_version"] == module.SCORING_VERSION
    assert result["ready"]["token_usage"] == original["ready"]["token_usage"]


def test_resume_requires_input_and_profile_fingerprints(tmp_path, monkeypatch):
    positions, data = fixture_data()
    calls = []
    run(tmp_path, monkeypatch, positions, data, calls)
    assert len(calls) == 50
    run(tmp_path, monkeypatch, positions, data, calls)
    assert len(calls) == 50
    data["ready"]["results"][1]["raw_response"] += " "
    run(tmp_path, monkeypatch, positions, data, calls)
    assert len(calls) == 100
    monkeypatch.setattr(FakeEvaluator, "scoring_profile", {"engine": "new", "depth": 30})
    run(tmp_path, monkeypatch, positions, data, calls)
    assert len(calls) == 150


def test_legal_flag_does_not_allow_illegal_saved_move(tmp_path, monkeypatch):
    positions, data = fixture_data()
    data["ready"]["results"][1]["model_move"] = "e2e5"
    with pytest.raises(ValueError, match="legal"):
        run(tmp_path, monkeypatch, positions, data, [])
    assert not (tmp_path / "corrected.json").exists()


def test_stable_identity_must_match_fen(tmp_path, monkeypatch):
    positions, data = fixture_data()
    data["ready"]["results"][1]["position_id"] = "unknown"
    with pytest.raises(ValueError, match="position_id"):
        run(tmp_path, monkeypatch, positions, data, [])


def test_cpu_only_path_never_constructs_model_clients(tmp_path, monkeypatch):
    from position_benchmark import run_benchmark
    def forbidden(*args, **kwargs):
        raise AssertionError("rescoring constructed an LLM client")
    for name in ["OpenRouterPlayer", "OpenRouterCompletionPlayer", "GeminiPlayer",
                 "CodexSubagentPlayer", "ClaudeCodePlayer"]:
        monkeypatch.setattr(run_benchmark, name, forbidden)
    positions, data = fixture_data()
    run(tmp_path, monkeypatch, positions, data, [])


def test_refuses_in_place_output(tmp_path):
    source = tmp_path / "results.json"
    source.write_text("{}")
    with pytest.raises(ValueError, match="isolated"):
        module.rescore_file(panel="core", input_path=source, output_path=source)


def test_replays_full_history_and_keeps_illegal_first_attempt(tmp_path, monkeypatch):
    positions, data = fixture_data()
    board = chess.Board()
    board.push_uci("e2e4")
    positions[0].update(fen=board.fen(), move_history=["e2e4"])
    for record in data.values():
        record["results"][0]["fen"] = board.fen()
    calls = []
    output = run(tmp_path, monkeypatch, positions, data, calls)
    assert calls[0] == (1, None)
    row = output["ready"]["results"][0]
    assert row["is_legal"] is False
    assert row["model_move"] == "bad"
    assert row["retry_move"] == "e2e4"
    assert row["cpl"] == 5020


def test_each_model_checkpoint_survives_failure(tmp_path, monkeypatch):
    positions, data = fixture_data()
    data["second"] = copy.deepcopy(data["ready"])
    source = tmp_path / "input.json"
    output = tmp_path / "corrected.json"
    source.write_text(json.dumps(data))
    calls = []
    class FailingEvaluator(FakeEvaluator):
        def score(self, board, move):
            if len(self.calls) == 50:
                raise RuntimeError("engine unavailable")
            return super().score(board, move)
    with pytest.raises(RuntimeError, match="engine unavailable"):
        module.rescore_file(panel="core", input_path=source, output_path=output,
                            positions=positions, workers=1,
                            evaluator_factory=lambda: FailingEvaluator(calls))
    checkpoint = json.loads(output.read_text())
    assert checkpoint["ready"]["summary"]["scoring_version"] == module.SCORING_VERSION
    assert checkpoint["second"] == data["second"]
    assert checkpoint["legacy"] == data["legacy"]
    completed = []
    module.rescore_file(panel="core", input_path=source, output_path=output,
                        positions=positions, workers=1,
                        evaluator_factory=lambda: FakeEvaluator(completed))
    assert len(completed) == 50


def test_resume_rejects_modified_corrected_output(tmp_path, monkeypatch):
    positions, data = fixture_data()
    calls = []
    run(tmp_path, monkeypatch, positions, data, calls)
    output_path = tmp_path / "corrected.json"
    output = json.loads(output_path.read_text())
    output["ready"]["results"][1]["model_move"] = "d2d4"
    output_path.write_text(json.dumps(output))
    run(tmp_path, monkeypatch, positions, data, calls)
    assert len(calls) == 100
    assert json.loads(output_path.read_text())["ready"]["results"][1]["model_move"] == "e2e4"


def test_full_engine_profile_is_recorded_once_in_summary(tmp_path, monkeypatch):
    positions, data = fixture_data()
    audit_profile = {"engine": {"name": "fake"}, "options": {"Hash": 64}, "depth": 30}
    monkeypatch.setattr(FakeEvaluator, "engine_profile", audit_profile, raising=False)
    result = run(tmp_path, monkeypatch, positions, data, [])
    record = result["ready"]
    assert record["summary"]["scoring_engine_profile"] == audit_profile
    assert all("scoring_engine_profile" not in row for row in record["results"])
    assert "scoring_engine_profile" not in record["summary"]["equal"]
    assert "scoring_engine_profile" not in result["legacy"]["summary"]
