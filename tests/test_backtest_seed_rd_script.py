import json

import pytest

from scripts import backtest_seed_rd as audit


def test_manifest_cannot_omit_required_input_hashes(tmp_path):
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest entries"):
        audit.evaluate(tmp_path)


def test_freeze_refuses_replacement_and_detects_input_tampering(tmp_path, monkeypatch):
    root, snapshot = tmp_path / "repo", tmp_path / "snapshot"
    root.mkdir()
    snapshot.mkdir()
    (root / "input.txt").write_text("original")
    (snapshot / "results_snapshot.json").write_text("[]")
    monkeypatch.setattr(audit, "ROOT", root)
    monkeypatch.setattr(audit, "INPUTS", ("input.txt",))
    audit.freeze(snapshot)
    with pytest.raises(ValueError, match="already exists"):
        audit.freeze(snapshot)
    (snapshot / "inputs/input.txt").write_text("changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        audit.evaluate(snapshot)


def test_prospective_cohort_requires_locked_zero_game_prediction(tmp_path, monkeypatch):
    def save(path, data):
        target = tmp_path / "inputs" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data))

    entry = {"prediction_locked": True, "production_prediction": 1500,
             "recorded_at": "2026-01-01T00:00:00Z", "family": "f", "lab": "l",
             "eligibility": {"prospective_holdout": True, "game_snapshot": {"games_played": 0}}}
    save("data/model_publish_dates.json", {
        "sol": {"model_id": "openai/gpt-5.6-sol"},
        "terra": {"model_id": "openai/gpt-5.6-terra"},
    })
    save("position_benchmark/results/stability_cap_shadow.json", {"entries": {
        "valid": entry, "unlocked": {**entry, "prediction_locked": False},
        "already_played": {**entry, "eligibility": {
            "prospective_holdout": True, "game_snapshot": {"games_played": 1}}},
    }})
    for panel in ("core", "game_like", "optional_blunder", "stability"):
        save(f"position_benchmark/results/{panel}.json", {})
    save("position_benchmark/results/core.json", {"sol": {}, "terra": {}})
    monkeypatch.setattr(audit, "predict_rating_from_model_data_with_supplement", lambda *a, **k: 1500)
    for panel in ("core_equal_50", "game_like_48", "optional_blunder_25"):
        save(f"position_benchmark/panels/{panel}.json", {"positions": []})
    path = tmp_path / "inputs/config/benchmark.yaml"
    path.parent.mkdir(parents=True)
    path.write_text("engines: []\n")
    prospective, retrospective, _, _ = audit.load_seeds(tmp_path)
    assert set(prospective) == {"valid"}
    assert {s["family"] for s in retrospective.values()} == {"openai/gpt-5.6"}
