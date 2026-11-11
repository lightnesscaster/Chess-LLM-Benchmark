"""Guard held-out recalibration against target and family leakage."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from position_benchmark import calibration_validation as cv

POLICY = json.loads(Path("position_benchmark/validation/2026-11-11-corrected-cpl-calibration-plan.json").read_text())


def rows_fixture():
    rows = []
    for group in range(9):
        for effort in range(2):
            features = [3.0 + group / 5, 5.0 + group * 5 + effort, 30.0 + group * 7]
            baseline = 1298.57 - 200.43 * features[0] + 15.39 * features[1] + 5.85 * features[2]
            rows.append(dict(player_id=f"family{group} ({effort})", family=f"family{group}",
                             lab=f"lab{group % 3}", games=40, game_only_rd=90,
                             target=baseline + 120, core_features=features,
                             blunder_features=None, game_like_features=None,
                             ready_stability_cap=None, baseline_prediction=baseline))
    return rows


def test_grouping_keeps_efforts_and_new_lines_together():
    assert cv.model_family("gpt-6-astra (xhigh)") == "gpt-6-astra"
    assert cv.model_family("gpt-6.1-sol (low)") == "gpt-6.1-sol"
    assert cv.model_family("gemini-4-carbon (high)") == "gemini-4-carbon"
    assert cv.model_family("claude-opus-4.7 (high)") == cv.model_family("claude-sonnet-4.6")
    assert cv.model_lab("o3 (high)") == cv.model_lab("gpt-6-astra") == "OpenAI"
    assert cv.model_lab("gemma-3") == cv.model_lab("gemini-4-argon") == "Google"


def test_cap_order_and_deadband_match_production():
    row = rows_fixture()[0]
    row.update(core_features=[0, 0, 0], blunder_features=[0, 0, 0],
               game_like_features=[1, 0, 0], ready_stability_cap=1100)
    # Core=1298.57, blunder is tied, game=1098.14, stability is within deadband.
    assert cv.predict_core(row, cv.LEGACY_COEFFICIENTS) == pytest.approx(1098.14)
    row["ready_stability_cap"] = 948.14
    assert cv.predict_core(row, cv.LEGACY_COEFFICIENTS) == pytest.approx(1098.14)
    row["ready_stability_cap"] = 948.13
    assert cv.predict_core(row, cv.LEGACY_COEFFICIENTS) == pytest.approx(948.13)


def test_training_only_weighted_normalization_and_coefficient_constraints():
    rows = rows_fixture()[:4]
    rows[0]["target"] = 100000
    model = cv.fit_candidate(rows, "core-ridge-1")
    assert model["coefficients"][1] <= 0
    assert model["coefficients"][2] >= 0
    assert model["coefficients"][3] >= 0
    assert model["normalization"]["mean"] == pytest.approx(np.mean([r["core_features"] for r in rows], axis=0))
    singleton = copy.deepcopy(rows[0])
    singleton.update(player_id="extra", family="extra", core_features=[90, 90, 90])
    weighted_model = cv.fit_candidate(rows + [singleton], "core-ridge-10")
    expected = (np.mean([r["core_features"] for r in rows[:2]], axis=0) +
                np.mean([r["core_features"] for r in rows[2:]], axis=0) + 90) / 3
    assert weighted_model["normalization"]["mean"] == pytest.approx(expected)
    assert cv.fit_candidate(rows, "core-ridge-1") == model


def test_nested_selection_excludes_carbon_and_heldout_targets():
    rows = rows_fixture()
    carbon = copy.deepcopy(rows[0])
    carbon.update(player_id="gemini-4-carbon (low)", family="gemini-4-carbon", target=1e8)
    policy = copy.deepcopy(POLICY)
    policy["bootstrap"]["resamples"] = 50
    result = cv.analyze(rows + [carbon], policy)
    assert result["cohort"]["configurations"] == 18
    assert result["carbon_diagnostics"][0]["player_id"] == carbon["player_id"]
    for fold in result["family_folds"]:
        assert set(fold["train_families"]).isdisjoint(fold["test_families"])
        assert not any("carbon" in p for p in fold["train_player_ids"])
        assert fold["selected_candidate"] == "offset"
    changed = copy.deepcopy(rows)
    changed[0]["target"] += 100000
    changed[1]["target"] += 100000
    other = cv.analyze(changed, policy)
    before = next(f for f in result["family_folds"] if f["test_families"] == ["family0"])
    after = next(f for f in other["family_folds"] if f["test_families"] == ["family0"])
    assert before["selected_model"] == after["selected_model"]
    assert result["uncertainty"]["method"] == "fixed"
    assert result["recommendation"]["mean_action"] == "retain"


def test_bootstrap_is_deterministic_and_uses_production_selected_method():
    rows = rows_fixture()
    for row in rows:
        row["target"] += 300
    policy = copy.deepcopy(POLICY)
    policy["bootstrap"]["resamples"] = 100
    first = cv.analyze(rows, policy)
    second = cv.analyze(rows, policy)
    assert first["bootstrap"] == second["bootstrap"]
    assert first["uncertainty"]["method"] == "fixed"
    assert first["uncertainty"]["metrics"]["row_rmse"] == pytest.approx(420)
    assert first["uncertainty"]["recommended_rd"] == 166  # Too few configurations.


def test_build_rows_matches_production_caps_and_conservative_stability_legality():
    from rating.rating_store import PositionBenchmarkInputs
    from position_benchmark.predictions import (
        CURRENT_BENCHMARK_VERSION, CURRENT_STABILITY_PROBE_VERSION,
        CURRENT_STABILITY_SELECTION_POLICY, CURRENT_STABILITY_POSITION_INDICES,
        predict_rating_from_model_data_with_supplement,
    )
    positions = [{"fen": f"position-{i}", "best_move": "e2e4", "type": "equal"} for i in range(50)]
    def record(size, illegal=0, cpl=10000):
        return {"summary": {"scoring_version": "root-reference-v1", "scoring_profile": {"test": True}},
                "results": [dict(position_idx=i, fen=positions[i]["fen"], model_move="e2e4",
                                 cpl=cpl, is_legal=i >= illegal, position_benchmark_version=CURRENT_BENCHMARK_VERSION,
                                 prompt_history_replay=True, stockfish_depth=30,
                                 scoring_version="root-reference-v1", scoring_profile={"test": True}) for i in range(size)]}
    core, game = record(50, illegal=1, cpl=100), record(48)
    stability = {"summary": dict(scoring_version="root-reference-v1", scoring_profile={"test": True},
                                 stability_probe_version=CURRENT_STABILITY_PROBE_VERSION,
                                 position_selection_policy=CURRENT_STABILITY_SELECTION_POLICY,
                                 selected_position_indices=list(CURRENT_STABILITY_POSITION_INDICES),
                                 attempted_positions=8, model_scored_moves=40, model_attempts=40,
                                 model_legal_moves=36, model_legal_pct=90, model_forfeits=0,
                                 score_depth=30)}
    inputs = PositionBenchmarkInputs({"gpt-6-astra (high)": core}, positions, None, None,
                                    {"gpt-6-astra (high)": game}, positions[:48],
                                    {"gpt-6-astra (high)": stability})
    rows = cv.build_rows(inputs, {"gpt-6-astra (high)": dict(rating=800, games_played=10, rating_deviation=150)})
    assert len(rows) == 1
    assert rows[0]["game_like_features"][0] == pytest.approx(np.log1p(5000))
    assert rows[0]["game_like_features"][2] == pytest.approx(8.047373696247694)
    assert rows[0]["baseline_prediction"] == pytest.approx(predict_rating_from_model_data_with_supplement(
        core, positions, game_like_model_data=game, game_like_positions=positions[:48], stability_probe_model_data=stability))


def test_promotion_uses_selected_procedure_errors_for_rd():
    rows = rows_fixture()
    extra = copy.deepcopy(rows)
    for row in extra:
        row["player_id"] += " extra"
    policy = copy.deepcopy(POLICY)
    policy["bootstrap"]["resamples"] = 50
    result = cv.analyze(rows + extra, policy)
    assert result["mean_gate"]["passed"]
    assert result["recommendation"]["mean_action"] == "promote"
    assert result["recommendation"]["production_model"]["kind"] == "offset"
    assert result["uncertainty"]["method"] == "adaptive"
    assert result["uncertainty"]["metrics"]["row_rmse"] < 1e-8
    assert result["uncertainty"]["recommended_rd"] == 166


def test_missing_stress_or_sensitivity_evidence_cannot_pass_gates():
    rows = rows_fixture()
    for row in rows:
        row.update(lab="same-provider", games=8, game_only_rd=190)
    policy = copy.deepcopy(POLICY)
    policy["bootstrap"]["resamples"] = 50
    result = cv.analyze(rows, policy)
    assert result["provider_folds"] == []
    assert not result["mean_gate"]["checks"]["provider_stress"]
    assert not result["mean_gate"]["checks"]["reliable_subset"]
    assert not result["mean_gate"]["checks"]["mature_subset"]


def test_cli_writes_auditable_artifacts_without_touching_input(tmp_path):
    import subprocess
    import sys
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps(rows_fixture()))
    before = rows.read_bytes()
    policy = tmp_path / "policy.json"
    small_policy = copy.deepcopy(POLICY)
    small_policy["bootstrap"]["resamples"] = 50
    policy.write_text(json.dumps(small_policy))
    output, report = tmp_path / "analysis.json", tmp_path / "report.md"
    subprocess.run([sys.executable, "scripts/validate_corrected_calibration.py", "--rows", str(rows),
                    "--policy", str(policy), "--output", str(output), "--report", str(report)], check=True,
                   capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["recommendation"]["mean_action"] == "retain"
    assert len(result["source_hashes"]) == 2
    assert "FAIL" in report.read_text()
    assert rows.read_bytes() == before


@pytest.mark.parametrize("tampered", ["inputs", "targets", "policy"])
def test_cli_rejects_tampered_frozen_manifest_before_writing(tmp_path, tampered):
    import hashlib
    import subprocess
    import sys
    paths = {name: tmp_path / f"{name}.json" for name in ("inputs", "targets", "policy")}
    for path in paths.values():
        path.write_text("{}")
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"inputs_sha256": digest(paths["inputs"]),
                                    "game_only_target_sha256": digest(paths["targets"]),
                                    "files": {str(paths["policy"]): digest(paths["policy"])}}))
    paths[tampered].write_text('{"tampered": true}')
    output, report = tmp_path / "analysis.json", tmp_path / "analysis.md"
    command = [sys.executable, "scripts/validate_corrected_calibration.py", "--manifest", str(manifest),
               "--output", str(output), "--report", str(report)]
    for name, path in paths.items():
        command += [f"--{name}", str(path)]
    completed = subprocess.run(command, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "Manifest hash mismatch" in completed.stderr
    assert not output.exists()
    assert not report.exists()


def test_cli_rejects_tampered_public_matrix_hash(tmp_path):
    import subprocess
    import sys
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps(rows_fixture()))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"matrix_sha256": "0" * 64}))
    completed = subprocess.run([sys.executable, "scripts/validate_corrected_calibration.py", "--rows", str(rows),
                                "--manifest", str(manifest), "--output", str(tmp_path / "out.json"),
                                "--report", str(tmp_path / "out.md")], capture_output=True, text=True)
    assert completed.returncode != 0
    assert "Manifest hash mismatch" in completed.stderr
    assert not (tmp_path / "out.json").exists()
