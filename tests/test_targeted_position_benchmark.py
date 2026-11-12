import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from position_benchmark import run_production as production
from position_benchmark.acquisition import load_acquisition_policy


@pytest.fixture(autouse=True)
def isolated_player_locks(monkeypatch, tmp_path):
    monkeypatch.setattr(production, "ROOT", tmp_path)


def launch(monkeypatch, *, measured=(), published=(), defer_after_core=False,
           full_audit=False, applied=True, stalls=False, replay_succeeds=False, partial=(), revoke_after_replay=False):
    policy = load_acquisition_policy()
    names = [panel.name for panel in policy.panels]
    measured, published = set(measured), set(published)
    actions = []

    def snapshot(player, policy):
        deferred = defer_after_core and "core" in published and not any(n in measured for n in names[1:])
        return {
            "player": player, "policy": policy.version, "published": sorted(published),
            "pending": [] if deferred else [n for n in names if n not in measured],
            "source": "model (low)" if deferred else None,
            "generation": 4, "applied_generation": 4 if applied else 3,
            "receipt": {"status": "applied", "generation": 4}, "rating": 1950.4,
        }

    def resume_rating_update():
        nonlocal applied, defer_after_core
        actions.append(("replay", "saved"))
        applied = replay_succeeds
        if revoke_after_replay:
            defer_after_core = False

    def run_panel(panel, player, **kwargs):
        actions.append(("measure", panel.name))
        if not stalls:
            measured.add(panel.name)
            published.add(panel.name)

    def sync_saved(panel, player, **kwargs):
        actions.append(("sync", panel.name))
        published.add(panel.name)

    monkeypatch.setattr(production, "load_player_configs", lambda: {"model (high)": {"model_name": "x"}})
    monkeypatch.setattr(production, "production_snapshot", snapshot)
    monkeypatch.setattr(production, "player_record", lambda panel, player: {"results": [{}]} if panel.name in partial else {})
    monkeypatch.setattr(production, "run_panel", run_panel)
    monkeypatch.setattr(production, "publish_saved", sync_saved)
    monkeypatch.setattr(production, "local_ready", lambda panel, player: panel.name in measured)
    monkeypatch.setattr(production, "resume_rating_update", resume_rating_update)
    monkeypatch.setattr(production, "live_rating", lambda player: 1950)
    result = production.run("model (high)", full_audit=full_audit)
    return result, actions


def test_reassesses_policy_after_new_core(monkeypatch):
    result, actions = launch(monkeypatch, defer_after_core=True)
    assert actions == [("measure", "core")]
    assert result["measured"] == ["core"]
    assert result["deferred"] == ["game_like", "continuation_stability"]
    assert result["source"] == "model (low)"


def test_explicit_audit_fills_only_unmeasured_panels(monkeypatch):
    result, actions = launch(monkeypatch, measured=["core"], published=["core"],
                             defer_after_core=True, full_audit=True)
    assert actions == [("measure", "game_like"), ("measure", "continuation_stability")]
    assert not result["deferred"]


def test_ready_local_panel_publishes_without_repeating_calls(monkeypatch):
    _, actions = launch(monkeypatch, measured=["core", "game_like"],
                        published=["continuation_stability"])
    assert actions == [("sync", "core"), ("sync", "game_like")]


def test_missing_publication_after_runner_stops(monkeypatch):
    with pytest.raises(RuntimeError, match="core.*publication"):
        launch(monkeypatch, stalls=True)


def test_stale_receipt_does_not_count_as_complete(monkeypatch):
    with pytest.raises(RuntimeError, match="rating.*receipt"):
        launch(monkeypatch, published=["core", "game_like", "continuation_stability"],
               applied=False)


def test_runner_uses_canonical_manifest_paths_and_resumes_current_static_rows(monkeypatch, tmp_path):
    original = load_acquisition_policy().panels[0]
    panel = replace(original, results_path=tmp_path / "results.json")
    row = json.loads(original.results_path.read_text())["gemini-3.1-pro-preview (medium)"]["results"][0]
    scorer = SimpleNamespace(scoring_profile=row["scoring_profile"], close=lambda: None)
    monkeypatch.setattr(production, "_EngineEvaluator", lambda *args: scorer)
    panel.results_path.write_text(json.dumps({"player": {"summary": {"scoring_version": row["scoring_version"], "scoring_profile": row["scoring_profile"]}, "results": [row]}}))
    commands = []
    monkeypatch.setattr(production.subprocess, "run", lambda args, **kwargs: commands.append(args))
    production.run_panel(panel, "player")
    assert "--retry-missing" in commands[-1]
    assert str(panel.results_path) in commands[-1]
    row["prompt_history_replay"] = False
    panel.results_path.write_text(json.dumps({"player": {"summary": {"scoring_version": row["scoring_version"], "scoring_profile": row["scoring_profile"]}, "results": [row]}}))
    production.run_panel(panel, "player")
    assert "--retry-missing" not in commands[-1]
    continuation = load_acquisition_policy().panels[-1]
    production.run_panel(continuation, "player")
    assert commands[-1][-6:] == ["--limit", "8", "--probe-plies", "8", "--random-seed", "1729"]


def test_retry_after_publication_recovers_pending_rating_replay(monkeypatch):
    result, actions = launch(monkeypatch, published=["core", "game_like", "continuation_stability"],
                             applied=False, replay_succeeds=True)
    assert actions == [("replay", "saved")]
    assert result["applied_generation"] == 4


def test_render_status_rejects_mismatched_policy(monkeypatch):
    def no_credentials():
        raise FileNotFoundError()
    monkeypatch.setattr(production, "get_firestore_client", no_credentials)
    monkeypatch.setattr(production, "render_job_receipt", lambda *_: {"player": "target", "policy": "old"})
    with pytest.raises(RuntimeError, match="acquisition policy"):
        production.production_snapshot("target", load_acquisition_policy())


def test_saved_partial_tail_needs_both_supplements_even_if_render_sees_no_local_files(monkeypatch):
    result, actions = launch(monkeypatch, measured=["core"], published=["core"],
                             defer_after_core=True, partial=["continuation_stability"])
    assert actions == [("measure", "game_like"), ("measure", "continuation_stability")]
    assert result["deferred"] == []


def test_concurrent_targeted_run_stops_before_checking_or_acquiring(monkeypatch):
    monkeypatch.setattr(production, "load_player_configs", lambda: {"target": {"model_name": "test"}})
    attempts = []

    def concurrent_start(player, policy):
        attempts.append(player)
        if len(attempts) > 1:
            pytest.fail("Concurrent command reached acquisition")
        with pytest.raises(RuntimeError, match="already running"):
            production.run(player)
        raise ValueError("finished concurrency test")

    monkeypatch.setattr(production, "production_snapshot", concurrent_start)
    with pytest.raises(ValueError, match="finished concurrency test"):
        production.run("target")
    assert attempts == ["target"]


def test_unpublished_local_sibling_cannot_defer_production_measurements(monkeypatch):
    policy = load_acquisition_policy()
    target = "gemini-3.1-pro-preview (high)"
    # This target passes against the lower-effort evidence in the local artifacts.
    # Firestore has only its own core in this scenario.
    target_core = json.loads(policy.panels[0].results_path.read_text())[target]
    values = {
        ("benchmark_results", target): target_core,
        ("ratings", target): {"rating": 2050},
        ("benchmark_rating_updates", "current"): {"generation": 5, "applied_generation": 5},
    }

    def reference(collection, player):
        contents = values.get((collection, player))
        doc = SimpleNamespace(id=player, exists=bool(contents), to_dict=lambda: contents)
        return SimpleNamespace(get=lambda: doc)

    db = SimpleNamespace(
        collection=lambda name: SimpleNamespace(document=lambda player: reference(name, player)),
        get_all=lambda refs: [ref.get() for ref in refs],
    )
    monkeypatch.setattr(production, "get_firestore_client", lambda: db)
    state = production.firestore_snapshot(target, policy)
    assert state["published"] == ["core"]
    assert state["source"] is None
    assert state["pending"] == ["game_like", "continuation_stability"]


def test_replay_refresh_requires_current_deferral_before_reporting_success(monkeypatch):
    with pytest.raises(RuntimeError, match="Incomplete production measurements"):
        launch(monkeypatch, measured=["core"], published=["core"], defer_after_core=True,
               applied=False, replay_succeeds=True, revoke_after_replay=True)

def test_mixed_scoring_partial_is_rescored_before_paid_missing_rows(monkeypatch, tmp_path):
    original = load_acquisition_policy().panels[0]
    panel = replace(original, results_path=tmp_path / "panel.json")
    sample = json.loads(original.results_path.read_text())["gemini-3.1-pro-preview (medium)"]
    row = sample["results"][0]
    expected = row["scoring_profile"]
    old_row = {**row, "scoring_version": "legacy", "scoring_profile": {"old": True}}
    saved = {"summary": {"player_id": "target"}, "results": [old_row]}
    panel.results_path.write_text(json.dumps({"target": saved}))
    scorer = SimpleNamespace(scoring_profile=expected, score=lambda *_: row, close=lambda: None)
    monkeypatch.setattr(production, "_EngineEvaluator", lambda *args: scorer)
    commands = []
    monkeypatch.setattr(production.subprocess, "run", lambda args, **kwargs: commands.append(args))
    production.run_panel(panel, "target", expected_profile=expected)
    assert "--retry-missing" in commands[0]
    corrected = json.loads(panel.results_path.read_text())["target"]
    assert corrected["results"][0]["model_move"] == row["model_move"]
    assert corrected["summary"]["scoring_profile"] == expected


def test_incompatible_local_stockfish_stops_before_model_calls(monkeypatch):
    panel = load_acquisition_policy().panels[1]
    scorer = SimpleNamespace(scoring_profile={"wrong": True}, close=lambda: None)
    monkeypatch.setattr(production, "_EngineEvaluator", lambda *args: scorer)
    runner = Mock()
    monkeypatch.setattr(production.subprocess, "run", runner)
    with pytest.raises(RuntimeError, match="Stockfish.*profile"):
        production.run_panel(panel, "target", expected_profile={"required": True})
    runner.assert_not_called()

def test_complete_local_mixed_scores_repaired_and_published_without_acquisition(monkeypatch, tmp_path):
    original = load_acquisition_policy().panels[0]
    player = "gemini-3.1-pro-preview (medium)"
    sample = json.loads(original.results_path.read_text())[player]
    old_score = dict(sample["results"][0])
    sample["results"][0]["scoring_version"] = "old"
    panel = replace(original, results_path=tmp_path / "panel.json")
    panel.results_path.write_text(json.dumps({player: sample}))
    expected = sample["summary"]["scoring_profile"]
    scorer = SimpleNamespace(scoring_profile=expected, close=lambda: None)
    monkeypatch.setattr(production, "_EngineEvaluator", lambda *args: scorer)

    def rescore(record, *args):
        record["results"][0] = old_score
        return record

    monkeypatch.setattr(production, "rescore_record", rescore)
    runner, sync = Mock(), Mock()
    monkeypatch.setattr(production.subprocess, "run", runner)
    monkeypatch.setattr(production, "sync_panel", sync)
    production.run_panel(panel, player, expected_profile=expected)
    runner.assert_not_called()
    sync.assert_called_once()
    assert sync.call_args.args[2]["results"][0] == old_score
