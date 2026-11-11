"""A scoring migration preserves evidence and refuses concurrent production writes."""

import copy
import json
from types import SimpleNamespace

import pytest

from rating.glicko2 import PlayerRating


class Database:
    def __init__(self, docs=None):
        self.docs = docs or {}
        self.commits = []

    def collection(self, path):
        db = self

        class Reference:
            def __init__(self, key):
                self.id, self.key = key, (path, key)

            def collection(self, name):
                return db.collection(f"{path}/{self.id}/{name}")

            def get(self, transaction=None):
                if transaction and transaction.writes:
                    raise AssertionError("Reads must precede writes")
                value = copy.deepcopy(db.docs.get(self.key))
                return SimpleNamespace(id=self.id, exists=value is not None, to_dict=lambda: value)

        class Collection:
            def document(self, key):
                return Reference(key)

            def stream(self, transaction=None):
                return [Reference(key).get(transaction) for name, key in list(db.docs) if name == path]

        return Collection()

    def transaction(self):
        return Transaction(self)

    def batch(self):
        return Transaction(self)


class Transaction:
    def __init__(self, db):
        self.db, self.writes = db, {}

    def set(self, ref, data):
        self.writes[ref.key] = copy.deepcopy(data)

    def commit(self):
        self.db.commits.append(copy.deepcopy(self.writes))
        self.db.docs.update(self.writes)


def panel(player="model", *, corrected=False):
    data = {"summary": {"player_id": player}, "results": [{
        "model_move": "e2e4", "raw_response": "e4", "prompt_tokens": 10,
        "best_move": "d2d4", "cpl": 80, "eval_model": 0, "is_best": False,
    }]}
    if corrected:
        data["summary"].update(scoring_version="v2", scoring_profile={"engine": "test"})
        data["results"][0].update(best_move="e2e4", cpl=0, eval_model=80, is_best=True,
                                  scoring_version="v2", scoring_profile={"engine": "test"})
    return data


@pytest.fixture
def migration(monkeypatch):
    from firebase_admin import firestore
    from position_benchmark import scoring_migration as module

    def transactional(function):
        def run(transaction):
            result = function(transaction)
            transaction.commit()
            return result
        return run

    monkeypatch.setattr(firestore, "transactional", transactional)
    monkeypatch.setattr(module, "_ready", lambda record, panel: bool(record.get("results")))
    monkeypatch.setattr(module, "_validate_corrected", lambda *_: None)
    monkeypatch.setattr(module, "_baseline_panels", lambda: ({}, {}, {}))
    monkeypatch.setattr(module, "_positions", lambda *_: [])
    monkeypatch.setattr(module, "benchmark_predictions_from_inputs", lambda inputs: {
        key: 1400.0 if record["results"][0]["cpl"] else 1800.0
        for key, record in inputs.core_results.items() if record.get("results")
    })
    monkeypatch.setattr(module, "_calculate", lambda results, predictions: {
        pid: PlayerRating(player_id=pid, rating=value).to_dict() for pid, value in predictions.items()
    })
    return module


def database():
    return Database({
        ("benchmark_results", "model"): panel(),
        ("ratings", "model"): PlayerRating(player_id="model", rating=1400).to_dict(),
        ("ratings", "unrelated"): PlayerRating(player_id="unrelated", rating=1212).to_dict(),
        ("benchmark_rating_updates", "current"): {"generation": 4, "applied_generation": 4},
    })


def test_preparation_is_read_only_and_apply_is_atomic_and_idempotent(migration, tmp_path):
    db = database()
    original = copy.deepcopy(db.docs)
    plan = migration.prepare(db, {"model": panel(corrected=True)}, {}, {}, tmp_path)
    assert db.docs == original
    assert json.loads(open(plan["backup_path"]).read())["snapshot"]["ratings"]["unrelated"] == original[("ratings", "unrelated")]
    plan = json.loads(json.dumps(plan))  # Prepared plans survive process boundaries.
    receipt = migration.apply(db, plan)
    assert db.docs[("ratings", "model")]["rating"] == 1800
    assert db.docs[("ratings", "unrelated")] == original[("ratings", "unrelated")]
    assert db.docs[("benchmark_rating_updates", "current")]["applied_generation"] == 5
    assert db.docs[("benchmark_rating_updates", "current")]["scoring_version"] == "v2"
    assert any("/chunks" in name for name, _ in db.docs)
    atomic = db.commits[-1]
    assert ("benchmark_results", "model") in atomic and ("ratings", "model") in atomic
    commits = len(db.commits)
    assert migration.apply(db, plan) == receipt
    assert len(db.commits) == commits


@pytest.mark.parametrize("change", ["response", "tokens", "move"])
def test_changed_raw_evidence_refuses_preparation(migration, tmp_path, change):
    db = database()
    candidate = panel(corrected=True)
    key, value = {"response": ("raw_response", "d4"), "tokens": ("prompt_tokens", 11),
                  "move": ("model_move", "d2d4")}[change]
    candidate["results"][0][key] = value
    with pytest.raises(ValueError, match="evidence"):
        migration.prepare(db, {"model": candidate}, {}, {}, tmp_path)
    assert db.commits == []


@pytest.mark.parametrize("collection,key", [("ratings", "new-player"), ("results", "new-game"),
                                            ("benchmark_results", "new-model"),
                                            ("benchmark_rating_updates", "current")])
def test_concurrent_collection_or_state_change_cannot_publish(migration, tmp_path, collection, key):
    db = database()
    plan = migration.prepare(db, {"model": panel(corrected=True)}, {}, {}, tmp_path)
    db.docs[(collection, key)] = {"concurrent": True}
    current = copy.deepcopy(db.docs)
    with pytest.raises(migration.ConcurrentMigration, match="changed"):
        migration.apply(db, plan)
    assert all(db.docs[key] == value for key, value in current.items())
    assert db.docs[("benchmark_results", "model")]["results"][0]["cpl"] == 80


def test_corrected_local_model_absent_from_production_is_included(migration, tmp_path, monkeypatch):
    db = database()
    legacy = {"summary": {"player_id": "legacy"}, "results": []}
    monkeypatch.setattr(migration, "_baseline_panels", lambda: ({"bundled": panel("bundled")}, {}, {}))
    candidate = {"model": panel(corrected=True), "bundled": panel("bundled", corrected=True), "legacy": legacy}
    plan = migration.prepare(db, candidate, {}, {}, tmp_path)
    assert set(plan["predictions_after"]) == {"model", "bundled"}
    assert plan["benchmark_after"]["bundled"]["results"][0]["cpl"] == 0
    assert "legacy" not in plan["benchmark_after"]


def test_game_statistics_must_remain_unchanged(migration, tmp_path, monkeypatch):
    db = database()
    monkeypatch.setattr(migration, "_calculate", lambda *_: {
        "model": PlayerRating(player_id="model", games_played=1, wins=1).to_dict()})
    with pytest.raises(ValueError, match="statistics"):
        migration.prepare(db, {"model": panel(corrected=True)}, {}, {}, tmp_path)


def test_candidate_plan_tampering_refused_before_backup_writes(migration, tmp_path):
    db = database()
    plan = migration.prepare(db, {"model": panel(corrected=True)}, {}, {}, tmp_path)
    plan["ratings_after"]["model"]["rating"] = 2500
    with pytest.raises(ValueError, match="hash"):
        migration.apply(db, plan)
    assert db.commits == []


def test_absent_production_model_requires_matching_bundled_evidence(migration, tmp_path):
    db = database()
    with pytest.raises(ValueError, match="source evidence"):
        migration.prepare(db, {"model": panel(corrected=True), "unknown": panel("unknown", corrected=True)}, {}, {}, tmp_path)


def test_ready_uncorrected_supplement_cannot_enter_candidate_predictions(migration, tmp_path):
    with pytest.raises(ValueError, match="provenance"):
        migration.prepare(database(), {"model": panel(corrected=True)}, {"model": panel()}, {}, tmp_path)


def test_budget_failure_precedes_any_firestore_write(migration, tmp_path, monkeypatch):
    monkeypatch.setattr(migration, "MAX_WRITES", 3)
    db = database()
    with pytest.raises(ValueError, match="budget"):
        migration.prepare(db, {"model": panel(corrected=True)}, {}, {}, tmp_path)
    assert db.commits == []


def test_failed_firestore_backup_cannot_touch_production(migration, tmp_path, monkeypatch):
    db = database()
    plan = migration.prepare(db, {"model": panel(corrected=True)}, {}, {}, tmp_path)
    monkeypatch.setattr(db, "batch", lambda: (_ for _ in ()).throw(OSError("backup offline")))
    before = copy.deepcopy(db.docs)
    with pytest.raises(OSError, match="backup offline"):
        migration.apply(db, plan)
    assert db.docs == before


def test_response_fingerprint_preserves_unknown_evidence_and_ignores_only_scores(migration):
    old = panel()
    fresh = panel(corrected=True)
    assert migration.evidence_fingerprint(old) == migration.evidence_fingerprint(fresh)
    fresh["results"][0]["history"] = ["e2e4"]
    assert migration.evidence_fingerprint(old) != migration.evidence_fingerprint(fresh)


def test_rescoring_audit_and_rank_labels_do_not_change_response_identity(migration):
    old = panel()
    corrected = panel(corrected=True)
    corrected["rescoring"] = {"input_sha256": "source", "output_sha256": "corrected"}
    corrected["results"][0]["rank_metadata_source"] = "screening-multipv"
    assert migration.evidence_fingerprint(old) == migration.evidence_fingerprint(corrected)


def test_firestore_timestamp_encoding_preserves_nanoseconds(migration):
    from google.api_core.datetime_helpers import DatetimeWithNanoseconds
    from datetime import timezone
    timestamp = DatetimeWithNanoseconds(2026, 11, 11, nanosecond=123456789, tzinfo=timezone.utc)
    encoded = json.loads(json.dumps(migration._encode(timestamp)))
    assert migration._decode(encoded).timestamp_pb() == timestamp.timestamp_pb()


def test_reusable_plan_round_trips_datetime_state(migration, tmp_path):
    from datetime import datetime, timezone
    db = database()
    timestamp = datetime(2026, 11, 11, tzinfo=timezone.utc)
    db.docs[("benchmark_rating_updates", "current")]["requested_at"] = timestamp
    plan = migration.prepare(db, {"model": panel(corrected=True)}, {}, {}, tmp_path)
    migration.apply(db, json.loads(json.dumps(plan)))
    assert db.docs[("benchmark_rating_updates", "current")]["requested_at"] == timestamp


def test_real_validation_accepts_a_complete_corrected_core_and_rejects_mixed_rows():
    from position_benchmark import scoring_migration as module
    from position_benchmark.evaluation import SCORING_VERSION
    from position_benchmark.layout import CORE_RESULTS_PATH
    records = json.loads(CORE_RESULTS_PATH.read_text())
    player, original = next((pid, record) for pid, record in records.items() if module._ready(record, "core"))
    corrected = copy.deepcopy(original)
    provenance = {"scoring_version": SCORING_VERSION, "scoring_profile": {"engine": "test", "depth": 30}}
    corrected["summary"].update(provenance)
    for row in corrected["results"]:
        row.update(provenance)
        if row.get("is_best"):
            row["cpl"] = 0
    module._validate_corrected(player, "core", corrected)
    corrected["results"][0].pop("scoring_version")
    with pytest.raises(ValueError, match="Mixed"):
        module._validate_corrected(player, "core", corrected)


def test_real_rescore_file_output_prepares_without_changing_saved_evidence(tmp_path, monkeypatch):
    from position_benchmark import scoring_migration as module
    from position_benchmark.evaluation import SCORING_VERSION
    from position_benchmark.layout import CORE_RESULTS_PATH
    from scripts.rescore_position_results import rescore_file

    records = json.loads(CORE_RESULTS_PATH.read_text())
    player, original = next((pid, record) for pid, record in records.items() if module._ready(record, "core"))
    original = copy.deepcopy(original)
    original["results"][0]["move_rank"] = 2

    class Evaluator:
        scoring_profile = {"engine": "cpu-test", "depth": 30}

        def score(self, board, move):
            best = move or next(iter(board.legal_moves))
            return {
                "best_move": best.uci(), "best_move_san": board.san(best),
                "eval_before": 100, "eval_best": 100, "eval_model": 100 if move else -10000,
                "cpl": 0 if move else 10100, "is_best": move is not None,
                "scoring_version": SCORING_VERSION, "scoring_profile": self.scoring_profile,
            }

    source = tmp_path / "source.json"
    source.write_text(json.dumps({player: original}))
    corrected = rescore_file(panel="core", input_path=source, output_path=tmp_path / "corrected.json",
                             workers=1, evaluator_factory=Evaluator)
    assert "rescoring" in corrected[player]
    assert any(row.get("rank_metadata_source") for row in corrected[player]["results"])
    db = Database({("benchmark_results", player): original,
                   ("ratings", player): PlayerRating(player_id=player).to_dict()})
    monkeypatch.setattr(module, "_baseline_panels", lambda: ({player: original}, {}, {}))
    monkeypatch.setattr(module, "_calculate", lambda results, predictions: {
        pid: PlayerRating(player_id=pid, rating=value).to_dict() for pid, value in predictions.items()})
    plan = module.prepare(db, corrected, {}, {}, tmp_path / "backups")
    assert player in plan["predictions_after"]
    assert plan["benchmark_after"][player] == corrected[player]
    assert module.evidence_fingerprint(original) == module.evidence_fingerprint(corrected[player])
    assert db.docs[("benchmark_results", player)] == original


def test_stale_remote_core_uses_authoritative_ready_bundled_source(migration, tmp_path, monkeypatch):
    db = database()
    db.docs[("benchmark_results", "model")]["results"][0]["raw_response"] = "legacy prompt response"
    db.docs[("benchmark_results", "model")]["summary"]["stale"] = True
    stale = copy.deepcopy(db.docs[("benchmark_results", "model")])
    monkeypatch.setattr(migration, "_ready", lambda record, panel: bool(record.get("results"))
                        and not record.get("summary", {}).get("stale"))
    monkeypatch.setattr(migration, "_baseline_panels", lambda: ({"model": panel()}, {}, {}))
    plan = migration.prepare(db, {"model": panel(corrected=True)}, {}, {}, tmp_path)
    assert plan["benchmark_after"]["model"]["results"][0]["raw_response"] == "e4"
    assert plan["snapshot"]["benchmark_results"]["model"] == stale


def test_real_ready_registry_normalization_preserves_response_identity():
    from position_benchmark import scoring_migration as module
    from position_benchmark.layout import CORE_RESULTS_PATH
    records = json.loads(CORE_RESULTS_PATH.read_text())
    normalized = copy.deepcopy(records["gemini-3.5-flash (medium)"])
    remote = copy.deepcopy(normalized)
    for row in remote["results"]:
        row.pop("position_id", None)
        row["position_idx"] = row.get("legacy_position_idx", row["position_idx"])
        row.pop("legacy_position_idx", None)
        row.pop("panel", None)
        row.pop("result_schema_version", None)
    assert module._ready(remote, "core")
    assert module._ready(normalized, "core")
    assert module._same_evidence(remote, normalized, "core")
    remote["results"][0]["model_move"] = "a1a2"
    assert not module._same_evidence(remote, normalized, "core")


@pytest.mark.parametrize("field,value", [("fen", "wrong FEN"), ("prompt_history_replay", False),
                                         ("prompt_tokens", 123456), ("raw_response", "new response"),
                                         ("conditional_retry_protocol_version", "changed")])
def test_registry_normalization_never_hides_context_or_response_changes(field, value):
    from position_benchmark import scoring_migration as module
    from position_benchmark.layout import CORE_RESULTS_PATH
    records = json.loads(CORE_RESULTS_PATH.read_text())
    source = copy.deepcopy(records["gemini-3.5-flash (medium)"])
    candidate = copy.deepcopy(source)
    candidate["results"][0].pop("panel", None)
    candidate["results"][0][field] = value
    assert not module._same_evidence(source, candidate, "core")
