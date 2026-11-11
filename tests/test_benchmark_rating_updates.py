"""Production benchmark seeds update saved history without losing new games."""

import copy
import json
from types import SimpleNamespace

import pytest

from game.models import GameResult
from position_benchmark.layout import CORE_POSITIONS_PATH
from position_benchmark.predictions import CURRENT_BENCHMARK_VERSION
from rating.glicko2 import PlayerRating
from rating.rating_store import PositionBenchmarkInputs
from rating import benchmark_recalculation as updates

PID = "gemini-4-carbon (low)"
PREDICTION = 1800.0


class Database:
    def __init__(self):
        self.docs = {}

    def collection(self, collection):
        db = self

        class Reference:
            def __init__(self, key):
                self.id, self.key = key, (collection, key)

            def get(self, transaction=None):
                if transaction and transaction.writes:
                    raise AssertionError("All reads must precede writes")
                value = copy.deepcopy(db.docs.get(self.key))
                return SimpleNamespace(id=self.id, exists=value is not None, to_dict=lambda: value)

        class Collection:
            def document(self, key):
                return Reference(key)

            def stream(self):
                return [Reference(key).get() for name, key in list(db.docs) if name == collection]

        return Collection()

    def get_all(self, refs, transaction=None):
        return [ref.get(transaction=transaction) for ref in refs]

    def transaction(self):
        return Transaction(self)


class Transaction:
    def __init__(self, db):
        self.db, self.writes = db, {}

    def get(self, ref):
        yield ref.get(transaction=self)

    def set(self, ref, data):
        self.writes[ref.key] = copy.deepcopy(data)


def draw(game_id):
    return GameResult(
        game_id=game_id, white_id="lichess:human", black_id=PID, winner="draw",
        termination="draw_agreement", moves=2, illegal_moves_white=0, illegal_moves_black=0,
        total_moves_white=1, total_moves_black=1, pgn_path="saved.pgn",
        created_at="2026-11-10T12:00:00+00:00", game_type="human_challenge",
        human_lichess_username="human", human_rating=PREDICTION, human_rating_deviation=60,
    )


def panel_record(positions):
    return {
        "summary": {"player_id": PID},
        "results": [
            {
                "position_idx": index,
                "fen": position["fen"],
                "model_move": position["best_move"],
                "cpl": 0,
                "is_legal": True,
                "position_benchmark_version": CURRENT_BENCHMARK_VERSION,
                "prompt_history_replay": True,
                "stockfish_depth": 30,
            }
            for index, position in enumerate(positions)
        ],
    }


@pytest.fixture
def db(monkeypatch):
    from firebase_admin import firestore
    from rating import rating_store
    database = Database()
    positions = json.loads(CORE_POSITIONS_PATH.read_text())["positions"]
    # These tests exercise transactional replay from a known seed, independently
    # of model acquisitions and later corrections to the production CPL corpus.
    record = panel_record(positions)
    inputs = PositionBenchmarkInputs(
        core_results={PID: record}, core_positions=positions,
        blunder_results=None, blunder_positions=None,
        game_like_results=None, game_like_positions=None, stability_results=None,
    )
    monkeypatch.setattr(updates, "load_position_benchmark_inputs", lambda *_, **__: inputs)
    monkeypatch.setattr(rating_store, "load_position_benchmark_inputs", lambda *_, **__: inputs)
    monkeypatch.setattr(updates, "benchmark_predictions_from_inputs", lambda _: {PID: PREDICTION})

    def transactional(function):
        def run(transaction):
            result = function(transaction)
            transaction.db.docs.update(transaction.writes)
            return result
        return run

    monkeypatch.setattr(firestore, "transactional", transactional)
    database.docs[(updates.STATE_COLLECTION, updates.STATE_ID)] = {"generation": 1, "applied_generation": 0}
    database.docs[("benchmark_results", PID)] = record
    database.docs[("ratings", PID)] = PlayerRating(player_id=PID, rating=1200, games_played=1, draws=1).to_dict()
    database.docs[("results", "draw-1")] = draw("draw-1").to_json()
    return database


def test_reseeds_existing_player_and_leaves_saved_games_unchanged(db):
    original = copy.deepcopy(db.docs[("results", "draw-1")])
    receipt = updates.recalculate_pending(db)
    rating = db.docs[("ratings", PID)]
    assert rating["rating"] == pytest.approx(PREDICTION)
    assert (rating["games_played"], rating["draws"]) == (1, 1)
    assert db.docs[("results", "draw-1")] == original
    assert receipt["generation"] == 1
    assert db.docs[(updates.STATE_COLLECTION, updates.STATE_ID)]["applied_generation"] == 1
    assert ("rating_recalculations", "benchmark-1") in db.docs
    saved = copy.deepcopy(db.docs)
    updates.recalculate_pending(db)
    assert db.docs == saved


def test_concurrent_game_forces_recalculation_without_overwriting_it(db, monkeypatch):
    calculate = updates._calculate
    raced = []

    def racing_calculation(*args, **kwargs):
        result = calculate(*args, **kwargs)
        if not raced:
            raced.append(True)
            db.docs[("results", "draw-2")] = draw("draw-2").to_json()
            db.docs[("ratings", PID)].update(games_played=2, draws=2, rating=1500)
        return result

    monkeypatch.setattr(updates, "_calculate", racing_calculation)
    updates.recalculate_pending(db)
    assert db.docs[("ratings", PID)]["games_played"] == 2
    assert db.docs[("ratings", PID)]["draws"] == 2
    assert db.docs[("ratings", PID)]["rating"] == pytest.approx(PREDICTION)


def test_failure_leaves_old_ratings_and_retryable_revision(db, monkeypatch):
    before = copy.deepcopy(db.docs)

    def fail(*args, **kwargs):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(updates, "_calculate", fail)
    with pytest.raises(RuntimeError, match="unavailable"):
        updates.recalculate_pending(db)
    assert db.docs == before


def test_failed_sync_keeps_panel_and_retries_without_replaying_games(db, monkeypatch):
    from position_benchmark import sync
    record = db.docs.pop(("benchmark_results", PID))
    db.docs.pop((updates.STATE_COLLECTION, updates.STATE_ID))
    calculate = updates._calculate
    monkeypatch.setattr(updates, "_calculate", lambda *_: (_ for _ in ()).throw(RuntimeError("offline")))
    with pytest.raises(RuntimeError, match="offline"):
        sync.write_panel(db, {"player_id": PID, "panel": "core", "record": record})
    assert db.docs[("benchmark_results", PID)] == record
    assert db.docs[(updates.STATE_COLLECTION, updates.STATE_ID)]["generation"] == 1
    assert db.docs[("ratings", PID)]["rating"] == 1200
    monkeypatch.setattr(updates, "_calculate", calculate)
    sync.write_panel(db, {"player_id": PID, "panel": "core", "record": record})
    assert db.docs[("ratings", PID)]["rating"] == pytest.approx(PREDICTION)


def test_first_benchmark_can_seed_a_player_before_any_saved_games(db):
    del db.docs[("results", "draw-1")]
    del db.docs[("ratings", PID)]
    receipt = updates.recalculate_pending(db)
    assert db.docs[("ratings", PID)]["rating"] == pytest.approx(PREDICTION)
    assert db.docs[("ratings", PID)]["games_played"] == 0
    assert db.docs[("ratings", PID)]["rating_deviation"] == 300.0
    assert db.docs[("ratings", PID)]["games_rd"] == 350.0
    assert receipt["seed_rd"] == 300.0
    assert receipt["calibration_version"] == "root-reference-v1-20261111-rd300"
    state = db.docs[(updates.STATE_COLLECTION, updates.STATE_ID)]
    assert state["calibration_version"] == receipt["calibration_version"]
    assert db.docs[("rating_recalculations", "benchmark-1")]["calibration_version"] == receipt["calibration_version"]


def test_same_panel_retry_does_not_increment_revision(db):
    from position_benchmark import sync
    record = db.docs.pop(("benchmark_results", PID))
    db.docs.pop((updates.STATE_COLLECTION, updates.STATE_ID))
    first = sync.write_panel(db, {"player_id": PID, "panel": "core", "record": record})
    saved = copy.deepcopy(db.docs)
    second = sync.write_panel(db, {"player_id": PID, "panel": "core", "record": record})
    assert db.docs == saved
    assert first["rating_recalculation"]["generation"] == second["rating_recalculation"]["generation"] == 1


def test_supplement_publication_queues_update_for_bundled_core(db, monkeypatch):
    from position_benchmark import sync
    from position_benchmark.layout import GAME_LIKE_POSITIONS_PATH
    panel = panel_record(json.loads(GAME_LIKE_POSITIONS_PATH.read_text())["positions"])
    db.docs.pop(("benchmark_results", PID))
    db.docs.pop((updates.STATE_COLLECTION, updates.STATE_ID))
    monkeypatch.setattr(updates, "recalculate_pending", lambda _: {"status": "pending"})
    sync.write_panel(db, {"player_id": PID, "panel": "game_like", "record": panel})
    assert db.docs[(updates.STATE_COLLECTION, updates.STATE_ID)]["generation"] == 1
    assert "results" not in db.docs[("benchmark_results", PID)]


def test_new_benchmark_revision_cannot_be_marked_applied_by_old_calculation(db, monkeypatch):
    calculate = updates._calculate
    changed = []

    def revise(*args, **kwargs):
        result = calculate(*args, **kwargs)
        if not changed:
            changed.append(True)
            db.docs[(updates.STATE_COLLECTION, updates.STATE_ID)]["generation"] = 2
        return result

    monkeypatch.setattr(updates, "_calculate", revise)
    receipt = updates.recalculate_pending(db)
    assert receipt["generation"] == 2
    assert ("rating_recalculations", "benchmark-1") not in db.docs


def test_revision_reader_preserves_last_good_value_on_failure(db, monkeypatch):
    import firebase_client
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setattr(firebase_client, "get_firestore_client", lambda: db)
    monkeypatch.setattr(updates, "_revision_cache", (0.0, 0))
    clock = [1000.0]
    monkeypatch.setattr(updates, "monotonic", lambda: clock[0])
    db.docs[(updates.STATE_COLLECTION, updates.STATE_ID)]["applied_generation"] = 7
    assert updates.applied_generation() == 7
    clock[0] += 11
    monkeypatch.setattr(firebase_client, "get_firestore_client", lambda: (_ for _ in ()).throw(OSError("offline")))
    assert updates.applied_generation() == 7
