import copy
import json
import uuid
from types import SimpleNamespace
import io

import pytest

from game.models import GameResult
from game import publication
from rating.glicko2 import PlayerRating


def test_production_snapshot_replaces_stale_zero_games_without_disk_writes(monkeypatch, tmp_path):
    from rating.rating_store import RatingStore
    path = tmp_path / "ratings.json"
    store = RatingStore(path=str(path), use_firestore=False, use_benchmark_predictions=False)
    store.set(PlayerRating(player_id="gpt-6-astra (medium)"))
    before = path.read_bytes()
    rows = [{"player_id": "gpt-6-astra (medium)", "rating": 1666,
             "rating_deviation": 187, "games_played": 4, "wins": 3,
             "losses": 1, "draws": 0, "is_frozen": False}]
    monkeypatch.setattr(publication.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(json.dumps(rows).encode()))
    publication.load_production_ratings(store)
    rating = store.get("gpt-6-astra (medium)")
    assert (rating.rating, rating.games_played, rating.games_rd) == (1666, 4, 187)
    assert path.read_bytes() == before


def payload():
    result = GameResult(
        game_id=str(uuid.uuid4()), white_id="white", black_id="black",
        winner="white", termination="resignation", moves=1,
        illegal_moves_white=0, illegal_moves_black=0,
        total_moves_white=1, total_moves_black=0, pgn_path="",
        created_at="2026-09-13T00:00:00+00:00",
    )
    return result, '[White "white"]\n[Black "black"]\n[Result "1-0"]\n[Termination "resignation"]\n\n1. e4 1-0'


class Database:
    def __init__(self):
        self.docs = {
            ("ratings", name): PlayerRating(player_id=name, rating=1500).to_dict()
            for name in ("white", "black")
        }

    def collection(self, name):
        db = self

        class Collection:
            def document(self, key):
                return (name, key)

        return Collection()


class Transaction:
    def __init__(self, db):
        self.db = db

    def get(self, ref):
        value = copy.deepcopy(self.db.docs.get(ref))

        class Snapshot:
            exists = value is not None

            def to_dict(self):
                return value

        yield Snapshot()

    def set(self, ref, value):
        self.db.docs[ref] = copy.deepcopy(value)


def test_publish_applies_both_ratings_only_once():
    r, pgn = payload()
    db = Database()
    tx = Transaction(db)
    first = publication.commit_result(tx, db, r, pgn, set(), set())
    saved = copy.deepcopy(db.docs)
    second = publication.commit_result(tx, db, r, pgn, set(), set())
    assert first["already_published"] is False
    assert second["already_published"] is True
    assert db.docs == saved
    assert db.docs[("ratings", "white")]["games_played"] == 1
    assert db.docs[("ratings", "white")]["rating"] > 1500
    assert db.docs[("ratings", "black")]["rating"] < 1500


def test_conflicting_game_id_is_not_overwritten():
    r, pgn = payload()
    db = Database()
    tx = Transaction(db)
    publication.commit_result(tx, db, r, pgn, set(), set())
    r.tokens_white = {"prompt_tokens": 999}
    saved = copy.deepcopy(db.docs)
    with pytest.raises(ValueError, match="conflict"):
        publication.commit_result(tx, db, r, pgn, set(), set())
    assert db.docs == saved


@pytest.mark.parametrize("termination", ["api_error", "error", "cancelled", "unknown", "normal"])
def test_failed_games_are_never_published(termination):
    r, pgn = payload()
    r.termination = termination
    db = Database()
    saved = copy.deepcopy(db.docs)
    with pytest.raises(ValueError):
        publication.commit_result(Transaction(db), db, r, pgn, set(), set())
    assert db.docs == saved


def test_pgn_mismatch_is_rejected():
    r, pgn = payload()
    r.moves = 2
    with pytest.raises(ValueError, match="plies"):
        publication.validate_result(r, pgn)


def test_anchor_and_ghost_rules_preserve_ratings_but_count_game():
    r, pgn = payload()
    db = Database()
    publication.commit_result(Transaction(db), db, r, pgn, {"white"}, {"white"})
    assert db.docs[("ratings", "white")]["rating"] == 1500
    assert db.docs[("ratings", "black")]["rating"] == 1500
    assert db.docs[("ratings", "black")]["losses"] == 1


def test_unknown_players_fail_without_partial_writes():
    r, pgn = payload()
    db = Database()
    del db.docs[("ratings", "black")]
    saved = copy.deepcopy(db.docs)
    with pytest.raises(ValueError, match="rating"):
        publication.commit_result(Transaction(db), db, r, pgn, set(), set())
    assert db.docs == saved


def test_render_publication_receipt_prevents_resubmission(monkeypatch, tmp_path):
    r, pgn = payload()
    pgn_path = tmp_path / "game.pgn"
    pgn_path.write_text(pgn)
    r.pgn_path = str(pgn_path)
    result_path = tmp_path / "game.json"
    result_path.write_text(json.dumps(r.to_json()))
    submissions = []

    def render(args, **kwargs):
        if args[1:3] == ["jobs", "create"]:
            submissions.append(args)
            return SimpleNamespace(returncode=0, stdout='{"id":"job-test"}')
        assert args[1] == "logs"
        marker = "PUBLICATION_COMPLETE " + json.dumps({"game_id": r.game_id})
        return SimpleNamespace(returncode=0, stdout=json.dumps({"message": marker}))

    monkeypatch.setattr(publication.subprocess, "run", render)
    first = publication.publish_saved_game(result_path)
    assert first["status"] == "published"
    assert json.loads(result_path.with_suffix(".publication").read_text())["status"] == "published"
    assert publication.publish_saved_game(result_path) == first
    assert len(submissions) == 1


def test_pending_receipt_resumes_same_render_job(monkeypatch, tmp_path):
    r, pgn = payload()
    path = tmp_path / "game.json"
    pgn_path = tmp_path / "game.pgn"
    pgn_path.write_text(pgn)
    r.pgn_path = str(pgn_path)
    path.write_text(json.dumps(r.to_json()))
    path.with_suffix(".publication").write_text(json.dumps({"game_id": r.game_id, "job_id": "existing-job", "status": "pending"}))

    def render(args, **kwargs):
        assert args[1:4] == ["logs", "--resources", "existing-job"]
        return SimpleNamespace(returncode=0, stdout=json.dumps({"message": "PUBLICATION_COMPLETE " + json.dumps({"game_id": r.game_id})}))

    monkeypatch.setattr(publication.subprocess, "run", render)
    assert publication.publish_saved_game(path)["status"] == "published"
