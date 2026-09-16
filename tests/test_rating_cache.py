"""Firestore cache copies must preserve flags and remain independent."""
import time
from types import SimpleNamespace

import pytest

import rating.rating_store as module
from rating.glicko2 import PlayerRating


@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("operation", ["fetch", "cached", "fallback", "save", "save_all"])
def test_firestore_cache_roundtrip_preserves_rating_fields(monkeypatch, frozen, operation):
    original = PlayerRating(
        player_id="test-model", rating=1137.1, rating_deviation=294.6,
        games_rd=310.0, games_played=4, losses=4, is_frozen=frozen,
    )
    expected = original.to_dict()
    writes = []

    def stream(timeout):
        if operation == "fallback":
            raise ConnectionError("Firestore unavailable")
        if operation == "cached":
            pytest.fail("A cache hit must not fetch Firestore")
        return [SimpleNamespace(to_dict=lambda: expected.copy())]

    collection = SimpleNamespace(
        stream=stream,
        document=lambda pid: SimpleNamespace(set=lambda data: writes.append(data)),
    )
    batch = SimpleNamespace(set=lambda ref, data: writes.append(data), commit=lambda: None)
    store = module.RatingStore.__new__(module.RatingStore)
    store._db = SimpleNamespace(collection=lambda name: collection, batch=lambda: batch)
    store._collection = "ratings"
    store._ratings = {original.player_id: original} if operation.startswith("save") else {}
    store._benchmark_predictions = {}
    cached = PlayerRating.from_dict(expected)
    cached.is_frozen = not frozen if operation.startswith("save") else frozen
    monkeypatch.setattr(module, "_firestore_cache", {original.player_id: cached} if operation != "fetch" else {})
    monkeypatch.setattr(module, "_firestore_cache_time", time.time() if operation == "cached" else 0)
    monkeypatch.setattr(module, "_should_invalidate_cache", lambda: False)

    if operation == "save":
        store._save_to_firestore(original.player_id)
    elif operation == "save_all":
        store._save_all_to_firestore()
    else:
        store._load_from_firestore()

    assert store._ratings[original.player_id].to_dict() == expected
    assert module._firestore_cache[original.player_id].to_dict() == expected
    if operation.startswith("save"):
        assert writes == [expected]
    store._ratings[original.player_id].is_frozen = not frozen
    assert module._firestore_cache[original.player_id].is_frozen is frozen
