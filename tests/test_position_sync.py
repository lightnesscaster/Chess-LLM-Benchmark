import copy
import json

import pytest

from position_benchmark import sync
from position_benchmark.layout import CORE_RESULTS_PATH


def test_supplement_sync_preserves_core_and_other_panels():
    old = {"summary": {"old": True}, "results": [1], "supplements": {"stability": {"keep": True}}}
    original = copy.deepcopy(old)
    merged = sync.merge_panel_record(old, "game_like", {"results": [2]})
    assert merged["results"] == [1]
    assert merged["supplements"] == {"stability": {"keep": True}, "game_like": {"results": [2]}}
    assert old == original
    replaced = sync.merge_panel_record(merged, "core", {"summary": {"fresh": True}, "results": [3]})
    assert replaced["summary"] == {"fresh": True}
    assert replaced["supplements"] == merged["supplements"]


def test_missing_local_credentials_uses_render(monkeypatch):
    import firebase_client
    record = json.loads(CORE_RESULTS_PATH.read_text())["gemini-3.1-pro-preview (medium)"]
    def missing():
        raise FileNotFoundError("no local key")
    monkeypatch.setattr(firebase_client, "get_firestore_client", missing)
    received = []
    monkeypatch.setattr(sync, "sync_via_render", lambda payload: received.append(payload) or {"synced": True})
    assert sync.sync_panel("gemini-3.1-pro-preview (medium)", "core", record) == {"synced": True}
    assert received[0]["player_id"] == "gemini-3.1-pro-preview (medium)"
    assert received[0]["record"] == record


def test_incomplete_results_cannot_be_published():
    with pytest.raises(ValueError):
        sync.sync_panel("gpt-6-astra (medium)", "core", {"results": []})


def test_wrong_player_cannot_be_published():
    record = json.loads(CORE_RESULTS_PATH.read_text())["gemini-3.1-pro-preview (medium)"]
    with pytest.raises(ValueError, match="player"):
        sync.sync_panel("different-player", "core", record)


@pytest.mark.parametrize("stale_core", [False, True])
def test_rating_reader_passes_synced_supplements_to_predictor(monkeypatch, stale_core):
    from types import SimpleNamespace
    import firebase_client
    import rating.rating_store as module
    pid = "gemini-3.1-pro-preview (medium)"
    core = json.loads(CORE_RESULTS_PATH.read_text())[pid]
    if stale_core:
        core["results"] = []
    core["supplements"] = {"game_like": {"remote": "game-like"}, "stability": {"remote": "stability"}}
    doc = SimpleNamespace(id=pid, to_dict=lambda: core)
    monkeypatch.setattr(firebase_client, "get_firestore_client", lambda: SimpleNamespace(collection=lambda name: SimpleNamespace(stream=lambda: [doc])))
    monkeypatch.setattr(module, "_benchmark_predictions_cache", None)
    monkeypatch.setattr(module, "_benchmark_predictions_cache_time", 0)
    def predict(data, positions, **kwargs):
        if data.get("supplements"):
            assert kwargs["game_like_model_data"] == {"remote": "game-like"}
            assert kwargs["stability_probe_model_data"] == {"remote": "stability"}
            assert data["results"]
            return 1234
        return 0
    monkeypatch.setattr(module, "predict_rating_from_model_data_with_supplement", predict)
    store = object.__new__(module.RatingStore)
    store._use_firestore = True
    assert store._load_benchmark_predictions(force_refresh=True)[pid] == 1234


@pytest.mark.parametrize("remote_state", ["missing", "stale", "empty", "unavailable"])
def test_rating_reader_falls_back_per_model(monkeypatch, remote_state):
    from types import SimpleNamespace
    import firebase_client
    import rating.rating_store as module

    pid = "gemini-3.1-pro-preview (medium)"
    store = object.__new__(module.RatingStore)
    store._use_firestore = False
    expected = store._load_benchmark_predictions(force_refresh=True)[pid]
    record = json.loads(CORE_RESULTS_PATH.read_text())[pid]
    if remote_state == "stale":
        record["results"] = []
    docs = [] if remote_state == "empty" else [SimpleNamespace(
        id=pid if remote_state == "stale" else "unrelated", to_dict=lambda: record)]
    def client():
        if remote_state == "unavailable":
            raise RuntimeError("offline")
        return SimpleNamespace(collection=lambda name: SimpleNamespace(stream=lambda: docs))
    monkeypatch.setattr(firebase_client, "get_firestore_client", client)
    store._use_firestore = True
    assert store._load_benchmark_predictions(force_refresh=True).get(pid) == expected
