import web.app as web_app


def test_applied_benchmark_revision_refreshes_warm_leaderboard(monkeypatch):
    from types import SimpleNamespace
    from tests.test_leaderboard_metadata import FakeRatingStore

    current = {"generation": 1, "rating": 1700}
    store = FakeRatingStore()
    original = store.get_sorted_ratings

    def ratings(**kwargs):
        rows = original(**kwargs)
        rows[0].rating = current["rating"]
        return rows

    store.get_sorted_ratings = ratings
    monkeypatch.setattr(web_app, "RatingStore", lambda **_: store)
    monkeypatch.setattr(web_app, "PGNLogger", lambda: SimpleNamespace(load_all_results=lambda: []))
    monkeypatch.setattr(web_app, "get_anchors_from_config", lambda: {})
    monkeypatch.setattr(web_app, "get_all_engine_ids_from_config", lambda: set())
    monkeypatch.setattr(web_app, "applied_benchmark_generation", lambda: current["generation"])
    for name, value in (("_leaderboard_cache", []), ("_leaderboard_cache_time", 0),
                        ("_leaderboard_cache_min_games", None), ("_leaderboard_refreshing", False),
                        ("_leaderboard_benchmark_generation", 0)):
        monkeypatch.setattr(web_app, name, value)
    assert web_app.get_leaderboard_data()[0]["rating"] == 1700
    current["rating"] = 1900
    assert web_app.get_leaderboard_data()[0]["rating"] == 1700
    current["generation"] = 2
    assert web_app.get_leaderboard_data()[0]["rating"] == 1900


def test_public_leaderboard_requests_models_after_their_first_game(monkeypatch):
    requested_minimums = []

    def fake_leaderboard_data(*, min_games, **_kwargs):
        requested_minimums.append(min_games)
        return []

    monkeypatch.setattr(web_app, "get_leaderboard_data", fake_leaderboard_data)
    web_app.app.config.update(TESTING=True)

    with web_app.app.test_client() as client:
        response = client.get("/leaderboard")

    assert response.status_code == 200
    assert requested_minimums == [1]


def test_leaderboard_cache_keeps_minimum_game_thresholds_separate(monkeypatch):
    class FakeRatingStore:
        def __init__(self, **_kwargs):
            pass

        def has_player(self, _player_id):
            return True

    class FakeStatsCollector:
        def add_results(self, _results):
            pass

    class FakePGNLogger:
        def load_all_results(self):
            return []

    class FakeLeaderboard:
        def __init__(self, _rating_store, _stats):
            pass

        def get_leaderboard(self, *, min_games, sort_by):
            return [{
                "player_id": "test-model",
                "minimum": min_games,
                "sort": sort_by,
            }]

    monkeypatch.setattr(web_app, "RatingStore", FakeRatingStore)
    monkeypatch.setattr(web_app, "StatsCollector", FakeStatsCollector)
    monkeypatch.setattr(web_app, "PGNLogger", FakePGNLogger)
    monkeypatch.setattr(web_app, "Leaderboard", FakeLeaderboard)
    monkeypatch.setattr(web_app, "get_anchors_from_config", lambda: {})
    monkeypatch.setattr(web_app, "get_all_engine_ids_from_config", lambda: set())
    monkeypatch.setattr(web_app, "_leaderboard_cache", [])
    monkeypatch.setattr(web_app, "_leaderboard_cache_time", 0)
    monkeypatch.setattr(web_app, "_leaderboard_cache_min_games", None)
    monkeypatch.setattr(web_app, "_leaderboard_refreshing", False)

    five_game_result = web_app.get_leaderboard_data(min_games=5)
    one_game_result = web_app.get_leaderboard_data(min_games=1)

    assert five_game_result[0]["minimum"] == 5
    assert one_game_result[0]["minimum"] == 1
