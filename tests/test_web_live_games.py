import importlib

import pytest

import web.app as web_app


def local_store(tmp_path, monkeypatch):
    spec = importlib.util.find_spec("game.live_games")
    assert spec is not None, "Live snapshot backend is missing"
    live = importlib.import_module("game.live_games")
    store = live.LiveGameStore(path=tmp_path, use_firestore=False)
    monkeypatch.setattr(web_app, "get_live_game_store", lambda: store, raising=False)
    return live, store


def test_anonymous_live_api_reads_sanitized_unexpired_games(tmp_path, monkeypatch):
    live, store = local_store(tmp_path, monkeypatch)
    import chess
    import time
    now = time.time()
    store.write(live.build_snapshot("current", "<img src=x>", "a/b (high)", chess.Board(), started_at=now, now=now))
    store.write(live.build_snapshot("expired", "w", "b", chess.Board(), started_at=now-3000, now=now-3000))
    with web_app.app.test_client() as client:
        response = client.get("/api/live-games")
    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    games = response.get_json()["games"]
    assert [g["game_id"] for g in games] == ["current"]
    assert games[0]["white"] == "<img src=x>"


def test_empty_live_list_is_distinct_from_unavailable_storage(tmp_path, monkeypatch):
    _, store = local_store(tmp_path, monkeypatch)
    with web_app.app.test_client() as client:
        response = client.get("/api/live-games")
        assert response.status_code == 200
        assert response.get_json() == {"games": []}
        monkeypatch.setattr(store, "list_games", lambda: (_ for _ in ()).throw(OSError("private failure")))
        response = client.get("/api/live-games")
    assert response.status_code == 503
    assert response.get_json() == {"error": "Live games are temporarily unavailable."}
    assert response.headers["Cache-Control"] == "no-store"


def test_live_page_is_anonymous_and_does_not_load_history(monkeypatch):
    monkeypatch.setitem(web_app.app.config, "SECRET_KEY", "live-page-test-only")
    web_app.app.jinja_env.cache.clear()
    monkeypatch.setattr(web_app, "get_leaderboard_data", lambda *a, **k: pytest.fail("history loaded"))
    monkeypatch.setattr(web_app, "get_all_games", lambda *a, **k: pytest.fail("history loaded"))
    with web_app.app.test_client() as client:
        response = client.get("/live")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="live-app"' in html
    assert '/static/js/live.js' in html
    assert 'href="/live"' in html
