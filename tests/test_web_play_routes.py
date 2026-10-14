import copy
from html.parser import HTMLParser
from http.cookies import SimpleCookie
from types import SimpleNamespace

import pytest
import yaml

import web.app as web_app
import web.play_service as play_service
import web.approved_players as approved_players
from web.lichess import LichessClassicalSnapshot


class FakePlayerStore:
    def __init__(self):
        self.records = {}

    def get_player(self, email):
        return self.records.get(str(email).strip().casefold())

    def save_lichess_username(self, email, username):
        normalized = str(email).strip().casefold()
        record = self.records.setdefault(normalized, {"email": normalized})
        record["lichess_username"] = username
        return record

    def claim_lichess_username(self, email, username, *, allow_missing=False):
        return self.save_lichess_username(email, username)


class ParentClassParser(HTMLParser):
    """Capture each element ID's immediate parent classes."""

    VOID_TAGS = {
        "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
        "meta", "param", "source", "track", "wbr",
    }

    def __init__(self):
        super().__init__()
        self.stack = []
        self.parent_classes = {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        element_id = attributes.get("id")
        if element_id:
            parent = self.stack[-1][1] if self.stack else {}
            self.parent_classes[element_id] = set(parent.get("class", "").split())
        if tag not in self.VOID_TAGS:
            self.stack.append((tag, attributes))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                return


@pytest.fixture
def client(tmp_path, monkeypatch, play_store):
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "llms": [{
                "player_id": "route-model",
                "model_name": "provider/route-model",
                "temperature": 0.0,
            }]
        })
    )
    web_app.app.config.update(
        TESTING=True,
        SECRET_KEY="test-secret",
        SESSION_COOKIE_SECURE=False,
    )
    monkeypatch.setattr(web_app, "CONFIG_PATH", config_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("ADMIN_EMAILS", "johnstondaniel4@gmail.com")
    store = FakePlayerStore()
    monkeypatch.setattr(web_app, "get_approved_player_store", lambda: store)
    monkeypatch.setattr(web_app, "get_play_recovery_store", lambda: SimpleNamespace(lookup=lambda *_: None))
    monkeypatch.setattr(approved_players, "get_approved_player_store", lambda: store)
    monkeypatch.setattr(
        web_app,
        "fetch_classical_snapshot",
        lambda username: LichessClassicalSnapshot(
            username=str(username).strip(),
            rating=1847,
            rating_deviation=73,
            provisional=False,
        ),
        raising=False,
    )
    with web_app.app.test_client() as test_client:
        test_client.player_store = store
        test_client.play_store = play_store
        yield test_client


def _set_user(client, email="johnstondaniel4@gmail.com") -> str:
    with client.session_transaction() as flask_session:
        flask_session["user"] = {
            "uid": "firebase-admin",
            "email": email,
            "email_verified": True,
            "name": "Daniel",
            "picture": "",
        }
        flask_session["csrf_token"] = "c" * 43
        return flask_session["csrf_token"]


def _move_payload(client, move):
    state = client.play_store.load("johnstondaniel4@gmail.com")
    if state is None:
        return {"move": move}
    return {"move": move, "game_id": state["game_id"],
            "expected_ply": len(state["moves"]), "expected_fen": play_service.game_view(state)["fen"]}


def _save_test_state(client, state):
    claim = client.play_store.begin("johnstondaniel4@gmail.com", {}, starting=True)
    client.play_store.finish(claim, state)


def test_admin_play_page_lists_configured_models(client):
    _set_user(client)

    response = client.get("/admin/play")

    assert response.status_code == 200
    assert b"Play an LLM" in response.data
    assert b"route-model" in response.data


def test_approved_non_admin_can_open_player_arena(client):
    client.player_store.records["player@example.com"] = {
        "email": "player@example.com",
        "lichess_username": "Some_Player",
    }
    _set_user(client, "player@example.com")

    response = client.get("/play")

    assert response.status_code == 200
    assert b"Some_Player" in response.data
    assert b'readonly' in response.data


def test_player_cannot_switch_linked_lichess_account(client, monkeypatch):
    client.player_store.records["player@example.com"] = {
        "email": "player@example.com",
        "lichess_username": "Some_Player",
    }
    csrf = _set_user(client, "player@example.com")
    monkeypatch.setattr(
        web_app,
        "fetch_classical_snapshot",
        lambda _username: pytest.fail("locked username must be checked before Lichess lookup"),
    )

    response = client.post(
        "/api/play/start",
        json={"model_id": "route-model", "human_color": "white", "lichess_username": "Other_Player"},
        headers={"X-CSRF-Token": csrf},
    )

    assert response.status_code == 400
    assert "already linked" in response.get_json()["error"]


def test_admin_play_page_offers_a_labeled_keyboard_move_path(client):
    _set_user(client)

    response = client.get("/admin/play")
    html = response.get_data(as_text=True)

    assert 'aria-label="Keyboard move"' in html
    assert '<input id="keyboard-move"' in html
    assert '<button id="play-keyboard-move"' in html


def test_admin_play_page_offers_effort_separately_from_model(client):
    _set_user(client)

    html = client.get("/admin/play").get_data(as_text=True)

    assert '<fieldset class="effort-choice"' in html
    assert '<legend>Effort</legend>' in html
    assert 'name="reasoning_effort"' in html


def test_admin_play_page_offers_position_export_controls(client):
    _set_user(client)

    html = client.get("/admin/play").get_data(as_text=True)

    assert "Position data" in html
    assert '<button id="copy-fen"' in html
    assert '<button id="copy-pgn"' in html
    assert '<button id="download-pgn"' in html


def test_thinking_status_is_outside_the_board_frame(client):
    _set_user(client)

    html = client.get("/admin/play").get_data(as_text=True)
    parser = ParentClassParser()
    parser.feed(html)

    assert "thinking-rail" in parser.parent_classes
    assert "play-board-frame" not in parser.parent_classes["thinking-rail"]


def test_play_page_offers_move_history_controls(client):
    _set_user(client)

    html = client.get("/admin/play").get_data(as_text=True)

    assert 'id="move-start"' in html
    assert 'id="move-previous"' in html
    assert 'id="move-next"' in html
    assert 'id="move-live"' in html
    assert 'id="move-position"' in html


def test_anonymous_user_cannot_start_game(client):
    response = client.post(
        "/api/admin/play/start",
        json={"model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player"},
    )

    assert response.status_code == 403
    assert response.get_json()["error"] == "Approved player access is required."


def test_start_game_requires_csrf(client):
    _set_user(client)

    response = client.post(
        "/api/admin/play/start",
        json={"model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player"},
    )

    assert response.status_code == 403


def test_admin_can_start_game_and_state_is_saved_in_session(client):
    csrf_token = _set_user(client)

    response = client.post(
        "/api/admin/play/start",
        json={"model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player"},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["game"]["fen"].startswith("rnbqkbnr/pppppppp/")
    assert payload["game"]["turn"] == "human"
    assert payload["game"]["model_id"] == "route-model"
    assert payload["game"]["human_profile"] == {
        "username": "Some_Player",
        "rating": 1847,
        "rating_deviation": 73,
        "provisional": False,
        "rating_pool": "classical",
    }
    with client.session_transaction() as flask_session:
        assert client.play_store.load("johnstondaniel4@gmail.com")["moves"] == []


def test_admin_can_choose_reasoning_effort_separately(client):
    csrf_token = _set_user(client)

    response = client.post(
        "/api/admin/play/start",
        json={
            "model_id": "route-model",
            "human_color": "white",
            "reasoning_effort": "default",
            "lichess_username": "Some_Player",
        },
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    assert response.get_json()["game"]["reasoning_effort"] == "default"
    with client.session_transaction() as flask_session:
        assert client.play_store.load("johnstondaniel4@gmail.com")["reasoning_effort"] == "default"


def test_admin_move_uses_saved_state_and_returns_authoritative_position(
    client,
    monkeypatch,
):
    csrf_token = _set_user(client)
    start_response = client.post(
        "/api/admin/play/start",
        json={"model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player"},
        headers={"X-CSRF-Token": csrf_token},
    )
    assert start_response.status_code == 200
    monkeypatch.setattr(
        play_service,
        "_default_move_provider",
        lambda *_args: "e7e5",
    )

    response = client.post(
        "/api/admin/play/move",
        json=_move_payload(client, "e2e4"),
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    assert response.get_json()["game"]["moves"] == ["e2e4", "e7e5"]
    assert response.get_json()["game"]["san_moves"] == ["e4", "e5"]
    with client.session_transaction() as flask_session:
        assert client.play_store.load("johnstondaniel4@gmail.com")["moves"] == ["e2e4", "e7e5"]


@pytest.mark.parametrize("poll_path, poll_status", [
    ("/api/live-games", 200),
    ("/api/game/11111111-1111-4111-8111-111111111111", 404),
    ("/api/games", 200),
    ("/api/leaderboard", 200),
    ("/static/js/live.js", 200),
    ("/live", 200),
    ("/play", 200),
    ("/login", 302),
    ("/games", 200),
    ("/", 200),
])
def test_delayed_public_read_cannot_rewind_confirmed_play_moves(
    client, monkeypatch, poll_path, poll_status,
):
    """A read response arriving late must not replace the newer game cookie."""
    csrf = _set_user(client)
    with client.session_transaction() as saved:
        saved.permanent = True
    headers = {"X-CSRF-Token": csrf}
    assert client.post("/api/play/start", headers=headers, json={
        "model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player",
    }).status_code == 200
    replies = {"e2e4": "e7e5", "g1f3": "b8c6", "f1c4": "g8f6"}
    monkeypatch.setattr(
        play_service, "_default_move_provider",
        lambda _model, board, *_args: replies[board.peek().uci()],
    )
    assert client.post("/api/play/move", headers=headers, json=_move_payload(client, "e2e4")).status_code == 200
    cookie_name = web_app.app.config["SESSION_COOKIE_NAME"]
    previous_cookie = client.get_cookie(cookie_name).value
    monkeypatch.setattr(web_app, "get_live_game_store", lambda: SimpleNamespace(list_games=lambda **_: []))
    monkeypatch.setattr(web_app, "get_game", lambda _game_id: None)
    monkeypatch.setattr(web_app, "get_all_games", lambda **_: [])
    monkeypatch.setattr(web_app, "get_leaderboard_data", lambda **_: [])

    # Another tab sends its poll before Nf3, but delivers its response afterward.
    with web_app.app.test_client() as reader:
        reader.set_cookie(cookie_name, previous_cookie)
        delayed_response = reader.get(poll_path)
    assert delayed_response.status_code == poll_status
    move_response = client.post("/api/play/move", headers=headers, json=_move_payload(client, "g1f3"))
    assert move_response.get_json()["game"]["san_moves"] == ["e4", "e5", "Nf3", "Nc6"]
    for header in delayed_response.headers.getlist("Set-Cookie"):
        cookies = SimpleCookie(header)
        if cookie_name in cookies:
            client.set_cookie(cookie_name, cookies[cookie_name].value)

    response = client.post("/api/play/move", headers=headers, json=_move_payload(client, "f1c4"))

    assert response.status_code == 200
    assert response.get_json()["game"]["san_moves"] == ["e4", "e5", "Nf3", "Nc6", "Bc4", "Nf6"]


def test_player_can_restore_an_approved_turn_without_model_calls_or_rerating(client, monkeypatch):
    csrf = _set_user(client)
    headers = {"X-CSRF-Token": csrf}
    assert client.post("/api/play/start", headers=headers, json={
        "model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player",
    }).status_code == 200
    with client.session_transaction() as saved:
        state = copy.deepcopy(client.play_store.load("johnstondaniel4@gmail.com"))
        state["moves"] = ["e2e4", "e7e5", "f1c4", "b8c6"]
        _save_test_state(client, state)
    restored = {**state, "moves": ["e2e4", "e7e5", "g1f3", "g8f6"]}
    monkeypatch.setattr(web_app, "get_play_recovery_store", lambda: SimpleNamespace(
        restore=lambda *_args: copy.deepcopy(restored),
    ), raising=False)
    monkeypatch.setattr(web_app, "record_human_challenge", lambda *_: pytest.fail("Recovery must not rate a game"))
    monkeypatch.setattr(web_app, "play_human_move", lambda *_: pytest.fail("Recovery must not request an LLM move"))

    response = client.post("/api/play/recover", headers=headers, json={
        **_move_payload(client, ""), "recovery_id": "approved-recovery",
    })

    assert response.status_code == 200
    game = response.get_json()["game"]
    assert game["game_id"] == state["game_id"]
    assert game["san_moves"] == ["e4", "e5", "Nf3", "Nf6"]
    assert game["turn"] == "human"
    with client.session_transaction() as saved:
        assert client.play_store.load("johnstondaniel4@gmail.com")["human_profile"] == state["human_profile"]


def test_duplicate_or_overlapping_requests_call_model_only_once(client, monkeypatch):
    csrf = _set_user(client)
    headers = {"X-CSRF-Token": csrf}
    client.post("/api/play/start", headers=headers, json={
        "model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player",
    })
    payload = _move_payload(client, "e2e4")
    calls = []
    def provider(*_):
        calls.append(1)
        with web_app.app.test_client() as other_tab:
            other_tab.set_cookie("session", client.get_cookie("session").value)
            assert other_tab.post("/api/play/move", headers=headers, json=payload).status_code == 409
        return "e7e5"
    monkeypatch.setattr(play_service, "_default_move_provider", provider)
    assert client.post("/api/play/move", headers=headers, json=payload).status_code == 200
    duplicate = client.post("/api/play/move", headers=headers, json=payload)
    assert duplicate.status_code == 409
    assert duplicate.get_json()["game"]["san_moves"] == ["e4", "e5"]
    assert calls == [1]


def test_reinstalled_legacy_cookie_cannot_replace_saved_game(client, monkeypatch):
    csrf = _set_user(client)
    headers = {"X-CSRF-Token": csrf}
    client.post("/api/play/start", headers=headers, json={
        "model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player",
    })
    old_state = client.play_store.load("johnstondaniel4@gmail.com")
    monkeypatch.setattr(play_service, "_default_move_provider", lambda *_: "e7e5")
    response = client.post("/api/play/move", headers=headers, json=_move_payload(client, "e2e4"))
    assert response.status_code == 200
    with client.session_transaction() as saved:
        saved["admin_play_game"] = old_state
    assert '"san_moves": ["e4", "e5"]' in client.get("/play").get_data(as_text=True)
    with client.session_transaction() as saved:
        assert "admin_play_game" not in saved


def test_missing_position_guard_is_rejected_before_provider_call(client, monkeypatch):
    csrf = _set_user(client)
    headers = {"X-CSRF-Token": csrf}
    client.post("/api/play/start", headers=headers, json={
        "model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player",
    })
    monkeypatch.setattr(web_app, "play_human_move", lambda *_: pytest.fail("Unguarded move must not reach model"))
    assert client.post("/api/play/move", headers=headers, json={"move": "e2e4"}).status_code == 409


def test_paid_turn_is_recovered_after_commit_failure_without_another_model_call(client, monkeypatch):
    csrf = _set_user(client)
    headers = {"X-CSRF-Token": csrf}
    client.post("/api/play/start", headers=headers, json={
        "model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player",
    })
    calls = []
    def provider(*_):
        calls.append(1)
        client.play_store._db.fail_next_write = True
        return "e7e5"
    monkeypatch.setattr(play_service, "_default_move_provider", provider)
    response = client.post("/api/play/move", headers=headers, json=_move_payload(client, "e2e4"))
    assert response.status_code == 503
    assert '"san_moves": ["e4", "e5"]' in client.get("/play").get_data(as_text=True)
    assert calls == [1]


def test_recovery_commit_failure_keeps_corrected_outcome_for_reload(client, monkeypatch):
    csrf = _set_user(client)
    headers = {"X-CSRF-Token": csrf}
    client.post("/api/play/start", headers=headers, json={
        "model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player",
    })
    original = client.play_store.load("johnstondaniel4@gmail.com")
    corrected = {**original, "moves": ["e2e4", "e7e5"]}
    def restore(*_):
        client.play_store._db.fail_next_write = True
        return corrected
    monkeypatch.setattr(web_app, "get_play_recovery_store", lambda: SimpleNamespace(restore=restore, lookup=lambda *_: None))
    response = client.post("/api/play/recover", headers=headers, json={
        **_move_payload(client, ""), "recovery_id": "approved-recovery",
    })
    assert response.status_code == 503
    assert '"san_moves": ["e4", "e5"]' in client.get("/play").get_data(as_text=True)
    assert client.play_store.load("johnstondaniel4@gmail.com")["game_id"] == original["game_id"]


def test_new_game_cannot_discard_a_finished_game_with_pending_rating(client, monkeypatch):
    csrf = _set_user(client)
    headers = {"X-CSRF-Token": csrf}
    setup = {"model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player"}
    first = client.post("/api/play/start", headers=headers, json=setup).get_json()["game"]
    monkeypatch.setattr(play_service, "_default_move_provider", lambda *_: "invalid")
    def fail_publication(*_):
        raise OSError("Temporary publication outage")
    monkeypatch.setattr(web_app, "record_human_challenge", fail_publication)
    completed = client.post("/api/play/move", headers=headers, json=_move_payload(client, "e2e4"))
    assert completed.status_code == 503
    assert completed.get_json()["game"]["status"] == "finished"
    assert client.post("/api/play/start", headers=headers, json=setup).status_code == 503
    assert client.play_store.load("johnstondaniel4@gmail.com")["game_id"] == first["game_id"]
    calls = []
    monkeypatch.setattr(web_app, "record_human_challenge", lambda state, email: calls.append(state["game_id"]) or {"recorded": True})
    assert client.get("/play").status_code == 200
    assert calls == [first["game_id"]]
    assert client.play_store.load("johnstondaniel4@gmail.com")["rating_result"] == {"recorded": True}


def test_recovery_requires_authentication_and_csrf(client):
    assert client.post("/api/play/recover", json={}).status_code == 403
    _set_user(client)
    assert client.post("/api/play/recover", json={}).status_code == 403


def test_stale_recovery_request_does_not_replace_current_cookie(client, monkeypatch):
    csrf = _set_user(client)
    with client.session_transaction() as saved:
        saved.permanent = True
        _save_test_state(client, {"game_id": "current-game", "moves": [], "human_color": "white", "status": "active"})
    monkeypatch.setattr(web_app, "get_play_recovery_store", lambda: pytest.fail("Stale request must not reach store"))

    response = client.post("/api/play/recover", headers={"X-CSRF-Token": csrf}, json={
        "game_id": "older-game", "recovery_id": "approval",
    })

    assert response.status_code == 409
    assert not response.headers.getlist("Set-Cookie")
    with client.session_transaction() as saved:
        assert client.play_store.load("johnstondaniel4@gmail.com")["game_id"] == "current-game"


def test_play_page_offers_approved_recovery_only_for_its_signed_game(client, monkeypatch):
    csrf = _set_user(client)
    client.post("/api/play/start", headers={"X-CSRF-Token": csrf}, json={
        "model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player",
    })
    calls = []
    monkeypatch.setattr(web_app, "get_play_recovery_store", lambda: SimpleNamespace(
        lookup=lambda state, email: calls.append((state, email)) or {"id": "approved-recovery"},
    ))
    page = client.get("/play")
    assert page.status_code == 200
    assert '"recovery": {"id": "approved-recovery"}' in page.get_data(as_text=True)
    assert calls[0][0]["human_profile"]["rating"] == 1847
    assert calls[0][1] == "johnstondaniel4@gmail.com"


def test_invalid_start_request_returns_actionable_error(client):
    csrf_token = _set_user(client)

    response = client.post(
        "/api/admin/play/start",
        json={"model_id": "not-configured", "human_color": "white", "lichess_username": "Some_Player"},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "That model is not available for web play."


def test_move_without_active_game_returns_400(client):
    csrf_token = _set_user(client)

    response = client.post(
        "/api/admin/play/move",
        json=_move_payload(client, "e2e4"),
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Start a new game first."


def test_provider_failure_keeps_previous_session_state(client, monkeypatch):
    csrf_token = _set_user(client)
    assert client.post(
        "/api/admin/play/start",
        json={"model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player"},
        headers={"X-CSRF-Token": csrf_token},
    ).status_code == 200
    with client.session_transaction() as flask_session:
        original = copy.deepcopy(client.play_store.load("johnstondaniel4@gmail.com"))

    def fail_provider(*_args):
        raise play_service.ProviderError("The LLM could not provide a move.")

    monkeypatch.setattr(play_service, "_default_move_provider", fail_provider)

    response = client.post(
        "/api/admin/play/move",
        json=_move_payload(client, "e2e4"),
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 502
    assert response.get_json()["error"].startswith("The LLM could not provide a move. Reference: ")
    with client.session_transaction() as flask_session:
        original["llm_accounting_status"] = "partial"
        assert client.play_store.load("johnstondaniel4@gmail.com") == original


def test_failed_retry_preserves_billed_usage_without_advancing_board(client, monkeypatch):
    csrf_token = _set_user(client)
    headers = {"X-CSRF-Token": csrf_token}
    assert client.post("/api/play/start", headers=headers, json={
        "model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player",
    }).status_code == 200
    attempts = iter([
        {"move": "e7e4", "tokens": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}},
        play_service.TransientAPIError("unavailable"),
        {"move": "e7e5", "tokens": {"prompt_tokens": 200, "completion_tokens": 70, "total_tokens": 270}},
    ])
    def provider(*_):
        attempt = next(attempts)
        if isinstance(attempt, Exception):
            raise attempt
        return attempt
    monkeypatch.setattr(play_service, "_default_move_provider", provider)
    assert client.post("/api/play/move", headers=headers, json=_move_payload(client, "e2e4")).status_code == 502
    with client.session_transaction() as saved:
        state = client.play_store.load("johnstondaniel4@gmail.com")
        assert state["moves"] == []
        assert state["llm_tokens"]["total_tokens"] == 150
        assert state["llm_accounting_status"] == "partial"
    assert client.post("/api/play/move", headers=headers, json=_move_payload(client, "e2e4")).status_code == 200
    with client.session_transaction() as saved:
        state = client.play_store.load("johnstondaniel4@gmail.com")
        assert state["moves"] == ["e2e4", "e7e5"]
        assert state["llm_tokens"] == {"prompt_tokens": 300, "completion_tokens": 120, "total_tokens": 420}
        assert state["llm_accounting_status"] == "partial"


def test_finished_game_is_scored_and_rating_summary_is_returned(client, monkeypatch):
    csrf_token = _set_user(client)
    assert client.post(
        "/api/play/start",
        json={"model_id": "route-model", "human_color": "white", "lichess_username": "Some_Player"},
        headers={"X-CSRF-Token": csrf_token},
    ).status_code == 200
    monkeypatch.setattr(play_service, "_default_move_provider", lambda *_args: "invalid")
    calls = []
    monkeypatch.setattr(
        web_app,
        "record_human_challenge",
        lambda state, email: calls.append((copy.deepcopy(state), email)) or {
            "recorded": True,
            "model_rating": 1512,
            "model_rating_deviation": 168,
        },
        raising=False,
    )

    response = client.post(
        "/api/play/move",
        json=_move_payload(client, "e2e4"),
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    game = response.get_json()["game"]
    assert game["status"] == "finished"
    assert game["winner"] == "human"
    assert game["rating_result"] == {
        "recorded": True,
        "model_rating": 1512,
        "model_rating_deviation": 168,
    }
    assert calls[0][1] == "johnstondaniel4@gmail.com"


def test_game_finished_during_opening_model_turn_is_scored(client, monkeypatch):
    csrf_token = _set_user(client)
    monkeypatch.setattr(play_service, "_default_move_provider", lambda *_args: "invalid")
    calls = []
    monkeypatch.setattr(
        web_app,
        "record_human_challenge",
        lambda state, email: calls.append((copy.deepcopy(state), email)) or {
            "recorded": True,
            "model_rating": 1188,
            "model_rating_deviation": 170,
        },
    )

    response = client.post(
        "/api/play/start",
        json={"model_id": "route-model", "human_color": "black", "lichess_username": "Some_Player"},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    game = response.get_json()["game"]
    assert game["status"] == "finished"
    assert game["rating_result"]["recorded"] is True
    assert len(calls) == 1
