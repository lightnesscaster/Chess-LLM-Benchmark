import copy

import pytest
import yaml

import web.app as web_app
import web.play_service as play_service


@pytest.fixture
def client(tmp_path, monkeypatch):
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
    with web_app.app.test_client() as test_client:
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


def test_admin_play_page_lists_configured_models(client):
    _set_user(client)

    response = client.get("/admin/play")

    assert response.status_code == 200
    assert b"Play an LLM" in response.data
    assert b"route-model" in response.data


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


def test_anonymous_user_cannot_start_game(client):
    response = client.post(
        "/api/admin/play/start",
        json={"model_id": "route-model", "human_color": "white"},
    )

    assert response.status_code == 403
    assert response.get_json()["error"] == "Administrator access is required."


def test_start_game_requires_csrf(client):
    _set_user(client)

    response = client.post(
        "/api/admin/play/start",
        json={"model_id": "route-model", "human_color": "white"},
    )

    assert response.status_code == 403


def test_admin_can_start_game_and_state_is_saved_in_session(client):
    csrf_token = _set_user(client)

    response = client.post(
        "/api/admin/play/start",
        json={"model_id": "route-model", "human_color": "white"},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["game"]["fen"].startswith("rnbqkbnr/pppppppp/")
    assert payload["game"]["turn"] == "human"
    assert payload["game"]["model_id"] == "route-model"
    with client.session_transaction() as flask_session:
        assert flask_session["admin_play_game"]["moves"] == []


def test_admin_can_choose_reasoning_effort_separately(client):
    csrf_token = _set_user(client)

    response = client.post(
        "/api/admin/play/start",
        json={
            "model_id": "route-model",
            "human_color": "white",
            "reasoning_effort": "default",
        },
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    assert response.get_json()["game"]["reasoning_effort"] == "default"
    with client.session_transaction() as flask_session:
        assert flask_session["admin_play_game"]["reasoning_effort"] == "default"


def test_admin_move_uses_saved_state_and_returns_authoritative_position(
    client,
    monkeypatch,
):
    csrf_token = _set_user(client)
    start_response = client.post(
        "/api/admin/play/start",
        json={"model_id": "route-model", "human_color": "white"},
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
        json={"move": "e2e4"},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 200
    assert response.get_json()["game"]["moves"] == ["e2e4", "e7e5"]
    assert response.get_json()["game"]["san_moves"] == ["e4", "e5"]
    with client.session_transaction() as flask_session:
        assert flask_session["admin_play_game"]["moves"] == ["e2e4", "e7e5"]


def test_invalid_start_request_returns_actionable_error(client):
    csrf_token = _set_user(client)

    response = client.post(
        "/api/admin/play/start",
        json={"model_id": "not-configured", "human_color": "white"},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "That model is not available for web play."


def test_move_without_active_game_returns_400(client):
    csrf_token = _set_user(client)

    response = client.post(
        "/api/admin/play/move",
        json={"move": "e2e4"},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Start a new game first."


def test_provider_failure_keeps_previous_session_state(client, monkeypatch):
    csrf_token = _set_user(client)
    assert client.post(
        "/api/admin/play/start",
        json={"model_id": "route-model", "human_color": "white"},
        headers={"X-CSRF-Token": csrf_token},
    ).status_code == 200
    with client.session_transaction() as flask_session:
        original = copy.deepcopy(flask_session["admin_play_game"])

    def fail_provider(*_args):
        raise play_service.ProviderError("The LLM could not provide a move.")

    monkeypatch.setattr(play_service, "_default_move_provider", fail_provider)

    response = client.post(
        "/api/admin/play/move",
        json={"move": "e2e4"},
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 502
    assert response.get_json()["error"] == "The LLM could not provide a move."
    with client.session_transaction() as flask_session:
        assert flask_session["admin_play_game"] == original
