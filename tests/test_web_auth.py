import pytest

import web.app as web_app
from web.auth import safe_next_url


@pytest.fixture
def client(monkeypatch):
    web_app.app.config.update(
        TESTING=True,
        SECRET_KEY="test-secret",
        SESSION_COOKIE_SECURE=False,
    )
    monkeypatch.setenv("ADMIN_EMAILS", "johnstondaniel4@gmail.com")
    with web_app.app.test_client() as test_client:
        yield test_client


def _csrf(client) -> str:
    client.get("/login")
    with client.session_transaction() as flask_session:
        return flask_session["csrf_token"]


def _login(client, monkeypatch, *, email: str, verified: bool = True):
    token_claims = {
        "uid": "firebase-user-1",
        "email": email,
        "email_verified": verified,
        "name": "Test User",
        "picture": "https://example.test/avatar.png",
    }
    monkeypatch.setattr(
        web_app,
        "verify_firebase_token",
        lambda token: token_claims if token == "valid-token" else None,
        raising=False,
    )
    return client.post(
        "/api/auth/session",
        json={"id_token": "valid-token"},
        headers={"X-CSRF-Token": _csrf(client)},
    )


def test_login_page_initializes_csrf_session(client):
    response = client.get("/login")

    assert response.status_code == 200
    with client.session_transaction() as flask_session:
        assert len(flask_session["csrf_token"]) >= 32


def test_missing_session_secret_disables_login_without_breaking_public_pages(
    client,
    monkeypatch,
):
    web_app.app.config["SECRET_KEY"] = None
    monkeypatch.setattr(web_app, "get_leaderboard_data", lambda **_kwargs: [])

    assert client.get("/leaderboard").status_code == 200
    assert client.get("/login").status_code == 503


@pytest.mark.parametrize(
    "candidate",
    ["/\\evil.example", "/admin/\\evil.example", "//evil.example"],
)
def test_login_next_url_rejects_browser_normalized_external_paths(candidate):
    assert safe_next_url(candidate) == "/"


def test_verified_firebase_user_can_create_site_session(client, monkeypatch):
    response = _login(
        client,
        monkeypatch,
        email="member@example.com",
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "user": {
            "email": "member@example.com",
            "is_admin": False,
            "name": "Test User",
            "picture": "https://example.test/avatar.png",
        }
    }
    with client.session_transaction() as flask_session:
        assert flask_session["user"]["uid"] == "firebase-user-1"


def test_unverified_firebase_email_cannot_create_site_session(client, monkeypatch):
    response = _login(
        client,
        monkeypatch,
        email="member@example.com",
        verified=False,
    )

    assert response.status_code == 403
    assert response.get_json()["error"] == "A verified email address is required."
    with client.session_transaction() as flask_session:
        assert "user" not in flask_session


def test_auth_session_rejects_missing_csrf(client):
    response = client.post(
        "/api/auth/session",
        json={"id_token": "anything"},
    )

    assert response.status_code == 403


def test_auth_session_rejects_non_object_json(client):
    response = client.post(
        "/api/auth/session",
        json=["not", "an", "object"],
        headers={"X-CSRF-Token": _csrf(client)},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Send a valid login token."


def test_anonymous_user_is_redirected_from_admin_play(client):
    response = client.get("/admin/play")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/login?next=/admin/play")


def test_non_admin_user_is_forbidden_from_admin_play(client, monkeypatch):
    assert _login(
        client,
        monkeypatch,
        email="member@example.com",
    ).status_code == 200

    response = client.get("/admin/play")

    assert response.status_code == 403


def test_designated_admin_can_reach_admin_play(client, monkeypatch):
    assert _login(
        client,
        monkeypatch,
        email="  JohnstonDaniel4@GMAIL.COM ",
    ).status_code == 200

    response = client.get("/admin/play")

    assert response.status_code == 200


def test_logout_clears_user_session(client, monkeypatch):
    assert _login(
        client,
        monkeypatch,
        email="member@example.com",
    ).status_code == 200
    with client.session_transaction() as flask_session:
        csrf_token = flask_session["csrf_token"]

    response = client.post(
        "/logout",
        headers={"X-CSRF-Token": csrf_token},
    )

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/")
    with client.session_transaction() as flask_session:
        assert "user" not in flask_session
