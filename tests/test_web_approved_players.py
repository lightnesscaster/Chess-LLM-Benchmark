import pytest

import web.app as web_app
from web.approved_players import ApprovedPlayerStore, is_approved_player


class FakeApprovedPlayerStore:
    def __init__(self):
        self.records = {}

    def list_players(self):
        return [self.records[email] for email in sorted(self.records)]

    def get_player(self, email):
        return self.records.get(email.strip().casefold())

    def approve(self, email, approved_by):
        normalized = email.strip().casefold()
        if not normalized or "@" not in normalized:
            raise ValueError("Enter a valid email address.")
        self.records.setdefault(
            normalized,
            {"email": normalized, "approved_by": approved_by, "lichess_username": None},
        )
        return self.records[normalized]

    def remove(self, email):
        return self.records.pop(email.strip().casefold(), None) is not None


class FakeSnapshot:
    def __init__(self, data):
        self._data = data
        self.exists = data is not None

    def to_dict(self):
        return dict(self._data or {})


class FakeDocument:
    def __init__(self, records, document_id):
        self.records = records
        self.id = document_id

    def get(self, transaction=None):
        return FakeSnapshot(self.records.get(self.id))

    def set(self, data, merge=False):
        if merge:
            self.records[self.id] = {**self.records.get(self.id, {}), **data}
        else:
            self.records[self.id] = dict(data)


class FakeCollection:
    def __init__(self, records):
        self.records = records

    def document(self, document_id):
        return FakeDocument(self.records, document_id)


class FakeTransaction:
    def set(self, reference, data, merge=False):
        reference.set(data, merge=merge)


class FakeFirestore:
    def __init__(self):
        self.records = {}

    def collection(self, _name):
        return FakeCollection(self.records)

    def transaction(self):
        return FakeTransaction()


@pytest.fixture
def approved_client(monkeypatch):
    store = FakeApprovedPlayerStore()
    web_app.app.config.update(
        TESTING=True,
        SECRET_KEY="test-secret",
        SESSION_COOKIE_SECURE=False,
    )
    monkeypatch.setenv("ADMIN_EMAILS", "johnstondaniel4@gmail.com")
    monkeypatch.setattr(
        web_app,
        "get_approved_player_store",
        lambda: store,
        raising=False,
    )
    with web_app.app.test_client() as client:
        yield client, store


def _set_user(client, email):
    with client.session_transaction() as flask_session:
        flask_session["user"] = {
            "uid": f"uid:{email}",
            "email": email,
            "email_verified": True,
            "name": "Player",
            "picture": "",
        }
        flask_session["csrf_token"] = "c" * 43
        return flask_session["csrf_token"]


def test_admin_players_page_lists_approved_accounts(approved_client):
    client, store = approved_client
    store.approve("Player@Example.com", "johnstondaniel4@gmail.com")
    _set_user(client, "johnstondaniel4@gmail.com")

    response = client.get("/admin/players")

    assert response.status_code == 200
    assert "Approved players" in response.get_data(as_text=True)
    assert "player@example.com" in response.get_data(as_text=True)


def test_non_admin_cannot_manage_approved_accounts(approved_client):
    client, _store = approved_client
    _set_user(client, "player@example.com")

    assert client.get("/admin/players").status_code == 403


def test_admin_can_approve_a_normalized_email(approved_client):
    client, store = approved_client
    csrf = _set_user(client, "johnstondaniel4@gmail.com")

    response = client.post(
        "/api/admin/players",
        json={"email": "  Player@Example.com "},
        headers={"X-CSRF-Token": csrf},
    )

    assert response.status_code == 201
    assert response.get_json()["player"]["email"] == "player@example.com"
    assert store.get_player("player@example.com") is not None


def test_admin_can_remove_an_approved_email(approved_client):
    client, store = approved_client
    store.approve("player@example.com", "johnstondaniel4@gmail.com")
    csrf = _set_user(client, "johnstondaniel4@gmail.com")

    response = client.delete(
        "/api/admin/players/player@example.com",
        headers={"X-CSRF-Token": csrf},
    )

    assert response.status_code == 200
    assert response.get_json() == {"removed": True}
    assert store.get_player("player@example.com") is None


def test_admin_player_changes_require_csrf(approved_client):
    client, _store = approved_client
    _set_user(client, "johnstondaniel4@gmail.com")

    response = client.post(
        "/api/admin/players",
        json={"email": "player@example.com"},
    )

    assert response.status_code == 403


def test_configured_admin_is_implicitly_approved(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "admin@example.com")
    user = {"email": "ADMIN@example.com", "email_verified": True}

    assert is_approved_player(user, store=FakeApprovedPlayerStore()) is True


def test_allowlisted_verified_account_is_approved(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "admin@example.com")
    store = FakeApprovedPlayerStore()
    store.approve("player@example.com", "admin@example.com")
    user = {"email": "Player@example.com", "email_verified": True}

    assert is_approved_player(user, store=store) is True


def test_unverified_allowlisted_account_is_not_approved(monkeypatch):
    monkeypatch.setenv("ADMIN_EMAILS", "admin@example.com")
    store = FakeApprovedPlayerStore()
    store.approve("player@example.com", "admin@example.com")
    user = {"email": "player@example.com", "email_verified": False}

    assert is_approved_player(user, store=store) is False


def test_first_lichess_username_claim_is_atomic_and_cannot_be_overwritten(monkeypatch):
    from firebase_admin import firestore

    monkeypatch.setattr(firestore, "transactional", lambda function: function)
    store = ApprovedPlayerStore(db=FakeFirestore())
    store.approve("player@example.com", "admin@example.com")

    claimed = store.claim_lichess_username("player@example.com", "Some_Player")

    assert claimed["lichess_username"] == "Some_Player"
    with pytest.raises(ValueError, match="already linked"):
        store.claim_lichess_username("player@example.com", "Other_Player")
    assert store.get_player("player@example.com")["lichess_username"] == "Some_Player"


def test_username_claim_cannot_recreate_a_revoked_player(monkeypatch):
    from firebase_admin import firestore

    monkeypatch.setattr(firestore, "transactional", lambda function: function)
    store = ApprovedPlayerStore(db=FakeFirestore())

    with pytest.raises(ValueError, match="approval is required"):
        store.claim_lichess_username("revoked@example.com", "Some_Player")

    assert store.get_player("revoked@example.com") is None


def test_implicit_admin_can_claim_username_without_player_record(monkeypatch):
    from firebase_admin import firestore

    monkeypatch.setattr(firestore, "transactional", lambda function: function)
    store = ApprovedPlayerStore(db=FakeFirestore())

    record = store.claim_lichess_username(
        "admin@example.com",
        "Admin_Player",
        allow_missing=True,
    )

    assert record["lichess_username"] == "Admin_Player"
