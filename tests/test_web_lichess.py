import pytest

from web.lichess import LichessLookupError, fetch_classical_snapshot


class FakeResponse:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def test_rating_snapshot_uses_classical_instead_of_rapid():
    session = FakeSession(FakeResponse(200, {
        "id": "some_player",
        "username": "Some_Player",
        "perfs": {
            "rapid": {
                "games": 42,
                "rating": 1847,
                "rd": 73,
                "prog": 9,
                "prov": True,
            },
            "classical": {
                "games": 12,
                "rating": 1722,
                "rd": 118,
                "prog": -3,
                "prov": False,
            },
        },
    }))

    snapshot = fetch_classical_snapshot(" some_player ", session=session)

    assert snapshot.to_dict() == {
        "username": "Some_Player",
        "rating": 1722,
        "rating_deviation": 118,
        "provisional": False,
        "rating_pool": "classical",
    }
    assert session.calls == [(
        "https://lichess.org/api/user/some_player",
        {
            "headers": {"Accept": "application/json", "User-Agent": "ChessBenchLLM/1.0"},
            "timeout": 8,
        },
    )]


def test_fetch_classical_snapshot_rejects_account_without_classical_rating():
    session = FakeSession(FakeResponse(200, {
        "id": "new_player",
        "username": "New_Player",
        "perfs": {
            "rapid": {"games": 2, "rating": 1500, "rd": 300},
            "blitz": {"games": 2, "rating": 1500, "rd": 300},
        },
    }))

    with pytest.raises(LichessLookupError, match="no Classical rating"):
        fetch_classical_snapshot("new_player", session=session)


def test_fetch_classical_snapshot_reports_missing_account():
    session = FakeSession(FakeResponse(404, {"error": "Not found"}))

    with pytest.raises(LichessLookupError, match="not found"):
        fetch_classical_snapshot("missing-player", session=session)


@pytest.mark.parametrize("username", ["", "   ", "x" * 65])
def test_fetch_classical_snapshot_rejects_invalid_username_without_request(username):
    session = FakeSession(FakeResponse(200, {}))

    with pytest.raises(LichessLookupError, match="valid Lichess username"):
        fetch_classical_snapshot(username, session=session)

    assert session.calls == []


def test_fetch_classical_snapshot_rejects_malformed_profile():
    session = FakeSession(FakeResponse(200, {"username": "Broken", "perfs": {"classical": {"rating": "high", "rd": None}}}))

    with pytest.raises(LichessLookupError, match="valid Classical rating"):
        fetch_classical_snapshot("broken", session=session)
