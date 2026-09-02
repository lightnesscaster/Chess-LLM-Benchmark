import pytest

from web.lichess import LichessLookupError, fetch_rapid_snapshot


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


def test_fetch_rapid_snapshot_uses_canonical_username_rating_and_rd():
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
            }
        },
    }))

    snapshot = fetch_rapid_snapshot(" some_player ", session=session)

    assert snapshot.to_dict() == {
        "username": "Some_Player",
        "rating": 1847,
        "rating_deviation": 73,
        "provisional": True,
    }
    assert session.calls == [(
        "https://lichess.org/api/user/some_player",
        {
            "headers": {"Accept": "application/json", "User-Agent": "ChessBenchLLM/1.0"},
            "timeout": 8,
        },
    )]


def test_fetch_rapid_snapshot_rejects_account_without_rapid_rating():
    session = FakeSession(FakeResponse(200, {
        "id": "new_player",
        "username": "New_Player",
        "perfs": {"blitz": {"games": 2, "rating": 1500, "rd": 300}},
    }))

    with pytest.raises(LichessLookupError, match="no Rapid rating"):
        fetch_rapid_snapshot("new_player", session=session)


def test_fetch_rapid_snapshot_reports_missing_account():
    session = FakeSession(FakeResponse(404, {"error": "Not found"}))

    with pytest.raises(LichessLookupError, match="not found"):
        fetch_rapid_snapshot("missing-player", session=session)


@pytest.mark.parametrize("username", ["", "   ", "x" * 65])
def test_fetch_rapid_snapshot_rejects_invalid_username_without_request(username):
    session = FakeSession(FakeResponse(200, {}))

    with pytest.raises(LichessLookupError, match="valid Lichess username"):
        fetch_rapid_snapshot(username, session=session)

    assert session.calls == []


def test_fetch_rapid_snapshot_rejects_malformed_profile():
    session = FakeSession(FakeResponse(200, {"username": "Broken", "perfs": {"rapid": {"rating": "high", "rd": None}}}))

    with pytest.raises(LichessLookupError, match="valid Rapid rating"):
        fetch_rapid_snapshot("broken", session=session)
