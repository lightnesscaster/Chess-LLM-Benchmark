"""Draw negotiation keeps human games authoritative and publishes once."""
import copy
import io
import subprocess

import chess
import chess.pgn
import pytest
import yaml

import web.app as web_app
import web.play_service as service
from web.human_challenges import _pgn_for_state, build_human_challenge_result
from test_web_play_routes import client, _set_user, _save_test_state, _move_payload

ENV = {"OPENROUTER_API_KEY": "test"}


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "draw.yaml"
    path.write_text(yaml.safe_dump({"llms": [{"player_id": "draw-model", "model_name": "test/model"}]}))
    return path


def new_game(config_path, color="white", provider=None, rated=False):
    profile = {"username": "tester", "rating": 1800, "rating_deviation": 60} if rated else None
    return service.start_game("draw-model", color, config_path, ENV, provider, human_profile=profile)


def test_human_offer_can_be_accepted_without_a_model_move(config_path):
    state = new_game(config_path)
    received = []
    def accept(model, board, *_):
        received.append((model["opponent_draw_offer"], board.ply()))
        return {"move": "accept_draw", "tokens": {"prompt_tokens": 12, "completion_tokens": 2, "total_tokens": 14}}
    result = service.play_human_move(state, "e2e4", config_path, ENV, accept, offer_draw=True)
    assert result["moves"] == ["e2e4"]
    assert result["termination"] == "draw_agreement"
    assert result["winner"] == "draw"
    assert result["draw_offer"] is None
    assert result["draw_events"] == [{"action": "offer", "by": "white", "ply": 1}, {"action": "accept", "by": "black", "ply": 1}]
    assert received == [(True, 1)]
    assert result["llm_tokens"]["total_tokens"] == 14
    assert state["moves"] == []


def test_model_offer_human_accept_keeps_board_and_exports_evidence(config_path):
    state = new_game(config_path, "black", lambda *_: "e2e4 offer_draw", rated=True)
    accepted = service.human_draw_action(state, "accept_draw", config_path, ENV)
    assert accepted["moves"] == ["e2e4"]
    assert accepted["winner"] == "draw"
    assert accepted["termination"] == "draw_agreement"
    assert state["status"] == "active"
    view = service.game_view(accepted)
    assert view["draw_offer"] is None
    pgn = chess.pgn.read_game(io.StringIO(view["pgn"]))
    assert pgn.end().comment == "[%draw_offer white] [%draw_accept black]"
    result = build_human_challenge_result(accepted, "player@example.com")
    saved_pgn = chess.pgn.read_game(io.StringIO(_pgn_for_state(accepted, result)))
    assert saved_pgn.end().comment == pgn.end().comment


def test_decline_and_counteroffer_record_at_the_respective_plies(config_path):
    state = new_game(config_path, "black", lambda *_: "e2e4 offer_draw")
    replied = service.play_human_move(state, "e7e5", config_path, ENV, lambda *_: "g1f3 offer_draw", offer_draw=True)
    assert replied["draw_offer"] == "white"
    assert replied["draw_events"] == [
        {"action": "offer", "by": "white", "ply": 1},
        {"action": "decline", "by": "black", "ply": 1},
        {"action": "offer", "by": "black", "ply": 2},
        {"action": "decline", "by": "white", "ply": 2},
        {"action": "offer", "by": "white", "ply": 3},
    ]


def test_explicit_decline_is_a_no_call_action_and_invalid_move_preserves_offer(config_path):
    state = new_game(config_path, "black", lambda *_: "e2e4 offer_draw")
    snapshot = copy.deepcopy(state)
    with pytest.raises(service.GameStateError):
        service.play_human_move(state, "e7e4", config_path, ENV, lambda *_: pytest.fail("illegal move must not call"), offer_draw=True)
    assert state == snapshot
    declined = service.human_draw_action(state, "decline_draw", config_path, ENV)
    assert declined["status"] == "active"
    assert declined["moves"] == state["moves"]
    assert declined["draw_offer"] is None
    assert declined["draw_events"][-1] == {"action": "decline", "by": "black", "ply": 1}
    with pytest.raises(service.GameStateError):
        service.human_draw_action(declined, "accept_draw", config_path, ENV)


@pytest.mark.parametrize("response", ["accept_draw", "e7e4 offer_draw"])
def test_invalid_model_offer_or_acceptance_retries_without_creating_offer(config_path, response):
    state = new_game(config_path)
    attempts = iter([response, "e7e5"])
    result = service.play_human_move(state, "e2e4", config_path, ENV, lambda *_: next(attempts))
    assert result["moves"] == ["e2e4", "e7e5"]
    assert result["llm_illegal_moves"] == 1
    assert result["draw_offer"] is None
    assert result["draw_events"] == []


def test_illegal_model_retry_preserves_human_pending_offer(config_path):
    state = new_game(config_path)
    attempts = iter(["e7e4 offer_draw", "accept_draw"])
    seen = []
    def provider(model, *_):
        seen.append(model["opponent_draw_offer"])
        return next(attempts)
    result = service.play_human_move(state, "e2e4", config_path, ENV, provider, offer_draw=True)
    assert result["termination"] == "draw_agreement"
    assert seen == [True, True]
    assert len(result["draw_events"]) == 2


def test_terminal_move_takes_precedence_over_attached_draw_offer(config_path):
    state = new_game(config_path)
    state["moves"] = ["f2f3", "e7e5", "g2g4"]
    state["human_color"] = "black"
    result = service.play_human_move(state, "d8h4", config_path, ENV, lambda *_: pytest.fail("terminal position"), offer_draw=True)
    assert result["termination"] == "checkmate"
    assert result["draw_offer"] is None
    assert result["draw_events"] == []


def test_legacy_active_state_and_wrong_side_acceptance(config_path):
    state = new_game(config_path)
    state.pop("draw_offer", None)
    state.pop("draw_events", None)
    result = service.play_human_move(state, "e2e4", config_path, ENV, lambda *_: "e7e5")
    assert result["draw_events"] == []
    with pytest.raises(service.GameStateError):
        service.human_draw_action(result, "accept_draw", config_path, ENV)
    result["draw_offer"] = "white"
    with pytest.raises(service.GameStateError):
        service.human_draw_action(result, "accept_draw", config_path, ENV)


def test_accept_route_durably_saves_then_scores_once_and_rejects_duplicate(client, monkeypatch):
    csrf = _set_user(client)
    state = service.start_game("route-model", "black", web_app.CONFIG_PATH, ENV, lambda *_: "e2e4 offer_draw",
        human_profile={"username": "tester", "rating": 1800, "rating_deviation": 60})
    _save_test_state(client, state)
    called = []
    def score(finished, _email):
        assert client.play_store.load("johnstondaniel4@gmail.com")["termination"] == "draw_agreement"
        called.append(finished["game_id"])
        return {"recorded": True, "model_rating": 1500, "model_rating_deviation": 300}
    monkeypatch.setattr(web_app, "record_human_challenge", score)
    monkeypatch.setattr(service, "_default_move_provider", lambda *_: pytest.fail("accept must never call a model"))
    payload = dict(_move_payload(client, None), action="accept_draw", draw_offer="black", moves=[])
    response = client.post("/api/play/move", json=payload, headers={"X-CSRF-Token": csrf})
    assert response.status_code == 200
    assert response.get_json()["game"]["termination"] == "draw_agreement"
    assert response.get_json()["game"]["moves"] == ["e2e4"]
    duplicate = client.post("/api/play/move", json=payload, headers={"X-CSRF-Token": csrf})
    assert duplicate.status_code == 400
    assert called == [state["game_id"]]
    with client.session_transaction() as session:
        assert "admin_play_game" not in session


def test_stale_accept_and_forged_offer_are_rejected(client, monkeypatch):
    csrf = _set_user(client)
    state = service.start_game("route-model", "black", web_app.CONFIG_PATH, ENV, lambda *_: "e2e4 offer_draw",
        human_profile={"username": "tester", "rating": 1800, "rating_deviation": 60})
    _save_test_state(client, state)
    payload = dict(_move_payload(client, None), action="accept_draw")
    stale = dict(payload, expected_ply=0, expected_fen=chess.Board().fen())
    assert client.post("/api/play/move", json=stale, headers={"X-CSRF-Token": csrf}).status_code == 409
    declined = client.post("/api/play/move", json=dict(payload, action="decline_draw"), headers={"X-CSRF-Token": csrf})
    assert declined.status_code == 200
    assert declined.get_json()["game"]["draw_offer"] is None
    forged = client.post("/api/play/move", json=dict(payload, draw_offer="white"), headers={"X-CSRF-Token": csrf})
    assert forged.status_code == 400
    assert client.play_store.load("johnstondaniel4@gmail.com")["status"] == "active"
    assert client.post("/api/play/move", json=payload).status_code == 403


def test_accept_commit_failure_replays_journal_then_rates_once(client, monkeypatch):
    csrf = _set_user(client)
    state = service.start_game("route-model", "black", web_app.CONFIG_PATH, ENV, lambda *_: "e2e4 offer_draw",
        human_profile={"username": "tester", "rating": 1800, "rating_deviation": 60})
    _save_test_state(client, state)
    original_action = web_app.human_draw_action
    def save_then_fail(*args):
        updated = original_action(*args)
        client.play_store._db.fail_next_write = True
        return updated
    monkeypatch.setattr(web_app, "human_draw_action", save_then_fail)
    recorded = []
    monkeypatch.setattr(web_app, "record_human_challenge", lambda state, _: recorded.append(state["game_id"]) or {"recorded": True})
    response = client.post("/api/play/move", json=dict(_move_payload(client, None), action="accept_draw"), headers={"X-CSRF-Token": csrf})
    assert response.status_code == 503
    assert recorded == []
    assert client.get("/play").status_code == 200
    assert client.get("/play").status_code == 200
    restored = client.play_store.load("johnstondaniel4@gmail.com")
    assert restored["moves"] == ["e2e4"]
    assert restored["termination"] == "draw_agreement"
    assert recorded == [state["game_id"]]


def test_provider_failure_rolls_back_draw_events_but_retains_paid_accounting(config_path):
    state = new_game(config_path, "black", lambda *_: "e2e4 offer_draw")
    before = copy.deepcopy(state)
    attempts = iter([{"move": "e2e5", "tokens": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}, service.ProviderError("unavailable")])
    def provider(*_):
        attempt = next(attempts)
        if isinstance(attempt, Exception):
            raise attempt
        return attempt
    with pytest.raises(service.ProviderError) as failure:
        service.play_human_move(state, "e7e5", config_path, ENV, provider, offer_draw=True)
    assert state == before
    assert state["draw_offer"] == "white"
    assert failure.value.accounting_state["llm_tokens"]["total_tokens"] == 11
    assert failure.value.accounting_state["llm_accounting_status"] == "partial"


def test_draw_routes_require_authentication(client):
    assert client.post("/api/play/move", json={"action": "accept_draw"}).status_code == 403


@pytest.mark.parametrize("mode", ["offer", "accept", "decline", "history", "promotion"])
def test_draw_controls_send_guarded_actions(mode):
    completed = subprocess.run(["node", "tests/js/play_draw_offers_test.js", mode], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.mark.parametrize("action", [[], {}, None, "offer_draw"])
def test_malformed_action_is_rejected_without_changing_saved_offer(client, action):
    csrf = _set_user(client)
    state = service.start_game("route-model", "black", web_app.CONFIG_PATH, ENV, lambda *_: "e2e4 offer_draw",
        human_profile={"username": "tester", "rating": 1800, "rating_deviation": 60})
    _save_test_state(client, state)
    response = client.post("/api/play/move", json=dict(_move_payload(client, None), action=action), headers={"X-CSRF-Token": csrf})
    assert response.status_code == 400
    assert client.play_store.load("johnstondaniel4@gmail.com") == state


def test_draw_offer_controls_appear_in_the_position_status_card(client):
    _set_user(client)
    html = client.get("/play").get_data(as_text=True)
    position_card = html.split('<div class="turn-card">', 1)[1].split('<div class="move-ledger">', 1)[0]
    assert 'id="draw-offer-status"' in position_card
    assert 'id="accept-draw"' in position_card
    assert 'id="decline-draw"' in position_card
    assert 'id="offer-draw"' in position_card
