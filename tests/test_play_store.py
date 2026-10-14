import copy

import chess
import pytest


@pytest.fixture
def state():
    return {'game_id': '11111111-1111-4111-8111-111111111111', 'model_id': 'test',
            'human_color': 'white', 'human_profile': {'username': 'Player'},
            'status': 'active', 'moves': ['e2e4', 'e7e5']}


def expected(state):
    board = chess.Board()
    for uci in state['moves']:
        board.push_uci(uci)
    return {'game_id': state['game_id'], 'expected_fen': board.fen(), 'expected_ply': len(state['moves'])}


def test_legacy_cookie_migrates_once_and_cannot_overwrite_server_turn(play_store, state):
    assert play_store.load('owner@example.com', state) == state
    claim = play_store.begin('owner@example.com', expected(state))
    updated = {**state, 'moves': state['moves']+['g1f3', 'b8c6']}
    play_store.finish(claim, updated)
    assert play_store.load('owner@example.com', state) == updated
    assert play_store.load('another@example.com') is None


def test_conflicting_tabs_and_duplicate_moves_do_not_get_another_turn(play_store, state):
    from web.play_store import PlayConflict
    play_store.load('owner@example.com', state)
    claim = play_store.begin('owner@example.com', expected(state))
    with pytest.raises(PlayConflict):
        play_store.begin('owner@example.com', expected(state))
    with pytest.raises(PlayConflict):
        play_store.begin('owner@example.com', {}, starting=True)
    updated = {**state, 'moves': state['moves']+['g1f3','b8c6']}
    play_store.finish(claim, updated)
    with pytest.raises(PlayConflict):
        play_store.begin('owner@example.com', expected(state))
    assert play_store.load('owner@example.com') == updated


@pytest.mark.parametrize('change', [
    {'game_id': 'different-game'}, {'expected_fen': chess.STARTING_FEN},
    {'expected_ply': 0}, {'expected_ply': False},
])
def test_turn_is_bound_to_exact_game_and_position(play_store, state, change):
    from web.play_store import PlayConflict
    play_store.load('owner@example.com', state)
    with pytest.raises(PlayConflict):
        play_store.begin('owner@example.com', {**expected(state), **change})


def test_provider_failure_keeps_paid_usage_but_releases_turn(play_store, state):
    play_store.load('owner@example.com', state)
    claim = play_store.begin('owner@example.com', expected(state))
    partial = {**state, 'llm_tokens': {'total_tokens': 150}, 'llm_accounting_status': 'partial'}
    play_store.finish(claim, partial)
    assert play_store.load('owner@example.com') == partial
    retry = play_store.begin('owner@example.com', expected(state))
    play_store.finish(retry, partial)


def test_late_expired_request_cannot_overwrite_newer_claim(play_store, state, monkeypatch):
    from web import play_store as service
    monkeypatch.setattr(service.time, 'time', lambda: 1000)
    play_store.load('owner@example.com', state)
    old = play_store.begin('owner@example.com', expected(state))
    monkeypatch.setattr(service.time, 'time', lambda: 1000+service.TURN_LEASE_SECONDS+1)
    new = play_store.begin('owner@example.com', expected(state))
    with pytest.raises(service.PlayConflict):
        play_store.finish(old, {**state, 'moves': state['moves']+['g1f3','b8c6']})
    play_store.finish(new, state)


def test_finished_unpublished_game_cannot_be_replaced(play_store, state):
    from web.play_store import PlayConflict
    finished = {**state, 'status': 'finished', 'winner': 'human'}
    play_store.load('owner@example.com', finished)
    with pytest.raises(PlayConflict):
        play_store.begin('owner@example.com', {}, starting=True)
    play_store.annotate_rating('owner@example.com', state['game_id'], {'recorded': True})
    claim = play_store.begin('owner@example.com', {}, starting=True)
    play_store.finish(claim, claim.state)


def test_completed_turn_survives_database_failure_and_reloads_without_replaying(play_store, state):
    play_store.load('owner@example.com', state)
    claim = play_store.begin('owner@example.com', expected(state))
    completed = {**state, 'moves': state['moves']+['g1f3', 'b8c6']}
    play_store._db.fail_next_write = True
    with pytest.raises(OSError):
        play_store.finish(claim, completed)
    assert play_store.load('owner@example.com') == completed
    from web.play_store import PlayConflict
    with pytest.raises(PlayConflict):
        play_store.begin('owner@example.com', expected(state))


def test_failed_commit_cannot_be_overwritten_by_rollback_cleanup(play_store, state):
    from web.play_store import PlayConflict
    play_store.load('owner@example.com', state)
    claim = play_store.begin('owner@example.com', expected(state))
    completed = {**state, 'moves': state['moves']+['g1f3','b8c6']}
    play_store._db.fail_next_write = True
    with pytest.raises(OSError):
        play_store.finish(claim, completed)
    with pytest.raises(PlayConflict):
        play_store.finish(claim, state)
    assert play_store.load('owner@example.com') == completed


def test_claim_renewal_protects_long_turn_beyond_initial_lease(play_store, state, monkeypatch):
    from web import play_store as service
    clock = [1000]
    monkeypatch.setattr(service.time, 'time', lambda: clock[0])
    play_store.load('owner@example.com', state)
    claim = play_store.begin('owner@example.com', expected(state))
    clock[0] += service.TURN_LEASE_SECONDS-1
    play_store.renew(claim)
    clock[0] += 2
    with pytest.raises(service.PlayConflict):
        play_store.begin('owner@example.com', expected(state))
    play_store.finish(claim, state)
