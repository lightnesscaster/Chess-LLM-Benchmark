from copy import deepcopy

import pytest

from rating.glicko2 import PlayerRating
from web.human_challenges import (
    HumanChallengeError,
    build_human_challenge_result,
    record_human_challenge,
    score_model_against_human,
)


def _finished_state(*, human_rd=70, winner="llm"):
    return {
        "game_id": "human-game-1",
        "started_at": "2026-09-02T12:00:00+00:00",
        "model_id": "reasoner",
        "rated_model_id": "reasoner (high)",
        "model_name": "provider/reasoner",
        "reasoning_effort": "high",
        "human_color": "white",
        "human_profile": {
            "username": "Some_Player",
            "rating": 1847,
            "rating_deviation": human_rd,
            "provisional": human_rd >= 110,
        },
        "moves": ["f2f3", "e7e5", "g2g4", "d8h4"],
        "status": "finished",
        "winner": winner,
        "termination": "checkmate",
        "llm_illegal_moves": 0,
    }


def test_completed_game_builds_replayable_human_challenge_result():
    result = build_human_challenge_result(
        _finished_state(),
        "Player@Example.com",
        completed_at="2026-09-02T12:05:00+00:00",
    )

    assert result.game_type == "human_challenge"
    assert result.white_id == "lichess:Some_Player"
    assert result.black_id == "reasoner (high)"
    assert result.winner == "black"
    assert result.human_email == "player@example.com"
    assert result.human_lichess_username == "Some_Player"
    assert result.human_rating == 1847
    assert result.human_rating_deviation == 70
    assert result.human_rating_provisional is False
    assert result.moves == 4


def test_unfinished_game_cannot_be_scored():
    state = _finished_state()
    state.update(status="active", winner=None, termination=None)

    with pytest.raises(HumanChallengeError, match="finished"):
        build_human_challenge_result(state, "player@example.com")


def test_finished_game_without_valid_snapshot_cannot_be_scored():
    state = _finished_state()
    state["human_profile"]["rating_deviation"] = None

    with pytest.raises(HumanChallengeError, match="snapshot"):
        build_human_challenge_result(state, "player@example.com")


def test_human_rd_controls_how_much_the_model_rating_moves():
    model = PlayerRating(
        player_id="reasoner (high)",
        rating=1500,
        rating_deviation=180,
        games_rd=180,
    )
    certain_human = build_human_challenge_result(_finished_state(human_rd=45), "player@example.com")
    uncertain_human = build_human_challenge_result(_finished_state(human_rd=300), "player@example.com")

    certain_update = score_model_against_human(deepcopy(model), certain_human)
    uncertain_update = score_model_against_human(deepcopy(model), uncertain_human)

    assert certain_update.rating > model.rating
    assert uncertain_update.rating > model.rating
    assert certain_update.rating - model.rating > uncertain_update.rating - model.rating


class FakeRatingStore:
    def __init__(self):
        self.rating = PlayerRating(
            player_id="reasoner (high)",
            rating=1500,
            rating_deviation=180,
            games_rd=180,
        )
        self.set_calls = 0

    def has_player(self, player_id):
        return player_id == self.rating.player_id

    def get(self, player_id):
        assert player_id == "reasoner (high)"
        return deepcopy(self.rating)

    def set(self, rating):
        self.rating = deepcopy(rating)
        self.set_calls += 1


class FakePGNLogger:
    def __init__(self):
        self.results = {}
        self.pgns = {}

    def load_result(self, game_id):
        return self.results.get(game_id)

    def save_game(self, result, pgn):
        self.results[result.game_id] = result
        self.pgns[result.game_id] = pgn
        return result


def test_recording_same_finished_game_twice_updates_rating_once():
    rating_store = FakeRatingStore()
    logger = FakePGNLogger()

    first = record_human_challenge(
        _finished_state(),
        "player@example.com",
        rating_store=rating_store,
        pgn_logger=logger,
    )
    second = record_human_challenge(
        _finished_state(),
        "player@example.com",
        rating_store=rating_store,
        pgn_logger=logger,
    )

    assert first["recorded"] is True
    assert second["recorded"] is False
    assert rating_store.set_calls == 1
    assert "[White \"lichess:Some_Player\"]" in logger.pgns["human-game-1"]
    assert "[Black \"reasoner (high)\"]" in logger.pgns["human-game-1"]
