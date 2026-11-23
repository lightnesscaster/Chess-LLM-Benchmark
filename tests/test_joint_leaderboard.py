from dataclasses import replace

import pytest

from game.models import GameResult
from rating.glicko2 import PlayerRating
from rating.joint_fit import fit_leaderboard
from rating.rating_store import RatingStore


def game(white, black, winner):
    return GameResult(game_id=f"{white}-{black}", white_id=white, black_id=black, winner=winner,
                      termination="checkmate", moves=30, total_moves_white=15, total_moves_black=15,
                      illegal_moves_white=0, illegal_moves_black=0, pgn_path="", created_at="2026-01-01T00:00:00Z")


def test_fit_uses_each_score_once_and_respects_fixed_and_one_way_opponents():
    engines = [{"player_id": "anchor", "rating": 1200}, {"player_id": "ghost", "ghost": True, "anchor": False}]
    games = [game("a", "anchor", "white"), game("anchor", "a", "black"), game("b", "a", "draw")]
    human = game("lichess:tester", "b", "black").model_copy(update={
        "game_type":"human_challenge", "human_lichess_username":"tester",
        "human_rating":1500, "human_rating_deviation":50})
    games.append(human)
    base = PlayerRating("a", rating=2500, games_played=3, is_frozen=True)
    def fitted(results):
        store = RatingStore(snapshot=({"a":replace(base)}, {"a":1300, "b":1300}), anchor_ids={"anchor"})
        fit_leaderboard(store, results, engines)
        return store.get_all()
    ordinary = fitted(games)
    updated = fitted([game("ghost","a","white")] + list(reversed(games)))
    assert "lichess:tester" not in updated
    assert updated["anchor"].rating == 1200
    assert updated["a"].is_frozen
    assert (updated["a"].wins, updated["a"].losses, updated["a"].draws) == (2, 1, 1)
    assert updated["a"].games_played == 4
    for n in ("a","b"):
        assert updated[n].rating == pytest.approx(ordinary[n].rating, abs=.001)
        assert updated[n].rating_deviation == pytest.approx(ordinary[n].rating_deviation, abs=.001)
        assert updated[n].rating_deviation < 300
    assert updated["ghost"].wins == 1
