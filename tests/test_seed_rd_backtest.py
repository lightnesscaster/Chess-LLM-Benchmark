from copy import deepcopy

import pytest

from rating.glicko2 import Glicko2System
from rating.seed_rd_backtest import replay, summarize


def game(number, white="model", black="anchor", winner="white", **extra):
    return {"game_id": str(number), "created_at": f"2026-01-{number:02d}T00:00:00Z",
            "white_id": white, "black_id": black, "winner": winner, **extra}


SEEDS = {"model": {"rating": 1500, "available_at": "2025-12-01T00:00:00Z",
                   "family": "family", "lab": "lab"}}
ENGINES = [{"player_id": "anchor", "rating": 1500}]


def run(games, rd=166, seeds=None, engines=None, model_ids=None):
    return replay(games, SEEDS if seeds is None else seeds,
                  ENGINES if engines is None else engines, model_ids or {}, rd)


def test_scores_before_updates_and_future_results_do_not_change_prefix():
    short = run([game(1)])
    long = run([game(2), game(1)])
    assert short["observations"] == long["observations"][:1]
    assert short["observations"][0]["expected_score"] == 0.5
    assert long["observations"][1]["expected_score"] > 0.5
    assert short["ratings"]["model"]["games_played"] == 1
    assert run([game(1), game(2)], rd=350)["ratings"]["model"]["rating"] > long["ratings"]["model"]["rating"]


def test_both_players_update_from_same_pre_game_state():
    seeds = {**SEEDS, "other": {**SEEDS["model"]}}
    result = run([game(1, black="other")], seeds=seeds)
    ratings = result["ratings"]
    assert ratings["model"]["rating"] - 1500 == pytest.approx(1500 - ratings["other"]["rating"])
    assert [o["expected_score"] for o in result["observations"]] == [0.5, 0.5]


def test_seeds_must_precede_first_game():
    seeds = {"model": {**SEEDS["model"], "available_at": "2026-01-02T00:00:00Z"}}
    with pytest.raises(ValueError, match="predates"):
        run([game(1)], seeds=seeds)


def test_anchor_history_uses_game_timestamp():
    engines = [{"player_id": "anchor", "rating": 1800,
                "rating_history": [{"before": "2026-01-02T00:00:00Z", "rating": 1500}]}]
    result = run([game(1), game(2)], engines=engines)
    assert [o["opponent_rating"] for o in result["observations"]] == [1500, 1800]
    assert result["ratings"]["anchor"]["rating"] == 1800


def test_fixed_human_is_never_added_to_pool_and_bad_snapshot_is_skipped():
    good = game(1, black="lichess:example", game_type="human_challenge",
                human_lichess_username="example", human_rating=2329, human_rating_deviation=148)
    bad = {**game(2), "game_type": "human_challenge"}
    result = run([good, bad])
    assert "lichess:example" not in result["ratings"]
    assert result["observations"][0]["opponent_rating"] == 2329
    assert len(result["rejected"]) == 1


def test_ghost_does_not_update_opponent_and_is_excluded_from_scoring():
    ghost = [{"player_id": "anchor", "rating": 1500, "anchor": False, "ghost": True}]
    result = run([game(1)], engines=ghost)
    assert result["observations"] == []
    assert result["ratings"]["model"]["games_played"] == 0
    assert result["ratings"]["anchor"]["games_played"] == 1


def test_duplicate_invalid_and_draw_outcomes_are_explicit():
    rows = [game(1, winner="draw"), game(1), game(2, winner="error"), game(3, black="model")]
    before = deepcopy(rows)
    result = run(rows)
    assert rows == before
    assert len(result["observations"]) == 1
    assert result["observations"][0]["score"] == 0.5
    assert len(result["rejected"]) == 3


def test_forecast_is_production_expected_score_with_opponent_rd():
    result = run([game(1)], seeds={"model": {**SEEDS["model"], "rating": 1700}})
    glicko = Glicko2System()
    mu, _ = glicko.glicko_to_glicko2(1700, 166)
    other_mu, phi = glicko.glicko_to_glicko2(1500, 30)
    assert result["observations"][0]["expected_score"] == pytest.approx(glicko.e_function(mu, other_mu, phi))


def test_summary_balances_families_not_number_of_games():
    rows = [{"family": "a", "player_id": "a", "game_id": str(i), "expected_score": 0.5,
             "score": 1, "prior_games": i} for i in range(3)]
    rows.append({"family": "b", "player_id": "b", "game_id": "x", "expected_score": 1,
                 "score": 1, "prior_games": 0})
    metrics = summarize(rows)
    assert metrics["observations"] == 4
    assert metrics["brier"] == pytest.approx(0.1875)
    assert metrics["family_macro_brier"] == pytest.approx(0.125)


def test_sibling_seed_sees_only_previously_played_lower_effort():
    low, high = "model (low)", "model (high)"
    seeds = {high: {**SEEDS["model"], "rating": 1200},
             low: {**SEEDS["model"], "rating": 1700}}
    ids = {low: "provider/model", high: "provider/model"}
    high_first = run([game(1, white=high), game(2, white=low)], seeds=seeds, model_ids=ids)
    low_first = run([game(1, white=low), game(2, white=high)], seeds=seeds, model_ids=ids)
    assert high_first["observations"][0]["rating"] == 1200
    assert low_first["observations"][1]["rating"] > 1700
