"""The calibrated prior widens uncertainty without changing seed means."""

import json
import sys
import asyncio
from types import SimpleNamespace

import pytest

import cli
from rating.rating_store import RatingStore
from rating.glicko2 import PlayerRating
from game.models import GameResult


LOW = "calibration-model (low)"
HIGH = "calibration-model (high)"


@pytest.fixture
def sibling_metadata(monkeypatch):
    from rating import rating_store

    monkeypatch.setattr(rating_store, "load_publish_dates", lambda: {
        player: {"model_id": "test/calibration-model"} for player in (LOW, HIGH)
    })


@pytest.mark.parametrize("seed_rd", [None, 166.0, 300.0, 350.0])
@pytest.mark.parametrize("benchmark_backed", [False, True])
@pytest.mark.parametrize("reset", [False, True])
def test_sibling_prior_only_uses_calibrated_rd_with_benchmark_evidence(
    tmp_path, monkeypatch, sibling_metadata, seed_rd, benchmark_backed, reset,
):
    predictions = {HIGH: 1200.0} if benchmark_backed else {}
    monkeypatch.setattr(RatingStore, "_load_benchmark_predictions", lambda *_: predictions)
    options = {} if seed_rd is None else {"benchmark_seed_rd": seed_rd}
    store = RatingStore(path=str(tmp_path / "ratings.json"), use_firestore=False, **options)
    store.set(PlayerRating(LOW, rating=1900.0), auto_save=False)
    if reset:
        store.set(PlayerRating(HIGH, rating=1450.0), auto_save=False)
        store.snapshot_for_pass(set(), {HIGH: PlayerRating(HIGH, rating=1200.0)})

    rating = store.get(HIGH)

    assert rating.rating == 1900.0
    expected_rd = seed_rd if seed_rd is not None else (300.0 if benchmark_backed else 166.0)
    assert rating.rating_deviation == expected_rd
    assert rating.games_rd == 350.0


@pytest.mark.parametrize("seed_rd,disabled", [(None, False), (166.0, False), (300.0, False), (350.0, False), (350.0, True)])
@pytest.mark.parametrize("benchmark_backed", [False, True])
def test_real_replay_preserves_sibling_only_rd_and_validation_overrides(
    tmp_path, monkeypatch, sibling_metadata, seed_rd, disabled, benchmark_backed,
):
    from rating import benchmark_recalculation as updates

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config = config_dir / "benchmark.yaml"
    config.write_text(json.dumps({"engines": [{
        "player_id": "ghost", "rating": 1800, "ghost": True,
    }], "llms": []}))
    monkeypatch.setattr(updates, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "is_reasoning_model", lambda player: player == LOW)
    games = [GameResult(
        game_id=f"game-{index}", white_id=player, black_id="ghost", winner="draw",
        termination="draw_agreement", moves=2, illegal_moves_white=0, illegal_moves_black=0,
        total_moves_white=1, total_moves_black=1, pgn_path="saved.pgn",
        created_at=f"2026-11-10T12:0{index}:00+00:00",
    ) for index, player in enumerate((LOW, HIGH))]
    predictions = {LOW: 1900.0, **({HIGH: 1500.0} if benchmark_backed else {})}

    if seed_rd is None:
        ratings = updates._calculate(games, predictions)
    else:
        output = tmp_path / "ratings.json"
        args = SimpleNamespace(config=str(config), verbose=False, validation_output=output,
            validation_seed_rd=seed_rd, validation_disable_benchmark_seeds=disabled,
            suppress_result_table=True)
        assert asyncio.run(cli.recalculate_ratings(
            args, results_snapshot=games, benchmark_predictions=predictions,
        )) == 0
        ratings = json.loads(output.read_text())

    expected_rd = seed_rd if seed_rd is not None else (300.0 if benchmark_backed else 166.0)
    assert ratings[HIGH]["rating"] == (1200.0 if disabled else 1900.0)
    assert ratings[HIGH]["rating_deviation"] == expected_rd
    assert ratings[HIGH]["games_rd"] == 350.0
    assert ratings[HIGH]["games_played"] == 1


@pytest.mark.parametrize("seed_rd", [None, 166.0, 350.0])
@pytest.mark.parametrize("existing", [False, True])
def test_benchmark_seed_preserves_mean_and_games_uncertainty(tmp_path, monkeypatch, seed_rd, existing):
    path = tmp_path / "ratings.json"
    if existing:
        path.write_text(json.dumps({"model": {
            "player_id": "model", "rating": 1500, "rating_deviation": 350,
        }}))
    monkeypatch.setattr(RatingStore, "_load_benchmark_predictions", lambda *_: {"model": 1234.0})
    options = {} if seed_rd is None else {"benchmark_seed_rd": seed_rd}
    store = RatingStore(path=str(path), use_firestore=False, **options)

    rating = store.get("model")

    assert rating.rating == 1234.0
    assert rating.rating_deviation == (300.0 if seed_rd is None else seed_rd)
    assert rating.games_rd == 350.0
    assert store.get("unseeded-model").rating_deviation == 350.0


def test_validation_cli_help_reports_current_production_seed_rd(monkeypatch, capsys):
    monkeypatch.setattr(cli, "BENCHMARK_SEED_RD", 275.0)
    monkeypatch.setattr(sys, "argv", ["cli.py", "recalculate", "--help"])
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 0
    assert "production remains 275" in " ".join(capsys.readouterr().out.split())
