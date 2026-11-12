import rating.rating_store as rating_store_module
import web.app as web_app
from rating.rating_store import RatingStore, load_position_benchmark_inputs
from web.position_results import build_position_rows, summarize_errors


def _row(**overrides):
    row = {
        "player_id": "model-a",
        "is_engine": False,
        "is_current": True,
        "status_reason": "ready",
        "positions": 50,
        "equal_cpl": 120,
        "best_pct": 40.0,
        "legal_pct": 100.0,
        "game_like_cpl": 200,
        "stability_risk": 0.0,
        "equal_prediction": 1500,
        "prediction": 1400,
        "capped": True,
        "game_rating": 1450,
        "games_played": 40,
        "rating_reliable": True,
        "error": 50,
    }
    row.update(overrides)
    return row


def test_positions_page_renders_rows(monkeypatch):
    rows = [_row(), _row(player_id="old-model", is_current=False, prediction=None, error=None)]
    monkeypatch.setattr(web_app, "get_position_rows", lambda: rows)
    web_app.app.config.update(TESTING=True)

    with web_app.app.test_client() as client:
        response = client.get("/positions")

    body = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "model-a" in body and "old-model" in body
    assert "Show 1 legacy results" in body
    assert "+50" in body
    assert "capped" in body


def test_positions_page_reports_unavailable_data(monkeypatch):
    monkeypatch.setattr(web_app, "get_position_rows", lambda: None)
    web_app.app.config.update(TESTING=True)

    with web_app.app.test_client() as client:
        page = client.get("/positions")
        api = client.get("/api/positions")

    assert page.status_code == 200
    assert "temporarily unavailable" in page.get_data(as_text=True)
    assert api.status_code == 503


def test_summarize_errors_ignores_rows_without_established_ratings():
    stats = summarize_errors([_row(error=100), _row(error=-200), _row(error=None)])
    assert stats == {"count": 2, "mae": 150, "rmse": 158, "bias": -50}


def test_page_predictions_match_rating_seeds(tmp_path, monkeypatch):
    """The page must show the same predictions that seed new players' ratings."""
    monkeypatch.setattr(rating_store_module, "_benchmark_predictions_cache", None)
    store = RatingStore(str(tmp_path / "ratings.json"), use_firestore=False)
    seeds = store._load_benchmark_predictions(force_refresh=True)
    assert seeds, "expected bundled position benchmark results"

    rows = build_position_rows(load_position_benchmark_inputs(use_firestore=False), [])
    shown = {r["player_id"]: r["prediction"] for r in rows if r["prediction"] is not None}

    assert shown.keys() == seeds.keys()
    for model, seed in seeds.items():
        assert shown[model] == round(seed)


def test_error_only_uses_established_game_ratings():
    inputs = load_position_benchmark_inputs(use_firestore=False)
    rows = build_position_rows(inputs, [])
    model = next(r["player_id"] for r in rows if r["prediction"] is not None)
    prediction = next(r["prediction"] for r in rows if r["player_id"] == model)

    leaderboard = [{"player_id": model, "rating": prediction + 120, "rating_deviation": 90,
                    "games_played": 60, "is_anchor": False}]
    row = next(r for r in build_position_rows(inputs, leaderboard) if r["player_id"] == model)
    assert row["error"] in (119, 120, 121)

    leaderboard[0]["rating_deviation"] = 200
    row = next(r for r in build_position_rows(inputs, leaderboard) if r["player_id"] == model)
    assert row["error"] is None
    assert row["game_rating"] == prediction + 120


def test_live_optional_panel_is_shared_by_page_and_rating_seeds(monkeypatch):
    from dataclasses import replace
    from position_benchmark.predictions import benchmark_result_readiness
    import web.position_results as page

    inputs = load_position_benchmark_inputs(use_firestore=False)
    model, record = next((pid, r) for pid, r in inputs.core_results.items()
                         if benchmark_result_readiness(r, inputs.core_positions).is_ready)
    core = {**record, "supplements": {"blunder": {"source": "live"}}}
    inputs = replace(inputs, core_results={model: core},
                     blunder_results={model: {"source": "bundled"}})

    def prediction(data, positions, **kwargs):
        return 1200 if kwargs["blunder_model_data"] == {"source": "live"} else 1800

    monkeypatch.setattr(page, "predict_rating_from_model_data_with_supplement", prediction)
    monkeypatch.setattr(rating_store_module, "predict_rating_from_model_data_with_supplement", prediction)
    seed = rating_store_module.benchmark_predictions_from_inputs(inputs)[model]
    assert seed == 1200
    assert build_position_rows(inputs, [])[0]["prediction"] == seed


def test_deferred_supplements_show_source_and_keep_targets_own_prediction(monkeypatch):
    from dataclasses import replace

    source, target = "gpt-6-astra (medium)", "gpt-6-astra (high)"
    inputs = load_position_benchmark_inputs(use_firestore=False)
    inputs = replace(
        inputs,
        core_results={player: inputs.core_results[player] for player in (source, target)},
        game_like_results={source: inputs.game_like_results[source]},
        stability_results={source: inputs.stability_results[source]},
    )
    rows = {row["player_id"]: row for row in build_position_rows(inputs, [])}
    assert rows[target].get("deferred_supplement_source") == source
    assert rows[source]["deferred_supplement_source"] is None
    assert rows[target]["game_like_cpl"] is None
    assert rows[target]["stability_risk"] is None
    assert rows[target]["prediction"] == round(
        rating_store_module.benchmark_predictions_from_inputs(inputs)[target]
    )
    assert rows[source]["game_like_cpl"] is not None
    assert rows[source]["stability_risk"] is not None

    monkeypatch.setattr(web_app, "get_position_rows", lambda: [rows[target]])
    with web_app.app.test_client() as client:
        html = client.get("/positions").get_data(as_text=True)
    assert html.count(">Deferred</span>") == 2
    assert source in html
