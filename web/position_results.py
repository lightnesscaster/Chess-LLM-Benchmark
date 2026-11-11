"""Position-benchmark results for the public web page.

Builds one row per benchmarked player from the same merged inputs and
predictor that seed leaderboard ratings, then joins each row with the
player's current game rating so the prediction error is visible.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable, Iterable

from position_benchmark.layout import (
    BLUNDER_RESULTS_PATH,
    CORE_RESULTS_PATH,
    GAME_LIKE_RESULTS_PATH,
    STABILITY_RESULTS_PATH,
)
from position_benchmark.predictions import (
    DEFAULT_GAME_LIKE_CPL_CAP,
    benchmark_result_readiness,
    collect_equal_position_metrics,
    collect_stability_probe_metrics,
    predict_rating,
    predict_rating_from_model_data_with_supplement,
)
from rating.rating_store import (
    PositionBenchmarkInputs,
    load_position_benchmark_inputs,
    should_use_firestore,
)

logger = logging.getLogger(__name__)

# Game ratings with a deviation above this are still preliminary (the
# leaderboard greys them out), so they are excluded from the error stats.
RELIABLE_RD = 150

_CACHE_TTL = 3600
_cache_lock = threading.Lock()
_cached_inputs: PositionBenchmarkInputs | None = None
_cached_time: float = 0.0

_RESULT_PATHS = (CORE_RESULTS_PATH, GAME_LIKE_RESULTS_PATH, STABILITY_RESULTS_PATH, BLUNDER_RESULTS_PATH)


def _results_changed_since(timestamp: float) -> bool:
    try:
        return any(p.exists() and p.stat().st_mtime > timestamp for p in _RESULT_PATHS)
    except OSError:
        return False


def get_position_benchmark_inputs(
    should_invalidate: Callable[[float], bool] | None = None,
) -> PositionBenchmarkInputs | None:
    """Load merged benchmark inputs, cached for an hour or until data changes."""
    global _cached_inputs, _cached_time
    with _cache_lock:
        fresh = (
            _cached_inputs is not None
            and time.time() - _cached_time < _CACHE_TTL
            and not _results_changed_since(_cached_time)
            and not (should_invalidate and should_invalidate(_cached_time))
        )
        if fresh:
            return _cached_inputs
        try:
            inputs = load_position_benchmark_inputs(use_firestore=should_use_firestore())
        except Exception:
            logger.exception("Could not load position benchmark results")
            return _cached_inputs
        if inputs is not None:
            _cached_inputs = inputs
            _cached_time = time.time()
        return inputs if inputs is not None else _cached_inputs


def _supplement(
    model_data: dict[str, Any], key: str, fallback: dict[str, Any] | None, model: str
) -> dict[str, Any] | None:
    return model_data.get("supplements", {}).get(key) or (
        fallback.get(model) if isinstance(fallback, dict) else None
    )


def _round(value: float | None, digits: int = 0) -> float | int | None:
    if value is None:
        return None
    return round(value) if digits == 0 else round(value, digits)


def build_position_rows(
    inputs: PositionBenchmarkInputs,
    leaderboard: Iterable[dict[str, Any]],
    engine_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """One display row per benchmarked player, sorted by prediction."""
    engine_ids = engine_ids or set()
    ratings = {entry["player_id"]: entry for entry in leaderboard}
    positions = inputs.core_positions
    rows = []

    for model, model_data in inputs.core_results.items():
        metrics = collect_equal_position_metrics(model_data.get("results", []), positions)
        if metrics is None:
            continue
        readiness = benchmark_result_readiness(model_data, positions)
        game_like_data = _supplement(model_data, "game_like", inputs.game_like_results, model)
        stability_data = _supplement(model_data, "stability", inputs.stability_results, model)

        equal_prediction = predict_rating(metrics.equal_cpl, metrics.best_pct, metrics.legal_pct)
        prediction = None
        if readiness.is_ready:
            # Same arguments as RatingStore._load_benchmark_predictions.
            prediction = predict_rating_from_model_data_with_supplement(
                model_data,
                positions,
                blunder_model_data=_supplement(model_data, "blunder", inputs.blunder_results, model),
                blunder_positions=inputs.blunder_positions,
                game_like_model_data=game_like_data,
                game_like_positions=inputs.game_like_positions,
                stability_probe_model_data=stability_data,
                require_ready=True,
            )

        game_like_cpl = None
        if game_like_data and inputs.game_like_positions:
            game_like_metrics = collect_equal_position_metrics(
                game_like_data.get("results", []),
                inputs.game_like_positions,
                cpl_cap=DEFAULT_GAME_LIKE_CPL_CAP,
            )
            if game_like_metrics is not None:
                game_like_cpl = game_like_metrics.equal_cpl

        stability_risk = None
        if stability_data:
            stability_metrics = collect_stability_probe_metrics(stability_data)
            if stability_metrics is not None:
                stability_risk = (
                    stability_metrics.forfeit_pct + stability_metrics.deduplicated_catastrophe_pct
                )

        entry = ratings.get(model)
        is_engine = model in engine_ids or bool(entry and entry.get("is_anchor"))
        game_rating = entry.get("rating") if entry else None
        rating_reliable = bool(
            entry
            and game_rating is not None
            and (entry.get("is_anchor") or entry.get("rating_deviation", 999) <= RELIABLE_RD)
        )
        error = (
            game_rating - prediction
            if prediction is not None and rating_reliable
            else None
        )

        rows.append({
            "player_id": model,
            "is_engine": is_engine,
            "is_current": readiness.is_ready,
            "status_reason": readiness.reason,
            "positions": metrics.total,
            "equal_cpl": _round(metrics.equal_cpl),
            "best_pct": _round(metrics.best_pct, 1),
            "legal_pct": _round(metrics.legal_pct, 1),
            "game_like_cpl": _round(game_like_cpl),
            "stability_risk": _round(stability_risk, 1),
            "equal_prediction": _round(equal_prediction),
            "prediction": _round(prediction),
            "capped": prediction is not None and prediction < equal_prediction - 0.5,
            "game_rating": game_rating,
            "games_played": entry.get("games_played") if entry else None,
            "rating_reliable": rating_reliable,
            "error": _round(error),
        })

    rows.sort(
        key=lambda r: (
            r["prediction"] is None,
            -(r["prediction"] if r["prediction"] is not None else r["equal_prediction"]),
        )
    )
    return rows


def summarize_errors(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """MAE / RMSE / bias of game rating minus prediction for reliable rows."""
    errors = [r["error"] for r in rows if r["error"] is not None]
    if not errors:
        return {"count": 0, "mae": None, "rmse": None, "bias": None}
    n = len(errors)
    return {
        "count": n,
        "mae": round(sum(abs(e) for e in errors) / n),
        "rmse": round(math.sqrt(sum(e * e for e in errors) / n)),
        "bias": round(sum(errors) / n),
    }
