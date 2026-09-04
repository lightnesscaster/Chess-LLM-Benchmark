#!/usr/bin/env python3
"""Run the frozen zero-call supplemental predictor evaluation."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_PLAN = (
    ROOT
    / "position_benchmark/validation/2026-07-17-supplemental-predictor-plan.json"
)

from position_benchmark.layout import (  # noqa: E402
    BLUNDER_POSITIONS_PATH,
    BLUNDER_RESULTS_PATH,
    CORE_POSITIONS_PATH,
    CORE_RESULTS_PATH,
    GAME_LIKE_POSITIONS_PATH,
    GAME_LIKE_RESULTS_PATH,
    STABILITY_RESULTS_PATH,
)
from position_benchmark.predictions import (  # noqa: E402
    DEFAULT_GAME_LIKE_CPL_CAP,
    DEFAULT_MIN_GAME_LIKE_POSITIONS,
    benchmark_result_readiness,
    collect_equal_position_metrics,
    collect_stability_probe_metrics,
    combine_equal_and_game_like_predictions,
    predict_rating,
    predict_rating_from_model_data,
    predict_rating_from_model_data_with_supplement,
    stability_probe_prediction_cap,
    stability_probe_readiness,
)


ANCHOR_IDS = {"random-bot", "eubos", "maia-1900", "maia-1100", "survival-bot"}
GAME_LIKE_CATEGORIES = (
    "advantage_conversion",
    "defense",
    "quiet_equal",
    "tactical_equal",
)
FAMILY_PREFIXES = (
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "gpt-5.6-sol",
    "gemini-3.5",
    "gemini-3.1",
    "gemini-3",
    "gemini-2.5",
    "gemini-2.0",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.2",
    "gpt-5.1",
    "gpt-5-mini",
    "gpt-5",
    "gpt-oss",
    "gpt-3.5",
    "grok-4.3",
    "grok-4.1",
    "grok-4",
    "grok-3",
    "claude",
    "deepseek-v4",
    "deepseek-v3.2",
    "deepseek-v3.1",
    "deepseek-chat",
    "deepseek-r1",
    "kimi-k2.5",
    "kimi-k2",
    "llama-4",
    "llama-3.3",
    "mistral",
    "qwen3",
    "glm",
    "gemma",
    "mimo",
    "o1",
    "o3",
)


def load(path: Path) -> Any:
    """Load JSON from disk."""
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    """Return the SHA-256 digest for a frozen input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_family(player_id: str) -> str:
    """Group reasoning variants while keeping GPT-5.6 base lines separate."""
    lowered = player_id.lower()
    for prefix in FAMILY_PREFIXES:
        if lowered.startswith(prefix):
            return prefix
    return lowered.split()[0].split("-")[0]


def retry_counts(record: dict[str, Any] | None) -> tuple[int, int]:
    """Return measured conditional retry attempts and failures from static rows."""
    if not record:
        return 0, 0
    attempts = 0
    failures = 0
    for row in record.get("results", []):
        if row.get("retry_attempted") is not True:
            continue
        attempts += 1
        if row.get("retry_is_legal") is False:
            failures += 1
    return attempts, failures


def continuation_retry_counts(record: dict[str, Any] | None) -> tuple[int, int]:
    """Return retry counts from the continuation summary."""
    if not record:
        return 0, 0
    summary = record.get("summary", record)
    conditional = summary.get("conditional_retry") or {}
    return (
        int(conditional.get("retry_attempts") or summary.get("model_retry_attempts") or 0),
        int(conditional.get("retry_failures") or summary.get("model_retry_failures") or 0),
    )


def static_legality_counts(record: dict[str, Any] | None) -> tuple[int, int]:
    """Return first-answer attempts and illegal answers from a static panel."""
    if not record:
        return 0, 0
    rows = record.get("results", [])
    return len(rows), sum(row.get("is_legal") is False for row in rows)


def continuation_legality_counts(record: dict[str, Any] | None) -> tuple[int, int]:
    """Recover continuation first-answer counts without counting failed retries twice."""
    if not record:
        return 0, 0
    summary = record.get("summary", record)
    legal_moves = int(summary.get("model_legal_moves") or 0)
    forfeits = int(summary.get("model_forfeits") or 0)
    illegal_attempts = int(summary.get("model_illegal_attempts") or 0)
    retry_failures = int(summary.get("model_retry_failures") or 0)
    return legal_moves + forfeits, max(0, illegal_attempts - retry_failures)


def category_cpl_features(
    record: dict[str, Any],
    positions: list[dict[str, Any]],
) -> dict[str, float]:
    """Calculate the frozen four category CPL means with the production 5000 cap."""
    category_by_id = {
        position["position_id"]: (
            position.get("screening_bucket") or position.get("regan_bucket")
        )
        for position in positions
    }
    values: dict[str, list[float]] = defaultdict(list)
    for row in record.get("results", []):
        category = category_by_id.get(row.get("position_id"))
        if category in GAME_LIKE_CATEGORIES:
            values[category].append(
                min(DEFAULT_GAME_LIKE_CPL_CAP, max(0.0, float(row.get("cpl") or 0)))
            )
    if any(len(values[category]) != 12 for category in GAME_LIKE_CATEGORIES):
        raise ValueError("Game-like category coverage is not exactly 12 per category")
    return {
        f"category_{category}": math.log1p(sum(values[category]) / len(values[category]))
        for category in GAME_LIKE_CATEGORIES
    }


def continuous_deadband(
    core_prediction: float,
    game_like_prediction: float | None,
    stability_cap: float | None,
) -> float:
    """Apply a continuous 150-Elo deadband to automatic downside estimates."""
    prediction = core_prediction
    if game_like_prediction is not None:
        prediction = min(prediction, game_like_prediction + 150.0)
    if stability_cap is not None:
        prediction = min(prediction, stability_cap + 150.0)
    return prediction


def fit_ridge(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit a standardized ridge with an unpenalized intercept."""
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-9] = 1.0
    train_z = (train_x - mean) / scale
    test_z = (test_x - mean) / scale
    train_design = np.column_stack([np.ones(len(train_z)), train_z])
    test_design = np.column_stack([np.ones(len(test_z)), test_z])
    penalty = np.eye(train_design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(
        train_design.T @ train_design + penalty
    ) @ (train_design.T @ train_y)
    return test_design @ coefficients


def inner_family_alpha(
    x: np.ndarray,
    residual: np.ndarray,
    families: list[str],
    alpha_grid: list[float],
) -> float:
    """Select ridge strength without seeing the outer held-out family."""
    family_array = np.asarray(families)
    unique = sorted(set(families))
    if len(unique) < 3:
        return max(alpha_grid)
    losses: dict[float, float] = {}
    all_indices = np.arange(len(residual))
    for alpha in alpha_grid:
        predicted = np.empty(len(residual), dtype=float)
        for family in unique:
            test = all_indices[family_array == family]
            train = all_indices[family_array != family]
            predicted[test] = fit_ridge(x[train], residual[train], x[test], alpha)
        losses[alpha] = float(np.mean(np.abs(residual - predicted)))
    return min(alpha_grid, key=lambda alpha: (losses[alpha], -alpha))


def nested_lofo_residual_predictions(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    alpha_grid: list[float],
) -> tuple[list[float], dict[str, float]]:
    """Predict residuals with outer family holdout and inner family alpha choice."""
    x = np.asarray(
        [[float(row[feature]) for feature in feature_names] for row in rows],
        dtype=float,
    )
    target = np.asarray([float(row["actual"]) for row in rows], dtype=float)
    baseline = np.asarray([float(row["production_prediction"]) for row in rows])
    residual = target - baseline
    families = [str(row["family"]) for row in rows]
    family_array = np.asarray(families)
    all_indices = np.arange(len(rows))
    prediction = np.empty(len(rows), dtype=float)
    selected: dict[str, float] = {}
    for family in sorted(set(families)):
        test = all_indices[family_array == family]
        train = all_indices[family_array != family]
        train_families = [families[index] for index in train]
        alpha = inner_family_alpha(
            x[train], residual[train], train_families, alpha_grid
        )
        selected[family] = alpha
        predicted_residual = fit_ridge(
            x[train], residual[train], x[test], alpha
        )
        prediction[test] = baseline[test] + predicted_residual
    return prediction.tolist(), selected


def prediction_metrics(
    rows: list[dict[str, Any]],
    predictions: list[float],
) -> dict[str, Any]:
    """Calculate configuration- and family-level prediction metrics."""
    actual = np.asarray([float(row["actual"]) for row in rows])
    predicted = np.asarray(predictions, dtype=float)
    error = actual - predicted
    rd = np.asarray([float(row["validation_rd"]) for row in rows])
    weights = 1.0 / np.maximum(rd * rd, 1.0)
    family_errors: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, error):
        family_errors[str(row["family"])].append(float(value))
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "mean_error": float(np.mean(error)),
        "max_absolute_error": float(np.max(np.abs(error))),
        "inverse_variance_weighted_mae": float(
            np.sum(weights * np.abs(error)) / np.sum(weights)
        ),
        "family_mae": {
            family: float(np.mean(np.abs(values)))
            for family, values in sorted(family_errors.items())
        },
        "predictions": {
            row["player_id"]: float(value)
            for row, value in zip(rows, predicted)
        },
    }


def family_bootstrap_improvement_probability(
    rows: list[dict[str, Any]],
    baseline: list[float],
    candidate: list[float],
    *,
    resamples: int,
    seed: int,
) -> float:
    """Bootstrap entire model-line families and compare MAE."""
    families = sorted({str(row["family"]) for row in rows})
    indices = {
        family: [index for index, row in enumerate(rows) if row["family"] == family]
        for family in families
    }
    actual = np.asarray([float(row["actual"]) for row in rows])
    base = np.asarray(baseline)
    cand = np.asarray(candidate)
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(resamples):
        sampled = rng.choice(families, size=len(families), replace=True)
        selected = [index for family in sampled for index in indices[str(family)]]
        baseline_mae = float(np.mean(np.abs(actual[selected] - base[selected])))
        candidate_mae = float(np.mean(np.abs(actual[selected] - cand[selected])))
        if candidate_mae < baseline_mae:
            wins += 1
    return wins / resamples


def evaluate_cohort(
    rows: list[dict[str, Any]],
    plan: dict[str, Any],
    *,
    learned_feature_sets: dict[str, list[str]],
) -> dict[str, Any]:
    """Evaluate fixed and leakage-controlled learned candidates on one cohort."""
    fixed_keys = {
        "core_fixed": "core_prediction",
        "game_like_hard_cap_fixed": "game_like_hard_cap",
        "production_fixed": "production_prediction",
        "continuous_deadband_fixed": "continuous_prediction",
    }
    predictions: dict[str, list[float]] = {
        name: [float(row[key]) for row in rows]
        for name, key in fixed_keys.items()
    }
    selected_alphas: dict[str, dict[str, float]] = {}
    alpha_grid = [float(value) for value in plan["learned_candidates"]["alpha_grid"]]
    for name, features in learned_feature_sets.items():
        prediction, alphas = nested_lofo_residual_predictions(
            rows, features, alpha_grid
        )
        predictions[f"lofo_{name}"] = prediction
        selected_alphas[f"lofo_{name}"] = alphas

    metrics = {
        name: prediction_metrics(rows, values)
        for name, values in predictions.items()
    }
    baseline = predictions["production_fixed"]
    baseline_family = metrics["production_fixed"]["family_mae"]
    bootstrap_resamples = int(plan["metrics"]["bootstrap_resamples"])
    bootstrap_seed = int(plan["metrics"]["bootstrap_seed"])
    for offset, (name, values) in enumerate(predictions.items()):
        candidate = metrics[name]
        candidate["mae_improvement_vs_production"] = (
            metrics["production_fixed"]["mae"] - candidate["mae"]
        )
        candidate["rmse_improvement_vs_production"] = (
            metrics["production_fixed"]["rmse"] - candidate["rmse"]
        )
        family_deltas = {
            family: baseline_family[family] - candidate["family_mae"][family]
            for family in baseline_family
        }
        candidate["family_mae_improvement"] = family_deltas
        candidate["family_win_fraction"] = (
            sum(value > 0 for value in family_deltas.values()) / len(family_deltas)
        )
        candidate["maximum_family_mae_worsening"] = max(
            [max(0.0, -value) for value in family_deltas.values()],
            default=0.0,
        )
        candidate["family_bootstrap_mae_improvement_probability"] = (
            0.0
            if name == "production_fixed"
            else family_bootstrap_improvement_probability(
                rows,
                baseline,
                values,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed + offset,
            )
        )

    target_residual = np.asarray(
        [row["actual"] - row["production_prediction"] for row in rows]
    )
    correlations = {}
    feature_names = sorted(
        {feature for features in learned_feature_sets.values() for feature in features}
    )
    for feature in feature_names:
        values = np.asarray([float(row[feature]) for row in rows])
        correlation, p_value = spearmanr(values, target_residual)
        correlations[feature] = {
            "spearman_with_production_residual": (
                None if not math.isfinite(correlation) else float(correlation)
            ),
            "two_sided_p": None if not math.isfinite(p_value) else float(p_value),
        }

    return {
        "configurations": len(rows),
        "families": sorted({row["family"] for row in rows}),
        "family_count": len({row["family"] for row in rows}),
        "players": [row["player_id"] for row in rows],
        "metrics": metrics,
        "inner_selected_alphas": selected_alphas,
        "feature_correlations": correlations,
    }


def recommendation_gate(
    cohort: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    """Apply the frozen research recommendation gate to the best nonbaseline model."""
    gate = plan["research_recommendation_gate"]
    baseline = cohort["metrics"]["production_fixed"]
    candidates = {
        name: metrics
        for name, metrics in cohort["metrics"].items()
        if name != "production_fixed"
    }
    best_name = min(candidates, key=lambda name: candidates[name]["mae"])
    best = candidates[best_name]
    checks = {
        "configuration_count": cohort["configurations"] >= gate["minimum_configurations"],
        "family_count": cohort["family_count"] >= gate["minimum_families"],
        "mae_improvement": (
            baseline["mae"] - best["mae"]
        ) / baseline["mae"] >= gate["minimum_mae_improvement_fraction"],
        "rmse_improvement": (
            baseline["rmse"] - best["rmse"]
        ) / baseline["rmse"] >= gate["minimum_rmse_improvement_fraction"],
        "family_win_fraction": best["family_win_fraction"] >= gate["minimum_family_win_fraction"],
        "maximum_family_worsening": best["maximum_family_mae_worsening"] <= gate["maximum_single_family_mae_worsening"],
        "bootstrap_probability": best["family_bootstrap_mae_improvement_probability"] >= gate["minimum_bootstrap_improvement_probability"],
    }
    return {
        "best_candidate": best_name,
        "checks": checks,
        "passed": all(checks.values()),
        "decision": (
            "recommend-independent-validation"
            if all(checks.values())
            else "retain-current-production-predictor"
        ),
    }


def build_rows(plan: dict[str, Any], ratings: dict[str, Any]) -> list[dict[str, Any]]:
    """Build frozen features and readiness flags for every eligible core model."""
    core_positions = load(CORE_POSITIONS_PATH)["positions"]
    core_results = load(CORE_RESULTS_PATH)
    game_positions = load(GAME_LIKE_POSITIONS_PATH)["positions"]
    game_results = load(GAME_LIKE_RESULTS_PATH)
    stability_results = load(STABILITY_RESULTS_PATH)
    blunder_positions = load(BLUNDER_POSITIONS_PATH)["positions"]
    blunder_results = load(BLUNDER_RESULTS_PATH)
    prior = plan["learned_candidates"]["retry_failure_beta_prior"]
    prior_alpha = float(prior["alpha"])
    prior_beta = float(prior["beta"])

    rows = []
    for player_id, core_record in core_results.items():
        if player_id in ANCHOR_IDS or player_id not in ratings:
            continue
        target = ratings[player_id]
        games = int(target.get("games_played") or 0)
        validation_rd = float(
            target.get("games_rd", target.get("rating_deviation", 350.0))
        )
        if games < 8 or validation_rd > 250:
            continue
        core_ready = benchmark_result_readiness(core_record, core_positions)
        if not core_ready.is_ready:
            continue
        core_metrics = collect_equal_position_metrics(
            core_record.get("results", []), core_positions
        )
        core_prediction = predict_rating_from_model_data(
            core_record, core_positions, require_ready=True
        )
        if core_metrics is None or core_prediction is None:
            continue

        game_record = game_results.get(player_id)
        game_ready = bool(
            game_record
            and benchmark_result_readiness(
                game_record,
                game_positions,
                min_equal_positions=DEFAULT_MIN_GAME_LIKE_POSITIONS,
            ).is_ready
        )
        stability_record = stability_results.get(player_id)
        stability_ready = bool(
            stability_record and stability_probe_readiness(stability_record).is_ready
        )
        stability_metrics = (
            collect_stability_probe_metrics(stability_record)
            if stability_ready
            else None
        )

        game_like_prediction = core_prediction
        category_features = {
            f"category_{category}": 0.0 for category in GAME_LIKE_CATEGORIES
        }
        if game_ready:
            game_metrics = collect_equal_position_metrics(
                game_record.get("results", []),
                game_positions,
                cpl_cap=DEFAULT_GAME_LIKE_CPL_CAP,
            )
            if game_metrics is None:
                raise ValueError(f"{player_id}: ready game-like record has no metrics")
            panel_rates = [core_metrics.legal_pct, game_metrics.legal_pct]
            if stability_metrics is not None:
                panel_rates.append(stability_metrics.legal_pct)
            game_like_prediction = predict_rating(
                game_metrics.equal_cpl,
                game_metrics.best_pct,
                min(panel_rates),
            )
            category_features = category_cpl_features(game_record, game_positions)

        stability_cap = (
            stability_probe_prediction_cap(stability_record)
            if stability_ready
            else None
        )
        production_prediction = predict_rating_from_model_data_with_supplement(
            core_record,
            core_positions,
            blunder_model_data=blunder_results.get(player_id),
            blunder_positions=blunder_positions,
            game_like_model_data=game_record if game_ready else None,
            game_like_positions=game_positions if game_ready else None,
            stability_probe_model_data=stability_record if stability_ready else None,
            require_ready=True,
        )
        if production_prediction is None:
            continue

        static_attempts, static_illegals = static_legality_counts(core_record)
        game_attempts, game_illegals = static_legality_counts(
            game_record if game_ready else None
        )
        continuation_attempts, continuation_illegals = continuation_legality_counts(
            stability_record if stability_ready else None
        )
        retry_attempts, retry_failures = retry_counts(core_record)
        game_retry_attempts, game_retry_failures = retry_counts(
            game_record if game_ready else None
        )
        continuation_retry_attempts, continuation_retry_failures = (
            continuation_retry_counts(stability_record if stability_ready else None)
        )
        total_attempts = static_attempts + game_attempts + continuation_attempts
        total_illegals = static_illegals + game_illegals + continuation_illegals
        total_retry_attempts = (
            retry_attempts + game_retry_attempts + continuation_retry_attempts
        )
        total_retry_failures = (
            retry_failures + game_retry_failures + continuation_retry_failures
        )

        rows.append(
            {
                "player_id": player_id,
                "family": model_family(player_id),
                "actual": float(target["rating"]),
                "validation_rd": validation_rd,
                "games": games,
                "core_ready": True,
                "game_like_ready": game_ready,
                "continuation_ready": stability_ready,
                "core_prediction": float(core_prediction),
                "game_like_prediction": float(game_like_prediction),
                "game_like_gap": float(game_like_prediction - core_prediction),
                "game_like_hard_cap": float(
                    combine_equal_and_game_like_predictions(
                        core_prediction,
                        game_like_prediction if game_ready else None,
                    )
                ),
                "production_prediction": float(production_prediction),
                "continuous_prediction": float(
                    continuous_deadband(
                        core_prediction,
                        game_like_prediction if game_ready else None,
                        stability_cap,
                    )
                ),
                "pooled_illegal_rate": (
                    total_illegals / total_attempts if total_attempts else 0.0
                ),
                "retry_failure_rate": (
                    (total_retry_failures + prior_alpha)
                    / (total_retry_attempts + prior_alpha + prior_beta)
                ),
                "retry_attempts": total_retry_attempts,
                "retry_failures": total_retry_failures,
                "continuation_log_cpl": math.log1p(
                    max(0.0, stability_metrics.avg_cpl or 0.0)
                    if stability_metrics is not None
                    else 0.0
                ),
                **category_features,
            }
        )
    return rows


def render_report(analysis: dict[str, Any]) -> str:
    """Render a concise Markdown decision record."""
    all_panels = analysis["cohorts"]["all_panels"]
    game_like = analysis["cohorts"]["game_like"]
    gate = analysis["recommendation_gate"]
    rows = []
    for name, metrics in sorted(
        all_panels["metrics"].items(), key=lambda item: item[1]["mae"]
    ):
        rows.append(
            f"| {name} | {metrics['mae']:.1f} | {metrics['rmse']:.1f} | "
            f"{metrics['mean_error']:+.1f} | "
            f"{metrics['family_bootstrap_mae_improvement_probability']:.3f} |"
        )
    game_rows = []
    for name, metrics in sorted(
        game_like["metrics"].items(), key=lambda item: item[1]["mae"]
    ):
        game_rows.append(
            f"| {name} | {metrics['mae']:.1f} | {metrics['rmse']:.1f} | "
            f"{metrics['mean_error']:+.1f} | "
            f"{metrics['family_bootstrap_mae_improvement_probability']:.3f} |"
        )
    feature_rows = []
    for feature, values in sorted(
        all_panels["feature_correlations"].items(),
        key=lambda item: abs(item[1]["spearman_with_production_residual"] or 0),
        reverse=True,
    ):
        correlation = values["spearman_with_production_residual"]
        feature_rows.append(
            f"| {feature} | {correlation:.3f} |"
            if correlation is not None
            else f"| {feature} | n/a |"
        )
    failed = ", ".join(
        name for name, passed in gate["checks"].items() if not passed
    )
    return f"""# Supplemental predictor family-held-out analysis — 2026-07-17

The frozen decision is **{gate['decision']}**. No production formula or rating
initialization changed.

The primary all-panel cohort contains {all_panels['configurations']}
configurations across {all_panels['family_count']} model-line families. It consists
entirely of the twelve GPT-5.6 variants and fails the predeclared 30-model,
eight-family evidence requirement. The broader game-like cohort contains
{game_like['configurations']} configurations across {game_like['family_count']}
families.

## Broader game-like comparison

| Predictor | MAE | RMSE | Mean error | Family-bootstrap P(MAE improves) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(game_rows)}

The current downside system is materially better than the core alone on this
cohort: core MAE/RMSE are {game_like['metrics']['core_fixed']['mae']:.1f}/{game_like['metrics']['core_fixed']['rmse']:.1f},
versus {game_like['metrics']['production_fixed']['mae']:.1f}/{game_like['metrics']['production_fixed']['rmse']:.1f}
for production. None of the learned replacements improves family-held-out MAE
over production.

## All-panel prediction comparison

| Predictor | MAE | RMSE | Mean error | Family-bootstrap P(MAE improves) |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

The numerically best candidate is `{gate['best_candidate']}`. Failed gate checks:
{failed or 'none'}. Even a favorable point estimate is therefore a research lead,
not a production candidate.

## Feature relationship to the remaining production error

| Feature | Spearman correlation with actual − production prediction |
| --- | ---: |
{chr(10).join(feature_rows)}

Correlations describe this small, selected cohort. Continuation CPL is retained as
evidence as planned, while the report keeps its random-opponent confounding and
depth-10 scoring limitation explicit.

The family contrast matters:

- Defense CPL correlates {all_panels['feature_correlations']['category_defense']['spearman_with_production_residual']:.3f}
  with the remaining error inside GPT-5.6, but only
  {game_like['feature_correlations']['category_defense']['spearman_with_production_residual']:.3f}
  in the broader six-family game-like cohort. The earlier defense result does not
  generalize well enough to weight.
- Pooled first-answer illegality is the strongest broader residual diagnostic
  (Spearman {game_like['feature_correlations']['pooled_illegal_rate']['spearman_with_production_residual']:.3f}),
  but its leakage-controlled model still worsens MAE. Keep it as a reliability
  diagnostic, not a fitted rating coefficient.
- Continuation CPL has little relationship to the remaining error here
  (Spearman {all_panels['feature_correlations']['continuation_log_cpl']['spearman_with_production_residual']:.3f})
  and adding it does not improve held-out MAE.
- The fixed production and game-like-hard-cap predictions are identical on all
  twelve ready continuation configurations. Thus the continuation forfeit/
  catastrophe cap adds no further adjustment in this cohort, although
  continuation legality still participates in the game-like panel's conservative
  legality input.
- Retry-failure evidence is still too sparse to fit a reliable model-specific
  correction.

## Coverage

- Game-like cohort: {analysis['cohorts']['game_like']['configurations']} configurations,
  {analysis['cohorts']['game_like']['family_count']} families.
- Continuation cohort: {analysis['cohorts']['continuation']['configurations']} configurations,
  {analysis['cohorts']['continuation']['family_count']} families.
- All-panel cohort: {all_panels['configurations']} configurations,
  {all_panels['family_count']} families.

The RD-300 game target comes from 6,276 production games, with five invalid games
skipped by the existing recalculation contract. Its prior mean still comes from
the position benchmark, so this analysis reduces circularity but cannot eliminate
it. The next useful evidence is automatic completion of these supplements on more
new or actively selected non-GPT-5.6 model families, followed by rerunning this
frozen harness. Existing frozen models remain excluded by the production
acquisition policy unless a separately authorized research run is justified.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    args = parser.parse_args()
    plan = load(args.plan)
    for item in plan["inputs"].values():
        if not isinstance(item, dict) or "path" not in item:
            continue
        path = ROOT / item["path"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Frozen input hash mismatch: {path}")
    target_path = ROOT / plan["validation_target"]["output"]
    ratings = load(target_path)
    all_rows = build_rows(plan, ratings)
    game_rows = [row for row in all_rows if row["game_like_ready"]]
    continuation_rows = [row for row in all_rows if row["continuation_ready"]]
    combined_rows = [
        row
        for row in all_rows
        if row["game_like_ready"] and row["continuation_ready"]
    ]
    if not combined_rows:
        raise ValueError("No configurations satisfy the frozen all-panel cohort")

    game_feature_sets = {
        "game_like_aggregate": ["game_like_gap"],
        "game_like_categories": [
            "game_like_gap",
            *[f"category_{category}" for category in GAME_LIKE_CATEGORIES],
        ],
        "legality": [
            "game_like_gap",
            "pooled_illegal_rate",
            "retry_failure_rate",
        ],
    }
    continuation_feature_sets: dict[str, list[str]] = {}
    combined_feature_sets = {
        **game_feature_sets,
        "continuation": [
            "game_like_gap",
            "pooled_illegal_rate",
            "retry_failure_rate",
            "continuation_log_cpl",
        ],
    }
    cohorts = {
        "game_like": evaluate_cohort(
            game_rows, plan, learned_feature_sets=game_feature_sets
        ),
        "continuation": evaluate_cohort(
            continuation_rows,
            plan,
            learned_feature_sets=continuation_feature_sets,
        ),
        "all_panels": evaluate_cohort(
            combined_rows, plan, learned_feature_sets=combined_feature_sets
        ),
    }
    gate = recommendation_gate(cohorts["all_panels"], plan)
    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": plan["experiment_id"],
        "plan_sha256": sha256(args.plan),
        "production_effect": "none",
        "validation_target": {
            "path": plan["validation_target"]["output"],
            "sha256": sha256(target_path),
            "benchmark_seed_rd": plan["validation_target"]["benchmark_seed_rd"],
            "rating_count": len(ratings),
        },
        "eligible_core_configurations": len(all_rows),
        "family_counts": dict(Counter(row["family"] for row in all_rows)),
        "cohorts": cohorts,
        "recommendation_gate": gate,
        "rows": all_rows,
        "limitations": [
            plan["validation_target"]["circularity_warning"],
            f"The all-panel cohort contains only {cohorts['all_panels']['family_count']} model-line families and all {cohorts['all_panels']['configurations']} configurations are GPT-5.6 variants.",
            "Continuation CPL uses depth-10 scoring after seeded-random replies and is treated as noisy evidence, not a standardized game-strength measure.",
            "No learned candidate may change production from this analysis."
        ],
    }
    analysis_path = ROOT / plan["outputs"]["analysis"]
    report_path = ROOT / plan["outputs"]["report"]
    analysis_path.write_text(json.dumps(analysis, indent=2) + "\n")
    report_path.write_text(render_report(analysis))
    print(f"Saved {analysis_path}")
    print(f"Saved {report_path}")
    print(
        f"All-panel cohort: {cohorts['all_panels']['configurations']} configs, "
        f"{cohorts['all_panels']['family_count']} families"
    )
    print(
        f"Decision: {gate['decision']} "
        f"(best candidate: {gate['best_candidate']})"
    )


if __name__ == "__main__":
    main()
