#!/usr/bin/env python3
"""Compare reasoning predictions with position-free game-rating targets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rating.reasoning_prediction import (  # noqa: E402
    build_reasoning_curves,
    fit_curve_prior,
)
from scripts.analyze_supplemental_predictors import build_rows  # noqa: E402


DEFAULT_COUNTERFACTUAL = (
    ROOT
    / "position_benchmark/validation/2026-07-17-no-position-seed-ratings.json"
)
DEFAULT_PLAN = (
    ROOT
    / "position_benchmark/validation/2026-07-17-supplemental-predictor-plan.json"
)
DEFAULT_JSON = (
    ROOT
    / "position_benchmark/validation/2026-07-17-position-seed-counterfactual.json"
)
DEFAULT_MARKDOWN = (
    ROOT
    / "position_benchmark/validation/2026-07-17-position-seed-counterfactual.md"
)
EXACT_CASES = (
    (
        "GPT-5.5",
        "openai/gpt-5.5",
        "gpt-5.5 (high)",
        "gpt-5.5 (xhigh)",
        "gpt-5.5",
    ),
    *(
        (
            f"GPT-5.6 {branch.title()}",
            f"openai/gpt-5.6-{branch}",
            f"gpt-5.6-{branch} (high)",
            f"gpt-5.6-{branch} (xhigh)",
            "gpt-5.6",
        )
        for branch in ("sol", "terra", "luna")
    ),
    (
        "DeepSeek V4 Flash",
        "deepseek/deepseek-v4-flash",
        "deepseek-v4-flash (high)",
        "deepseek-v4-flash (max)",
        "deepseek-v4-flash",
    ),
)
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def load(path: Path) -> Any:
    """Load JSON from disk."""
    return json.loads(path.read_text())


def metrics(errors: list[float]) -> dict[str, float]:
    """Summarize signed prediction errors."""
    return {
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
        "mean_error": sum(errors) / len(errors),
        "max_absolute_error": max(abs(error) for error in errors),
    }


def blend_metrics(rows: list[dict[str, float]], weight: float) -> dict[str, Any]:
    """Score one position-prediction weight against independent targets."""
    errors = [
        (1.0 - weight) * row["high_curve_prediction"]
        + weight * row["position_prediction"]
        - row["target_rating"]
        for row in rows
    ]
    return {"position_weight": weight, **metrics(errors), "errors": errors}


def optimal_weight(rows: list[dict[str, float]]) -> float:
    """Fit the unconstrained least-squares position weight."""
    deltas = [
        row["position_prediction"] - row["high_curve_prediction"] for row in rows
    ]
    targets = [
        row["target_rating"] - row["high_curve_prediction"] for row in rows
    ]
    denominator = sum(delta * delta for delta in deltas)
    return sum(delta * target for delta, target in zip(deltas, targets)) / denominator


def cohort_metrics(
    player_ids: list[str],
    *,
    ratings: dict[str, Any],
    position_predictions: dict[str, float],
) -> dict[str, Any]:
    """Score position predictions against position-free ratings."""
    retained = [
        player_id
        for player_id in player_ids
        if player_id in ratings and player_id in position_predictions
    ]
    errors = [
        position_predictions[player_id] - ratings[player_id]["rating"]
        for player_id in retained
    ]
    return {"configurations": len(retained), **metrics(errors)}


def model_timestamps(metadata: dict[str, Any]) -> dict[str, int]:
    """Return the earliest known release timestamp for each underlying model."""
    timestamps: dict[str, int] = {}
    for row in metadata.values():
        model_id = str(row["model_id"])
        timestamp = int(row.get("created_timestamp") or 0)
        existing = timestamps.get(model_id)
        if existing is None or timestamp < existing:
            timestamps[model_id] = timestamp
    return timestamps


def render_markdown(analysis: dict[str, Any]) -> str:
    """Render the concise counterfactual report."""
    grid = analysis["exact_xhigh"]["fixed_blends"]
    rows = analysis["exact_xhigh"]["rows"]
    holdout = analysis["exact_xhigh"]["gpt55_chronological_holdout"]
    cross_lab = analysis["exact_xhigh"]["deepseek_cross_lab_holdout"]
    lines = [
        "# Position-seed counterfactual — 2026-07-17",
        "",
        "All 6,276 production games were recalculated in an isolated rating store",
        "with position-benchmark predictions disabled. Non-anchor models therefore",
        "started from the ordinary legality/model-type fallback with RD 350; normal",
        "later-pass lower-effort sibling seeding was retained. Production ratings",
        "were not changed.",
        "",
        "## High final rating + Xhigh-position prediction",
        "",
        "Each High-only baseline uses a High→Xhigh increment learned only from",
        "earlier releases, excluding the target release cohort. GPT-5.5 therefore",
        "uses a zero increment: no earlier OpenAI line had an Xhigh game rating.",
        "",
        "| Position weight | MAE | RMSE | Bias |",
        "|---:|---:|---:|---:|",
    ]
    for result in grid:
        lines.append(
            f"| {result['position_weight']:.0%} | {result['mae']:.0f} | "
            f"{result['rmse']:.0f} | {result['mean_error']:+.0f} |"
        )
    lines.extend(
        [
            "",
            "| Model | Independent Xhigh | RD | High+curve | Position | 50/50 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['target_rating']:.0f} | "
            f"{row['target_rd']:.0f} | {row['high_curve_prediction']:.0f} | "
            f"{row['position_prediction']:.0f} | {row['blend_50_prediction']:.0f} |"
        )
    lines.extend(
        [
            "",
            f"Across all five exact cases, High-only MAE is {grid[0]['mae']:.0f},",
            f"position-only MAE is {grid[-1]['mae']:.0f}, and the fixed 50/50 blend",
            f"is best on the fixed grid at {grid[2]['mae']:.0f} MAE",
            f"({grid[2]['rmse']:.0f} RMSE).",
            "",
            "GPT-5.5 is a clean chronological holdout for the already-proposed 50/50",
            f"blend. Its absolute errors are {holdout['high_curve_absolute_error']:.0f}",
            f"for High-only, {holdout['position_absolute_error']:.0f} for position-only,",
            f"and {holdout['blend_50_absolute_error']:.0f} for the fixed blend. Thus it",
            "confirms a modest incremental benefit over High alone while strongly",
            "rejecting position-only prediction for this release.",
            "",
            "DeepSeek V4 Flash is the first frozen cross-lab test, and it is a clear",
            f"miss: High-only error {cross_lab['high_curve_absolute_error']:.0f},",
            f"position-only error {cross_lab['position_absolute_error']:.0f}, and",
            f"50/50 error {cross_lab['blend_50_absolute_error']:.0f}. The blend still",
            "has the lowest aggregate MAE, but this case shows that the OpenAI result",
            "does not transfer cleanly across labs.",
            "",
            "The evidence now spans five configurations, two labs, and three model",
            "releases. It is still too small to tune or deploy a production weight.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the isolated position-seed counterfactual analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfactual", type=Path, default=DEFAULT_COUNTERFACTUAL)
    parser.add_argument("--production", type=Path, default=ROOT / "data/ratings.json")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/model_publish_dates.json")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    ratings = load(args.counterfactual)
    production = load(args.production)
    metadata = load(args.metadata)
    plan = load(args.plan)
    current_rows = build_rows(plan, ratings)
    position_predictions = {
        row["player_id"]: float(row["production_prediction"])
        for row in current_rows
    }
    curves = build_reasoning_curves(ratings, metadata)
    timestamps = model_timestamps(metadata)

    rows: list[dict[str, Any]] = []
    for label, model_id, high_id, xhigh_id, release_cohort in EXACT_CASES:
        cutoff = timestamps[model_id]
        chronological_curves = {
            candidate_id: curve
            for candidate_id, curve in curves.items()
            if candidate_id == model_id or timestamps.get(candidate_id, 0) < cutoff
        }
        prior = fit_curve_prior(
            chronological_curves,
            target_model_id=model_id,
            exclude_release_cohort=True,
        )
        historical_delta = prior.delta("high", "xhigh")
        high_curve_prediction = ratings[high_id]["rating"] + historical_delta
        position_prediction = position_predictions[xhigh_id]
        rows.append(
            {
                "label": label,
                "model_id": model_id,
                "release_cohort": release_cohort,
                "high_id": high_id,
                "xhigh_id": xhigh_id,
                "high_rating": ratings[high_id]["rating"],
                "high_rd": ratings[high_id]["rating_deviation"],
                "historical_high_to_xhigh_delta": historical_delta,
                "high_curve_prediction": high_curve_prediction,
                "position_prediction": position_prediction,
                "blend_50_prediction": (high_curve_prediction + position_prediction) / 2,
                "target_rating": ratings[xhigh_id]["rating"],
                "target_rd": ratings[xhigh_id]["rating_deviation"],
                "games": ratings[xhigh_id]["games_played"],
                "production_target_rating": production[xhigh_id]["rating"],
                "position_free_minus_production": (
                    ratings[xhigh_id]["rating"] - production[xhigh_id]["rating"]
                ),
                "historical_training_lines": list(prior.lab_training_lines),
            }
        )

    fitted_weight = optimal_weight(rows)
    gpt55_row = rows[0]
    gpt55_holdout = {
        "high_curve_absolute_error": abs(
            gpt55_row["high_curve_prediction"] - gpt55_row["target_rating"]
        ),
        "position_absolute_error": abs(
            gpt55_row["position_prediction"] - gpt55_row["target_rating"]
        ),
        "blend_50_absolute_error": abs(
            gpt55_row["blend_50_prediction"] - gpt55_row["target_rating"]
        ),
    }
    deepseek_row = next(
        row for row in rows if row["model_id"] == "deepseek/deepseek-v4-flash"
    )
    deepseek_holdout = {
        "high_curve_absolute_error": abs(
            deepseek_row["high_curve_prediction"] - deepseek_row["target_rating"]
        ),
        "position_absolute_error": abs(
            deepseek_row["position_prediction"] - deepseek_row["target_rating"]
        ),
        "blend_50_absolute_error": abs(
            deepseek_row["blend_50_prediction"] - deepseek_row["target_rating"]
        ),
    }

    all_ids = [row["player_id"] for row in current_rows]
    game_like_ids = [
        row["player_id"] for row in current_rows if row["game_like_ready"]
    ]
    gpt56_ids = [player_id for player_id in all_ids if player_id.startswith("gpt-5.6-")]
    xhigh_ids = [player_id for player_id in gpt56_ids if player_id.endswith("(xhigh)")]
    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "production_effect": "none",
        "counterfactual": {
            "ratings_path": str(args.counterfactual.relative_to(ROOT)),
            "position_benchmark_seeds": False,
            "ordinary_initial_rd": 350,
            "fallback_initial_ratings": {
                "reasoning": 1200,
                "non_reasoning": 400,
                "low_legality": 0,
            },
            "normal_lower_effort_sibling_seeding_retained": True,
            "production_games_processed": 6276,
            "games_skipped": 5,
        },
        "exact_xhigh": {
            "rows": rows,
            "fixed_blends": [blend_metrics(rows, weight) for weight in WEIGHTS],
            "in_sample_optimal": {
                "position_weight": fitted_weight,
                **blend_metrics(rows, fitted_weight),
            },
            "gpt55_chronological_holdout": gpt55_holdout,
            "deepseek_cross_lab_holdout": deepseek_holdout,
        },
        "gpt56_xhigh": {
            "rows": [row for row in rows if row["release_cohort"] == "gpt-5.6"],
            "fixed_blends": [
                blend_metrics(
                    [row for row in rows if row["release_cohort"] == "gpt-5.6"],
                    weight,
                )
                for weight in WEIGHTS
            ],
            "production_target_position_only": cohort_metrics(
                xhigh_ids,
                ratings=production,
                position_predictions=position_predictions,
            ),
            "position_free_target_position_only": cohort_metrics(
                xhigh_ids,
                ratings=ratings,
                position_predictions=position_predictions,
            ),
        },
        "broader_position_only_checks": {
            "all_eligible": cohort_metrics(
                all_ids,
                ratings=ratings,
                position_predictions=position_predictions,
            ),
            "game_like_ready": cohort_metrics(
                game_like_ids,
                ratings=ratings,
                position_predictions=position_predictions,
            ),
            "gpt56_all_efforts": cohort_metrics(
                gpt56_ids,
                ratings=ratings,
                position_predictions=position_predictions,
            ),
        },
        "limitations": [
            "Only five exact Xhigh cases are available across two labs and three model releases.",
            "The GPT-5.6 Xhigh targets each have eight games and RD between 186 and 205.",
            "The no-position rating removes prior-mean circularity but remains a noisy game estimate.",
            "The in-sample optimal blend weight is descriptive, not a production recommendation.",
        ],
    }
    args.output_json.write_text(json.dumps(analysis, indent=2) + "\n")
    markdown = render_markdown(analysis)
    args.output_markdown.write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
