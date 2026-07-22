#!/usr/bin/env python3
"""Evaluate final-rating extrapolation across reasoning efforts.

This analysis uses final production game-benchmark ratings only. Position
benchmark data is intentionally absent because its normal seeding effect is
already baked into those final ratings.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rating.reasoning_prediction import (  # noqa: E402
    CurvePrior,
    EFFORT_INDEX,
    build_reasoning_curves,
    fit_curve_prior,
    predict_final_rating,
)


DEFAULT_RATINGS = ROOT / "data/ratings.json"
DEFAULT_METADATA = ROOT / "data/model_publish_dates.json"
DEFAULT_JSON = (
    ROOT
    / "position_benchmark/validation/2026-07-17-reasoning-rating-analysis.json"
)
DEFAULT_MARKDOWN = (
    ROOT
    / "position_benchmark/validation/2026-07-17-reasoning-rating-analysis.md"
)
MAX_TARGET_RD = 220.0
MIN_GAMES = 8


POLICIES = (
    {"name": "medium_to_high", "anchors": ("medium",), "target": "high"},
    {"name": "low_medium_to_high", "anchors": ("low", "medium"), "target": "high"},
    {"name": "medium_to_xhigh", "anchors": ("medium",), "target": "xhigh"},
    {"name": "low_medium_to_xhigh", "anchors": ("low", "medium"), "target": "xhigh"},
    {"name": "high_to_xhigh", "anchors": ("high",), "target": "xhigh"},
)


def load(path: Path) -> Any:
    """Load JSON from disk."""
    return json.loads(path.read_text())


def metric_summary(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    """Calculate final-rating error metrics for one candidate."""
    errors = np.asarray([row[key] - row["actual"] for row in rows], dtype=float)
    if not len(errors):
        return {"mae": math.nan, "rmse": math.nan, "max_absolute_error": math.nan}
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "max_absolute_error": float(np.max(np.abs(errors))),
        "mean_error": float(np.mean(errors)),
    }


def global_only(prior: CurvePrior) -> CurvePrior:
    """Return the global curve as a baseline with identical provenance."""
    return CurvePrior(
        increments=prior.global_increments,
        global_increments=prior.global_increments,
        training_lines=prior.training_lines,
        lab_training_lines=(),
    )


def linear_prediction(curve: Any, anchors: Sequence[str], target: str) -> float:
    """Linearly extrapolate the last two final ratings on the ordinal ladder."""
    if len(anchors) < 2:
        return curve.observations[anchors[-1]].rating
    left, right = anchors[-2:]
    left_index = EFFORT_INDEX[left]
    right_index = EFFORT_INDEX[right]
    target_index = EFFORT_INDEX[target]
    left_rating = curve.observations[left].rating
    right_rating = curve.observations[right].rating
    slope = (right_rating - left_rating) / (right_index - left_index)
    return float(right_rating + slope * (target_index - right_index))


def evaluate_policy(
    curves: dict[str, Any],
    *,
    anchors: Sequence[str],
    target: str,
) -> dict[str, Any]:
    """Evaluate a fixed acquisition policy with model and release leakage blocked."""
    eligible = []
    for curve in curves.values():
        if target not in curve.observations:
            continue
        if any(anchor not in curve.observations for anchor in anchors):
            continue
        required = [curve.observations[target]] + [
            curve.observations[anchor] for anchor in anchors
        ]
        if any(point.games_played < MIN_GAMES for point in required):
            continue
        if any(point.rating_deviation > MAX_TARGET_RD for point in required):
            continue
        eligible.append(curve)

    rows: list[dict[str, Any]] = []
    for curve in sorted(eligible, key=lambda item: item.model_id):
        prior = fit_curve_prior(
            curves,
            target_model_id=curve.model_id,
            exclude_release_cohort=True,
        )
        same_line_prior = fit_curve_prior(
            curves,
            target_model_id=curve.model_id,
            exclude_release_cohort=False,
        )
        combination = "mean" if len(anchors) > 1 and target == "xhigh" else "latest"
        lab_prediction = predict_final_rating(
            curve,
            anchor_efforts=anchors,
            target_effort=target,
            prior=prior,
            combination=combination,
        ).rating
        global_prediction = predict_final_rating(
            curve,
            anchor_efforts=anchors,
            target_effort=target,
            prior=global_only(prior),
            combination=combination,
        ).rating
        sibling_prediction = predict_final_rating(
            curve,
            anchor_efforts=anchors,
            target_effort=target,
            prior=same_line_prior,
            combination=combination,
        ).rating
        latest = max(anchors, key=EFFORT_INDEX.__getitem__)
        rows.append(
            {
                "model_id": curve.model_id,
                "lab": curve.lab,
                "release_cohort": curve.release_cohort,
                "actual": curve.observations[target].rating,
                "actual_rd": curve.observations[target].rating_deviation,
                "actual_games": curve.observations[target].games_played,
                "anchor_ratings": {
                    anchor: curve.observations[anchor].rating for anchor in anchors
                },
                "carry_latest": curve.observations[latest].rating,
                "linear_ordinal": linear_prediction(curve, anchors, target),
                "global_curve": global_prediction,
                "lab_curve_release_holdout": lab_prediction,
                "lab_curve_sibling_allowed": sibling_prediction,
                "anchor_combination": combination,
                "lab_training_lines": list(prior.lab_training_lines),
            }
        )

    candidate_keys = (
        "carry_latest",
        "linear_ordinal",
        "global_curve",
        "lab_curve_release_holdout",
        "lab_curve_sibling_allowed",
    )
    return {
        "anchors": list(anchors),
        "target": target,
        "eligible_lines": len(rows),
        "metrics": {key: metric_summary(rows, key) for key in candidate_keys},
        "rows": rows,
    }


def render_markdown(analysis: dict[str, Any]) -> str:
    """Render the concise human-readable validation report."""
    lines = [
        "# Final-rating reasoning-level prediction — 2026-07-17",
        "",
        "This analysis predicts **final production game-benchmark ratings** from",
        "one or two other final production ratings in the same model line. The",
        "ordinary position-benchmark initialization is already reflected in every",
        "rating; no position features or separate position blend are used.",
        "",
        "The primary `lab_curve_release_holdout` result excludes the entire target",
        "release cohort while learning the prior. In particular, Luna, Terra, and",
        "Sol are hidden together when GPT-5.6 is treated as a new release.",
        "",
        "## Results",
        "",
        "| Policy | Lines | Carry MAE | Global MAE | Lab/release-held MAE | RMSE | Max error |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in analysis["policies"].items():
        metrics = result["metrics"]
        primary = metrics["lab_curve_release_holdout"]
        lines.append(
            f"| {name} | {result['eligible_lines']} | "
            f"{metrics['carry_latest']['mae']:.0f} | "
            f"{metrics['global_curve']['mae']:.0f} | "
            f"{primary['mae']:.0f} | {primary['rmse']:.0f} | "
            f"{primary['max_absolute_error']:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The model is suitable as a high-RD prior, not as a replacement for",
            "measuring a reasoning level. Low plus Medium does not reliably predict",
            "Xhigh across releases: GPT-5.5's nearly flat curve did not anticipate",
            "GPT-5.6's much larger Xhigh gains. High is a materially better anchor",
            "for Xhigh.",
            "",
            "Generic `(thinking)` variants were inventoried but not forced onto",
            "Medium or High because different labs use that label incompatibly.",
            "Sparse targets or anchors below eight games, or with RD above 220, are",
            "retained in the source inventory but excluded from scored validation.",
            "",
            "No production rating or seeding behavior is changed by this research.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run the final-rating reasoning-level analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=DEFAULT_RATINGS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_MARKDOWN)
    args = parser.parse_args()

    ratings = load(args.ratings)
    metadata = load(args.metadata)
    curves = build_reasoning_curves(ratings, metadata)
    policies = {
        policy["name"]: evaluate_policy(
            curves,
            anchors=policy["anchors"],
            target=policy["target"],
        )
        for policy in POLICIES
    }
    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "target": "final production game-benchmark rating",
        "position_features_used": False,
        "ratings_path": str(args.ratings.relative_to(ROOT)),
        "metadata_path": str(args.metadata.relative_to(ROOT)),
        "multi_effort_lines": len(curves),
        "eligibility": {"min_games": MIN_GAMES, "max_rd": MAX_TARGET_RD},
        "validation": "leave-target-release-cohort-out",
        "policies": policies,
        "limitations": [
            "High/Xhigh coverage is small, especially for complete four-level curves.",
            "Final ratings still contain Glicko uncertainty and the normal benchmark prior.",
            "GPT-5.6 is the only multi-line release cohort, so release transfer is weakly identified.",
            "Candidate comparison is exploratory and is not authorization for a production change.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(analysis, indent=2) + "\n")
    args.output_markdown.write_text(render_markdown(analysis))
    print(render_markdown(analysis))


if __name__ == "__main__":
    main()
