#!/usr/bin/env python3
"""Build an offline diagnostic table for screening candidates.

This scores each candidate position and feature with the failure modes that
have shown up so far:
- repeat instability
- weak-model false positives
- saturation / triviality
- family-specific bias
- ordering failures against actual ratings
- whether the feature helps current benchmark prediction out of sample

It can also ingest additional evaluation bundles on subsets of the candidate
pool by matching positions via exact FEN.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binom


ANCHOR_IDS = {"random-bot", "eubos", "maia-1900", "maia-1100", "survival-bot"}
ANCHOR_ORDER_PAIRS = [
    ("eubos", "maia-1900"),
    ("maia-1900", "maia-1100"),
    ("maia-1100", "survival-bot"),
    ("survival-bot", "random-bot"),
]
FEATURE_OPTIONS = ("top3", "top5", "reciprocal_rank", "best")
DEFAULT_ALPHAS = (0.0, 0.1, 1.0, 10.0)
STATUS_ORDER = {
    "candidate": 6,
    "neutral": 5,
    "saturated": 4,
    "unstable": 3,
    "false_positive": 2,
    "anti_signal": 1,
    "rejected_holdout": 0,
    "no_data": -1,
}


@dataclass
class BundleSpec:
    name: str
    positions_path: Path
    result_paths: list[Path]


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_positions(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    return data["positions"] if isinstance(data, dict) else data


def survival_probability(legal_pct: float, game_length: int = 40) -> float:
    illegal_rate = max(0.0, min(1.0, 1.0 - legal_pct / 100.0))
    if illegal_rate <= 0:
        return 100.0
    if illegal_rate >= 1:
        return 0.0
    return 100.0 * (
        binom.pmf(0, game_length, illegal_rate)
        + binom.pmf(1, game_length, illegal_rate)
    )


def current_equal_formula(equal_cpl: float, best_pct: float, legal_pct: float) -> float:
    surv_40 = survival_probability(legal_pct)
    return (
        1298.57
        - 200.43 * math.log(equal_cpl + 1.0)
        + 15.39 * best_pct
        + 5.85 * surv_40
    )


def current_position_prediction(summary: dict[str, Any]) -> float | None:
    equal = summary.get("equal") or summary
    try:
        equal_cpl = float(equal["avg_cpl"])
        best_pct = float(equal["best_pct"])
        legal_pct = float(equal["legal_pct"])
    except (KeyError, TypeError, ValueError):
        return None
    return current_equal_formula(equal_cpl, best_pct, legal_pct)


def model_family(player_id: str) -> str:
    player = player_id.lower()
    prefixes = [
        "gemini-3.1",
        "gemini-3",
        "gemini-2.5",
        "gemini-2.0",
        "gpt-5.4",
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5-mini",
        "gpt-5",
        "gpt-oss",
        "gpt-3.5",
        "grok-4.1",
        "grok-4",
        "grok-3",
        "deepseek-v3.2",
        "deepseek-v3.1",
        "deepseek-chat",
        "deepseek-r1",
        "claude",
        "kimi-k2",
        "llama-4",
        "llama-3.3",
        "maia",
        "random",
        "survival",
        "eubos",
        "mistral",
        "qwen3",
        "glm",
        "gemma",
        "mimo",
    ]
    for prefix in prefixes:
        if player.startswith(prefix):
            return prefix
    return player.split()[0].split("-")[0]


def mean_pairwise_abs_delta(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    total = 0.0
    count = 0
    for idx, first in enumerate(values):
        for second in values[idx + 1 :]:
            total += abs(float(first) - float(second))
            count += 1
    return total / count if count else None


def pairwise_equal_rate(values: list[float], tol: float = 1e-9) -> float | None:
    if len(values) < 2:
        return None
    equal = 0
    total = 0
    for idx, first in enumerate(values):
        for second in values[idx + 1 :]:
            if abs(float(first) - float(second)) <= tol:
                equal += 1
            total += 1
    return equal / total if total else None


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def standardize_train_test(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return (train_x - mu) / sd, (test_x - mu) / sd


def fit_ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    train_z, test_z = standardize_train_test(train_x, test_x)
    train_design = np.column_stack([np.ones(len(train_z)), train_z])
    test_design = np.column_stack([np.ones(len(test_z)), test_z])
    if alpha <= 0:
        coef, *_ = np.linalg.lstsq(train_design, train_y, rcond=None)
        return test_design @ coef

    penalty = np.eye(train_design.shape[1], dtype=float) * alpha
    penalty[0, 0] = 0.0
    gram = train_design.T @ train_design + penalty
    rhs = train_design.T @ train_y
    try:
        coef = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(gram) @ rhs
    return test_design @ coef


def lofo_ridge_predict(
    X: np.ndarray,
    y: np.ndarray,
    families: list[str],
    alpha: float,
) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    family_arr = np.asarray(families)
    all_idx = np.arange(len(y))
    for family in sorted(set(families)):
        test = all_idx[family_arr == family]
        train = all_idx[family_arr != family]
        if len(train) < 3:
            pred[test] = y[train].mean() if len(train) else y.mean()
            continue
        pred[test] = fit_ridge_predict(X[train], y[train], X[test], alpha)
    return pred


def pairwise_order_penalty(
    actual_rows: list[dict[str, Any]],
    pred: np.ndarray,
    min_gap: float,
) -> float:
    weighted_penalty = 0.0
    considered = 0
    for i, row_i in enumerate(actual_rows):
        for j in range(i + 1, len(actual_rows)):
            row_j = actual_rows[j]
            gap = abs(float(row_i["actual"]) - float(row_j["actual"]))
            if gap < min_gap:
                continue
            considered += 1
            actual_sign = float(row_i["actual"]) - float(row_j["actual"])
            pred_sign = float(pred[i]) - float(pred[j])
            if actual_sign * pred_sign <= 0:
                weighted_penalty += gap
    if considered == 0:
        return 0.0
    return weighted_penalty / considered


def anchor_order_penalty(
    actual_rows: list[dict[str, Any]],
    pred: np.ndarray,
) -> float:
    pred_by_model = {str(row["model"]): float(pred[idx]) for idx, row in enumerate(actual_rows)}
    actual_by_model = {str(row["model"]): float(row["actual"]) for row in actual_rows}
    penalty = 0.0
    for stronger, weaker in ANCHOR_ORDER_PAIRS:
        if stronger not in pred_by_model or weaker not in pred_by_model:
            continue
        if pred_by_model[stronger] <= pred_by_model[weaker]:
            penalty += max(0.0, actual_by_model[stronger] - actual_by_model[weaker])
    return penalty


def parse_bundle_spec(value: str) -> BundleSpec:
    try:
        name, positions_raw, results_raw = value.split(":", 2)
    except ValueError as exc:
        raise SystemExit(
            f"Invalid --bundle {value!r}. Expected name:positions.json:results1.json,results2.json"
        ) from exc
    result_paths = [Path(part) for part in results_raw.split(",") if part.strip()]
    if not result_paths:
        raise SystemExit(f"Bundle {name!r} has no result files.")
    return BundleSpec(name=name.strip(), positions_path=Path(positions_raw), result_paths=result_paths)


def build_rows_by_id(
    result_dicts: list[dict[str, Any]],
    ratings: dict[str, Any],
    benchmark_results: dict[str, Any],
    max_rd: float,
    include_anchors: bool,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    player_ids = sorted({player_id for result_dict in result_dicts for player_id in result_dict})
    for player_id in player_ids:
        rating = ratings.get(player_id)
        benchmark = benchmark_results.get(player_id)
        if not rating or not benchmark:
            continue
        prediction = current_position_prediction(benchmark.get("summary") or {})
        if prediction is None:
            continue
        games = int(rating.get("games_played") or 0)
        rd = rating.get("games_rd", rating.get("rating_deviation"))
        reliable = (
            player_id in ANCHOR_IDS
            or (games > 0 and rd is not None and float(rd) < max_rd)
        )
        if not reliable:
            continue
        if not include_anchors and player_id in ANCHOR_IDS:
            continue
        rows[player_id] = {
            "model": player_id,
            "actual": float(rating["rating"]),
            "current_pred": float(prediction),
            "family": model_family(player_id),
        }
    return rows


def collect_position_feature_values(
    positions: list[dict[str, Any]],
    result_dicts: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
) -> dict[int, dict[str, dict[str, list[float]]]]:
    out = {
        idx: {feature: {} for feature in FEATURE_OPTIONS}
        for idx in range(len(positions))
    }
    for result_dict in result_dicts:
        for player_id, model_data in result_dict.items():
            if player_id not in rows_by_id:
                continue
            for item in model_data.get("results", []):
                pos_idx = item.get("position_idx")
                if not isinstance(pos_idx, int) or pos_idx not in out:
                    continue
                if item.get("move_rank") is None:
                    continue
                out[pos_idx]["top3"].setdefault(player_id, []).append(1.0 if item.get("top3") else 0.0)
                out[pos_idx]["top5"].setdefault(player_id, []).append(1.0 if item.get("top5") else 0.0)
                out[pos_idx]["best"].setdefault(player_id, []).append(1.0 if item.get("is_best") else 0.0)
                out[pos_idx]["reciprocal_rank"].setdefault(player_id, []).append(
                    float(item.get("reciprocal_rank") or 0.0)
                )
    return out


def fit_score_on_rating(
    ratings_arr: np.ndarray,
    scores_arr: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    X = np.column_stack([np.ones(len(ratings_arr)), ratings_arr])
    coef, *_ = np.linalg.lstsq(X, scores_arr, rcond=None)
    pred = X @ coef
    return pred, {"intercept": float(coef[0]), "rating_coef": float(coef[1])}


def saturation_share_for_feature(feature_name: str, scores: np.ndarray) -> float:
    if not len(scores):
        return float("nan")
    if feature_name in {"best", "top3", "top5"}:
        return float(np.mean(scores >= 1.0 - 1e-9))
    return float(np.mean(scores >= 0.5))


def repeat_threshold_for_feature(feature_name: str) -> float:
    if feature_name == "reciprocal_rank":
        return 0.18
    return 0.20


def analyze_feature_values(
    *,
    feature_name: str,
    player_values: dict[str, list[float]],
    rows_by_id: dict[str, dict[str, Any]],
    pair_gap: float,
    low_rating_threshold: float,
    high_rating_threshold: float,
    alphas: tuple[float, ...],
) -> dict[str, Any]:
    mean_scores: dict[str, float] = {}
    repeat_deltas: list[float] = []
    same_rates: list[float] = []

    for player_id, values in player_values.items():
        if player_id not in rows_by_id:
            continue
        mean_scores[player_id] = float(np.mean(values))
        repeat_delta = mean_pairwise_abs_delta(values)
        if repeat_delta is not None:
            repeat_deltas.append(repeat_delta)
        same_rate = pairwise_equal_rate(values)
        if same_rate is not None:
            same_rates.append(same_rate)

    if not mean_scores:
        return {}

    rows = [rows_by_id[player_id] for player_id in sorted(mean_scores)]
    scores_arr = np.asarray([mean_scores[row["model"]] for row in rows], dtype=float)
    rating_arr = np.asarray([row["actual"] for row in rows], dtype=float)
    current_arr = np.asarray([row["current_pred"] for row in rows], dtype=float)
    families = [str(row["family"]) for row in rows]

    low_scores = [mean_scores[row["model"]] for row in rows if row["actual"] <= low_rating_threshold]
    high_scores = [mean_scores[row["model"]] for row in rows if row["actual"] >= high_rating_threshold]
    high_low_gap = None
    if low_scores and high_scores:
        high_low_gap = float(np.mean(high_scores) - np.mean(low_scores))

    high_threshold = float(np.mean(high_scores)) if high_scores else None
    weak_false_positive_models: list[str] = []
    if high_threshold is not None:
        weak_false_positive_models = sorted(
            row["model"]
            for row in rows
            if row["actual"] <= low_rating_threshold and mean_scores[row["model"]] >= high_threshold
        )

    linear_pred, linear_coef = fit_score_on_rating(rating_arr, scores_arr)
    residuals = scores_arr - linear_pred
    family_groups: dict[str, list[float]] = {}
    for row, residual in zip(rows, residuals):
        family_groups.setdefault(str(row["family"]), []).append(float(residual))
    family_residuals = {
        family: float(np.mean(values))
        for family, values in sorted(family_groups.items())
    }
    max_abs_family_residual = max((abs(value) for value in family_residuals.values()), default=0.0)

    bundle_baseline_rmse = rmse(rating_arr, current_arr)
    bundle_baseline_mae = mae(rating_arr, current_arr)
    bundle_baseline_r2 = r2(rating_arr, current_arr)
    best_fit: dict[str, Any] | None = None
    if len(rows) >= 5 and len(set(families)) >= 2:
        X = np.asarray(
            [[row["current_pred"], mean_scores[row["model"]]] for row in rows],
            dtype=float,
        )
        for alpha in alphas:
            pred = lofo_ridge_predict(X, rating_arr, families, alpha)
            fit = {
                "alpha": float(alpha),
                "rmse": rmse(rating_arr, pred),
                "mae": mae(rating_arr, pred),
                "r2": r2(rating_arr, pred),
            }
            if best_fit is None or fit["rmse"] < best_fit["rmse"]:
                best_fit = fit

    return {
        "coverage": len(rows),
        "repeat_models": len(repeat_deltas),
        "repeat_abs_delta": float(np.mean(repeat_deltas)) if repeat_deltas else None,
        "repeat_same_rate": float(np.mean(same_rates)) if same_rates else None,
        "score_mean": float(np.mean(scores_arr)),
        "score_std": float(np.std(scores_arr)),
        "saturated_share": saturation_share_for_feature(feature_name, scores_arr),
        "low_mean": float(np.mean(low_scores)) if low_scores else None,
        "high_mean": float(np.mean(high_scores)) if high_scores else None,
        "high_low_gap": high_low_gap,
        "weak_false_positive_count": len(weak_false_positive_models),
        "weak_false_positive_models": weak_false_positive_models,
        "pair_penalty": pairwise_order_penalty(rows, scores_arr, pair_gap),
        "anchor_penalty": anchor_order_penalty(rows, scores_arr),
        "family_residuals": family_residuals,
        "max_abs_family_residual": float(max_abs_family_residual),
        "linear_fit": linear_coef,
        "baseline_rmse": bundle_baseline_rmse,
        "baseline_mae": bundle_baseline_mae,
        "baseline_r2": bundle_baseline_r2,
        "best_alpha": None if best_fit is None else best_fit["alpha"],
        "panel_rmse": None if best_fit is None else best_fit["rmse"],
        "panel_mae": None if best_fit is None else best_fit["mae"],
        "panel_r2": None if best_fit is None else best_fit["r2"],
        "delta_rmse": None if best_fit is None else bundle_baseline_rmse - best_fit["rmse"],
        "delta_mae": None if best_fit is None else bundle_baseline_mae - best_fit["mae"],
    }


def classify_feature(
    feature_name: str,
    primary_metrics: dict[str, Any] | None,
    extra_bundle_metrics: dict[str, dict[str, Any]],
) -> str:
    if not primary_metrics:
        return "no_data"
    delta_rmse = primary_metrics.get("delta_rmse")
    repeat_abs_delta = primary_metrics.get("repeat_abs_delta")
    weak_fp = int(primary_metrics.get("weak_false_positive_count") or 0)
    saturated_share = primary_metrics.get("saturated_share")
    high_low_gap = primary_metrics.get("high_low_gap")

    extra_deltas = [
        metrics.get("delta_rmse")
        for metrics in extra_bundle_metrics.values()
        if metrics and metrics.get("delta_rmse") is not None
    ]
    if any(delta is not None and delta <= -10.0 for delta in extra_deltas):
        return "rejected_holdout"
    if delta_rmse is not None and delta_rmse <= -8.0:
        return "anti_signal"
    if weak_fp > 0:
        return "false_positive"
    if repeat_abs_delta is not None and repeat_abs_delta > repeat_threshold_for_feature(feature_name):
        return "unstable"
    if (
        saturated_share is not None
        and saturated_share >= 0.75
        and (delta_rmse is None or delta_rmse <= 5.0)
    ):
        return "saturated"
    if high_low_gap is not None and high_low_gap < 0:
        return "anti_signal"
    if delta_rmse is not None and delta_rmse >= 10.0:
        return "candidate"
    return "neutral"


def composite_score(
    feature_name: str,
    primary_metrics: dict[str, Any] | None,
    extra_bundle_metrics: dict[str, dict[str, Any]],
) -> float:
    if not primary_metrics:
        return -1e9
    score = float(primary_metrics.get("delta_rmse") or 0.0)
    score -= 40.0 * float(primary_metrics.get("repeat_abs_delta") or 0.0)
    score -= 15.0 * float(primary_metrics.get("weak_false_positive_count") or 0.0)
    score -= 5.0 * float(primary_metrics.get("max_abs_family_residual") or 0.0)
    score -= 0.05 * float(primary_metrics.get("pair_penalty") or 0.0)
    score -= 10.0 * float(primary_metrics.get("saturated_share") or 0.0)
    high_low_gap = primary_metrics.get("high_low_gap")
    if high_low_gap is not None:
        score += 10.0 * float(high_low_gap)
    for metrics in extra_bundle_metrics.values():
        if not metrics:
            continue
        score += 2.0 * float(metrics.get("delta_rmse") or 0.0)
        score -= 10.0 * float(metrics.get("weak_false_positive_count") or 0.0)
        score -= 20.0 * float(metrics.get("repeat_abs_delta") or 0.0)
    if classify_feature(feature_name, primary_metrics, extra_bundle_metrics) == "candidate":
        score += 25.0
    return float(score)


def summarize_best_feature(
    feature_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    best = max(feature_rows, key=lambda row: row["composite_score"])
    return {
        "position_idx": best["position_idx"],
        "feature": best["feature"],
        "status": best["status"],
        "composite_score": best["composite_score"],
        "screen_delta_rmse": best["primary"].get("delta_rmse") if best["primary"] else None,
        "screen_repeat_abs_delta": best["primary"].get("repeat_abs_delta") if best["primary"] else None,
        "screen_false_positives": best["primary"].get("weak_false_positive_count") if best["primary"] else None,
        "screen_saturated_share": best["primary"].get("saturated_share") if best["primary"] else None,
        "known_holdout_deltas": {
            name: metrics.get("delta_rmse")
            for name, metrics in best["bundles"].items()
            if metrics.get("delta_rmse") is not None
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]], bundle_names: list[str]) -> None:
    fieldnames = [
        "position_idx",
        "position_id",
        "feature",
        "bucket",
        "move_number",
        "ply",
        "source_player_id",
        "best_move_san",
        "status",
        "composite_score",
        "screen_coverage",
        "screen_delta_rmse",
        "screen_repeat_abs_delta",
        "screen_repeat_same_rate",
        "screen_high_low_gap",
        "screen_false_positive_count",
        "screen_saturated_share",
        "screen_pair_penalty",
        "screen_anchor_penalty",
        "screen_max_abs_family_residual",
    ]
    for bundle_name in bundle_names:
        fieldnames.extend(
            [
                f"{bundle_name}_coverage",
                f"{bundle_name}_delta_rmse",
                f"{bundle_name}_repeat_abs_delta",
                f"{bundle_name}_false_positive_count",
            ]
        )

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            primary = row["primary"] or {}
            out = {
                "position_idx": row["position_idx"],
                "position_id": row["position_id"],
                "feature": row["feature"],
                "bucket": row["bucket"],
                "move_number": row["move_number"],
                "ply": row["ply"],
                "source_player_id": row["source_player_id"],
                "best_move_san": row["best_move_san"],
                "status": row["status"],
                "composite_score": row["composite_score"],
                "screen_coverage": primary.get("coverage"),
                "screen_delta_rmse": primary.get("delta_rmse"),
                "screen_repeat_abs_delta": primary.get("repeat_abs_delta"),
                "screen_repeat_same_rate": primary.get("repeat_same_rate"),
                "screen_high_low_gap": primary.get("high_low_gap"),
                "screen_false_positive_count": primary.get("weak_false_positive_count"),
                "screen_saturated_share": primary.get("saturated_share"),
                "screen_pair_penalty": primary.get("pair_penalty"),
                "screen_anchor_penalty": primary.get("anchor_penalty"),
                "screen_max_abs_family_residual": primary.get("max_abs_family_residual"),
            }
            for bundle_name in bundle_names:
                metrics = row["bundles"].get(bundle_name) or {}
                out[f"{bundle_name}_coverage"] = metrics.get("coverage")
                out[f"{bundle_name}_delta_rmse"] = metrics.get("delta_rmse")
                out[f"{bundle_name}_repeat_abs_delta"] = metrics.get("repeat_abs_delta")
                out[f"{bundle_name}_false_positive_count"] = metrics.get("weak_false_positive_count")
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--ratings", type=Path, default=Path("data/ratings.json"))
    parser.add_argument("--benchmark-results", type=Path, default=Path("position_benchmark/results.json"))
    parser.add_argument("--max-rd", type=float, default=100.0)
    parser.add_argument("--include-anchors", action="store_true")
    parser.add_argument("--pair-gap", type=float, default=250.0)
    parser.add_argument("--low-rating-threshold", type=float, default=1000.0)
    parser.add_argument("--high-rating-threshold", type=float, default=1700.0)
    parser.add_argument(
        "--bundle",
        action="append",
        default=[],
        help=(
            "Extra evaluation bundle in the form "
            "name:positions.json:results1.json,results2.json"
        ),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("position_benchmark/nonopening_candidate_diagnostics.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("position_benchmark/nonopening_candidate_diagnostics.csv"),
    )
    args = parser.parse_args()

    base_positions = load_positions(args.positions)
    base_fen_to_idx = {position["fen"]: idx for idx, position in enumerate(base_positions)}
    ratings = load_json(args.ratings)
    benchmark_results = load_json(args.benchmark_results)

    primary_result_dicts = [load_json(path) for path in args.results]
    primary_rows_by_id = build_rows_by_id(
        primary_result_dicts,
        ratings,
        benchmark_results,
        args.max_rd,
        args.include_anchors,
    )
    primary_feature_values = collect_position_feature_values(
        base_positions,
        primary_result_dicts,
        primary_rows_by_id,
    )

    extra_bundle_specs = [parse_bundle_spec(value) for value in args.bundle]
    extra_bundle_metrics: dict[str, dict[int, dict[str, dict[str, Any]]]] = {}
    for spec in extra_bundle_specs:
        bundle_positions = load_positions(spec.positions_path)
        bundle_result_dicts = [load_json(path) for path in spec.result_paths]
        rows_by_id = build_rows_by_id(
            bundle_result_dicts,
            ratings,
            benchmark_results,
            args.max_rd,
            args.include_anchors,
        )
        bundle_feature_values = collect_position_feature_values(
            bundle_positions,
            bundle_result_dicts,
            rows_by_id,
        )
        mapped: dict[int, dict[str, dict[str, Any]]] = {}
        for bundle_idx, position in enumerate(bundle_positions):
            base_idx = base_fen_to_idx.get(position.get("fen"))
            if base_idx is None:
                continue
            mapped.setdefault(base_idx, {})
            for feature_name in FEATURE_OPTIONS:
                metrics = analyze_feature_values(
                    feature_name=feature_name,
                    player_values=bundle_feature_values[bundle_idx][feature_name],
                    rows_by_id=rows_by_id,
                    pair_gap=args.pair_gap,
                    low_rating_threshold=args.low_rating_threshold,
                    high_rating_threshold=args.high_rating_threshold,
                    alphas=DEFAULT_ALPHAS,
                )
                if metrics:
                    mapped[base_idx][feature_name] = metrics
        extra_bundle_metrics[spec.name] = mapped

    feature_rows: list[dict[str, Any]] = []
    by_position: dict[int, list[dict[str, Any]]] = {}
    for position_idx, position in enumerate(base_positions):
        for feature_name in FEATURE_OPTIONS:
            primary_metrics = analyze_feature_values(
                feature_name=feature_name,
                player_values=primary_feature_values[position_idx][feature_name],
                rows_by_id=primary_rows_by_id,
                pair_gap=args.pair_gap,
                low_rating_threshold=args.low_rating_threshold,
                high_rating_threshold=args.high_rating_threshold,
                alphas=DEFAULT_ALPHAS,
            )
            bundles = {
                spec.name: extra_bundle_metrics.get(spec.name, {}).get(position_idx, {}).get(feature_name, {})
                for spec in extra_bundle_specs
            }
            status = classify_feature(feature_name, primary_metrics, bundles)
            score = composite_score(feature_name, primary_metrics, bundles)
            row = {
                "position_idx": position_idx,
                "position_id": position.get("position_id"),
                "feature": feature_name,
                "bucket": position.get("screening_bucket") or position.get("regan_bucket") or position.get("bucket"),
                "move_number": position.get("move_number"),
                "ply": position.get("ply"),
                "source_player_id": position.get("source_player_id"),
                "best_move_san": position.get("best_move_san"),
                "fen": position.get("fen"),
                "primary": primary_metrics,
                "bundles": bundles,
                "status": status,
                "composite_score": score,
            }
            feature_rows.append(row)
            by_position.setdefault(position_idx, []).append(row)

    feature_rows.sort(
        key=lambda row: (STATUS_ORDER.get(row["status"], -99), row["composite_score"]),
        reverse=True,
    )
    position_summaries = [
        {
            "position_idx": position_idx,
            "position_id": base_positions[position_idx].get("position_id"),
            "bucket": base_positions[position_idx].get("screening_bucket")
            or base_positions[position_idx].get("regan_bucket")
            or base_positions[position_idx].get("bucket"),
            "move_number": base_positions[position_idx].get("move_number"),
            "ply": base_positions[position_idx].get("ply"),
            "source_player_id": base_positions[position_idx].get("source_player_id"),
            "best_move_san": base_positions[position_idx].get("best_move_san"),
            "best_feature": summarize_best_feature(rows),
        }
        for position_idx, rows in sorted(by_position.items())
    ]
    position_summaries.sort(
        key=lambda row: row["best_feature"]["composite_score"],
        reverse=True,
    )

    output = {
        "metadata": {
            "positions": str(args.positions),
            "results": [str(path) for path in args.results],
            "bundle_names": [spec.name for spec in extra_bundle_specs],
            "max_rd": args.max_rd,
            "include_anchors": args.include_anchors,
            "pair_gap": args.pair_gap,
            "low_rating_threshold": args.low_rating_threshold,
            "high_rating_threshold": args.high_rating_threshold,
        },
        "top_feature_rows": feature_rows[:60],
        "feature_rows": feature_rows,
        "position_summaries": position_summaries,
    }
    args.output_json.write_text(json.dumps(output, indent=2))
    write_csv(args.output_csv, feature_rows, [spec.name for spec in extra_bundle_specs])

    print(f"Wrote diagnostics JSON to {args.output_json}")
    print(f"Wrote diagnostics CSV to {args.output_csv}")
    print()
    print("Top candidate features:")
    for row in feature_rows[:12]:
        primary = row["primary"] or {}
        bundle_bits = []
        for spec in extra_bundle_specs:
            metrics = row["bundles"].get(spec.name) or {}
            delta = metrics.get("delta_rmse")
            if delta is not None:
                bundle_bits.append(f"{spec.name}={delta:+.1f}")
        bundle_text = f" [{' '.join(bundle_bits)}]" if bundle_bits else ""
        print(
            f"  idx={row['position_idx']:>2} {row['bucket']:<20} {row['feature']:<15} "
            f"status={row['status']:<16} score={row['composite_score']:+7.1f} "
            f"screen_delta={primary.get('delta_rmse')!s:>6} "
            f"repeat={primary.get('repeat_abs_delta')!s:>6} "
            f"weak_fp={primary.get('weak_false_positive_count')!s:>2}"
            f"{bundle_text}"
        )


if __name__ == "__main__":
    main()
