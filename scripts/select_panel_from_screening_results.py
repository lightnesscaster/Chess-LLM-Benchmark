#!/usr/bin/env python3
"""Greedily select a tiny residual panel from screening benchmark results."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binom


ANCHOR_IDS = {"random-bot", "eubos", "maia-1900", "maia-1100", "survival-bot"}
FEATURE_OPTIONS = ("top3", "top5", "reciprocal_rank", "best")
DEFAULT_BUCKET_LIMITS = {
    "quiet_equal": 3,
    "tactical_equal": 3,
    "advantage_conversion": 3,
    "defense": 2,
}
ANCHOR_ORDER_PAIRS = [
    ("eubos", "maia-1900"),
    ("maia-1900", "maia-1100"),
    ("maia-1100", "survival-bot"),
    ("survival-bot", "random-bot"),
]


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


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


def parse_limits(value: str | None) -> dict[str, int]:
    if not value:
        return dict(DEFAULT_BUCKET_LIMITS)
    limits: dict[str, int] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        name, raw_count = part.split("=", 1)
        limits[name.strip()] = int(raw_count)
    return limits


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


def build_model_rows(
    *,
    ratings: dict[str, Any],
    benchmark_results: dict[str, Any],
    screening_results_list: list[dict[str, Any]],
    max_rd: float,
    include_anchors: bool,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    player_ids = sorted({player_id for results in screening_results_list for player_id in results})
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
        reliable = player_id in ANCHOR_IDS or (games > 0 and rd is not None and float(rd) < max_rd)
        if not reliable:
            continue
        if not include_anchors and player_id in ANCHOR_IDS:
            continue
        rows[player_id] = {
            "model": player_id,
            "actual": float(rating["rating"]),
            "current_pred": float(prediction),
            "residual": float(rating["rating"]) - float(prediction),
            "family": model_family(player_id),
        }
    return rows


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


def common_model_ids(
    rows_by_id: dict[str, dict[str, Any]],
    selection: list[tuple[dict[str, Any], str]],
) -> list[str]:
    if not selection:
        return sorted(rows_by_id)
    out: list[str] = []
    for player_id in sorted(rows_by_id):
        if all(player_id in candidate["scores"][feature_name] for candidate, feature_name in selection):
            out.append(player_id)
    return out


def fit_panel(
    *,
    rows_by_id: dict[str, dict[str, Any]],
    selection: list[tuple[dict[str, Any], str]],
    model_ids: list[str],
    alpha: float,
) -> dict[str, Any]:
    used_rows = [rows_by_id[player_id] for player_id in model_ids]
    y = np.asarray([row["actual"] for row in used_rows], dtype=float)
    current = np.asarray([row["current_pred"] for row in used_rows], dtype=float)
    families = [str(row["family"]) for row in used_rows]

    if not selection:
        pred = current
    else:
        columns: list[list[float]] = [[row["current_pred"] for row in used_rows]]
        for candidate, feature_name in selection:
            columns.append([candidate["scores"][feature_name][row["model"]] for row in used_rows])
        X = np.asarray(np.column_stack(columns), dtype=float)
        pred = lofo_ridge_predict(X, y, families, alpha)

    abs_err = np.abs(y - pred)
    return {
        "rows": used_rows,
        "n_rows": len(used_rows),
        "pred": pred,
        "rmse": rmse(y, pred),
        "mae": mae(y, pred),
        "r2": r2(y, pred),
        "max_abs_err": float(abs_err.max()) if len(abs_err) else float("nan"),
        "p90_abs_err": float(np.quantile(abs_err, 0.9)) if len(abs_err) else float("nan"),
    }


def pairwise_order_penalty(
    rows: list[dict[str, Any]],
    pred: np.ndarray,
    min_gap: float,
) -> float:
    weighted_penalty = 0.0
    considered = 0
    for i, row_i in enumerate(rows):
        for j in range(i + 1, len(rows)):
            row_j = rows[j]
            gap = abs(float(row_i["actual"]) - float(row_j["actual"]))
            if gap < min_gap:
                continue
            considered += 1
            actual_sign = float(row_i["actual"]) - float(row_j["actual"])
            pred_sign = float(pred[i]) - float(pred[j])
            if actual_sign == 0:
                continue
            if actual_sign * pred_sign <= 0:
                weighted_penalty += gap
    if considered == 0:
        return 0.0
    return weighted_penalty / considered


def anchor_order_penalty(
    rows: list[dict[str, Any]],
    pred: np.ndarray,
) -> float:
    pred_by_model = {str(row["model"]): float(pred[idx]) for idx, row in enumerate(rows)}
    actual_by_model = {str(row["model"]): float(row["actual"]) for row in rows}
    penalty = 0.0
    for stronger, weaker in ANCHOR_ORDER_PAIRS:
        if stronger not in pred_by_model or weaker not in pred_by_model:
            continue
        if pred_by_model[stronger] <= pred_by_model[weaker]:
            penalty += max(0.0, actual_by_model[stronger] - actual_by_model[weaker])
    return penalty


def panel_objective(
    panel: dict[str, Any],
    *,
    pair_gap: float,
    rmse_weight: float,
    mae_weight: float,
    pair_weight: float,
    anchor_weight: float,
    tail_weight: float,
) -> dict[str, float]:
    pair_penalty = pairwise_order_penalty(panel["rows"], panel["pred"], pair_gap)
    anchor_penalty = anchor_order_penalty(panel["rows"], panel["pred"])
    objective = (
        rmse_weight * float(panel["rmse"])
        + mae_weight * float(panel["mae"])
        + pair_weight * pair_penalty
        + anchor_weight * anchor_penalty
        + tail_weight * float(panel["p90_abs_err"])
    )
    return {
        "objective": float(objective),
        "pair_penalty": float(pair_penalty),
        "anchor_penalty": float(anchor_penalty),
    }


def opening_key(position: dict[str, Any], plies: int) -> str:
    history = position.get("move_history") or []
    return " ".join(history[:plies])


def build_candidates(
    *,
    positions: list[dict[str, Any]],
    screening_results_list: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
    pair_gap: float,
    low_rating_threshold: float,
    high_rating_threshold: float,
) -> list[dict[str, Any]]:
    by_run_player_idx: list[dict[str, dict[int, dict[str, Any]]]] = []
    for screening_results in screening_results_list:
        by_player_idx: dict[str, dict[int, dict[str, Any]]] = {}
        for player_id, model_data in screening_results.items():
            if player_id not in rows_by_id:
                continue
            items = {}
            for result in model_data.get("results", []):
                idx = result.get("position_idx")
                if isinstance(idx, int):
                    items[idx] = result
            if items:
                by_player_idx[player_id] = items
        by_run_player_idx.append(by_player_idx)

    candidates: list[dict[str, Any]] = []
    for idx, position in enumerate(positions):
        score_lists = {feature: {} for feature in FEATURE_OPTIONS}
        for by_player_idx in by_run_player_idx:
            for player_id, items in by_player_idx.items():
                result = items.get(idx)
                if result is None:
                    continue
                if result.get("move_rank") is None:
                    continue
                score_lists["top3"].setdefault(player_id, []).append(1.0 if result.get("top3") else 0.0)
                score_lists["top5"].setdefault(player_id, []).append(1.0 if result.get("top5") else 0.0)
                score_lists["reciprocal_rank"].setdefault(player_id, []).append(
                    float(result.get("reciprocal_rank") or 0.0)
                )
                score_lists["best"].setdefault(player_id, []).append(1.0 if result.get("is_best") else 0.0)

        scores = {feature: {} for feature in FEATURE_OPTIONS}
        feature_stats: dict[str, dict[str, Any]] = {}
        for feature_name, player_values in score_lists.items():
            repeat_deltas: list[float] = []
            low_values: list[float] = []
            high_values: list[float] = []
            feature_rows: list[dict[str, Any]] = []
            feature_pred: list[float] = []
            for player_id, values in player_values.items():
                mean_value = float(np.mean(values))
                scores[feature_name][player_id] = mean_value
                delta = mean_pairwise_abs_delta(values)
                if delta is not None:
                    repeat_deltas.append(delta)
                row = rows_by_id.get(player_id)
                if row is None:
                    continue
                actual = float(row["actual"])
                if actual <= low_rating_threshold:
                    low_values.append(mean_value)
                if actual >= high_rating_threshold:
                    high_values.append(mean_value)
                feature_rows.append(row)
                feature_pred.append(mean_value)

            high_low_gap = None
            if low_values and high_values:
                high_low_gap = float(np.mean(high_values) - np.mean(low_values))

            single_pair_penalty = 0.0
            if len(feature_rows) >= 2:
                single_pair_penalty = pairwise_order_penalty(
                    feature_rows,
                    np.asarray(feature_pred, dtype=float),
                    pair_gap,
                )

            feature_stats[feature_name] = {
                "coverage": len(scores[feature_name]),
                "repeat_models": len(repeat_deltas),
                "repeat_abs_delta": float(np.mean(repeat_deltas)) if repeat_deltas else None,
                "high_low_gap": high_low_gap,
                "single_pair_penalty": float(single_pair_penalty),
            }

        coverage = max((len(score_map) for score_map in scores.values()), default=0)
        if coverage == 0:
            continue
        candidates.append(
            {
                "position_idx": idx,
                "position": position,
                "bucket": str(position.get("screening_bucket") or position.get("regan_bucket") or position.get("bucket") or "unknown"),
                "opening_key": opening_key(position, 8),
                "game_id": str(position.get("game_id") or ""),
                "coverage": coverage,
                "scores": scores,
                "feature_stats": feature_stats,
            }
        )
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--results", type=Path, nargs="+", required=True)
    parser.add_argument("--ratings", type=Path, default=Path("data/ratings.json"))
    parser.add_argument("--benchmark-results", type=Path, default=Path("position_benchmark/results.json"))
    parser.add_argument("--output-positions", type=Path, default=Path("position_benchmark/selected_screening_panel.json"))
    parser.add_argument("--output-report", type=Path, default=Path("position_benchmark/selected_screening_panel_report.json"))
    parser.add_argument("--panel-size", type=int, default=8)
    parser.add_argument("--bucket-limits", default=None)
    parser.add_argument("--max-same-opening", type=int, default=1)
    parser.add_argument("--max-same-game", type=int, default=1)
    parser.add_argument("--max-rd", type=float, default=100.0)
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--include-anchors", action="store_true")
    parser.add_argument("--pair-gap", type=float, default=250.0)
    parser.add_argument("--rmse-weight", type=float, default=1.0)
    parser.add_argument("--mae-weight", type=float, default=0.35)
    parser.add_argument("--pair-weight", type=float, default=0.4)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--tail-weight", type=float, default=0.25)
    parser.add_argument("--low-rating-threshold", type=float, default=1000.0)
    parser.add_argument("--high-rating-threshold", type=float, default=1700.0)
    parser.add_argument("--min-high-low-gap", type=float)
    parser.add_argument("--max-repeat-abs-delta", type=float)
    parser.add_argument("--min-repeat-models", type=int, default=2)
    parser.add_argument("--max-single-pair-penalty", type=float)
    args = parser.parse_args()

    positions_data = load_json(args.positions)
    positions = positions_data["positions"] if isinstance(positions_data, dict) else positions_data
    screening_results_list = [load_json(path) for path in args.results]
    ratings = load_json(args.ratings)
    benchmark_results = load_json(args.benchmark_results)
    rows_by_id = build_model_rows(
        ratings=ratings,
        benchmark_results=benchmark_results,
        screening_results_list=screening_results_list,
        max_rd=args.max_rd,
        include_anchors=args.include_anchors,
    )
    if len(rows_by_id) < args.min_rows:
        raise SystemExit(f"Only {len(rows_by_id)} usable model rows.")

    candidates = build_candidates(
        positions=positions,
        screening_results_list=screening_results_list,
        rows_by_id=rows_by_id,
        pair_gap=args.pair_gap,
        low_rating_threshold=args.low_rating_threshold,
        high_rating_threshold=args.high_rating_threshold,
    )
    if not candidates:
        raise SystemExit("No candidate positions with usable screening scores.")

    bucket_limits = parse_limits(args.bucket_limits)
    selected: list[tuple[dict[str, Any], str]] = []
    selected_ids: set[int] = set()
    selected_buckets: Counter[str] = Counter()
    selected_openings: Counter[str] = Counter()
    selected_games: Counter[str] = Counter()

    rankings: list[dict[str, Any]] = []
    while len(selected) < args.panel_size:
        best_pick: dict[str, Any] | None = None

        for candidate in candidates:
            pos_idx = int(candidate["position_idx"])
            if pos_idx in selected_ids:
                continue
            if selected_buckets[candidate["bucket"]] >= bucket_limits.get(candidate["bucket"], args.panel_size):
                continue
            if selected_openings[candidate["opening_key"]] >= args.max_same_opening:
                continue
            if candidate["game_id"] and selected_games[candidate["game_id"]] >= args.max_same_game:
                continue

            candidate_best: dict[str, Any] | None = None
            for feature_name in FEATURE_OPTIONS:
                feature_stats = candidate["feature_stats"][feature_name]
                if args.max_repeat_abs_delta is not None:
                    repeat_abs_delta = feature_stats.get("repeat_abs_delta")
                    repeat_models = int(feature_stats.get("repeat_models") or 0)
                    if repeat_abs_delta is None or repeat_models < args.min_repeat_models:
                        continue
                    if float(repeat_abs_delta) > args.max_repeat_abs_delta:
                        continue
                if args.min_high_low_gap is not None:
                    high_low_gap = feature_stats.get("high_low_gap")
                    if high_low_gap is None or float(high_low_gap) < args.min_high_low_gap:
                        continue
                if args.max_single_pair_penalty is not None:
                    if float(feature_stats.get("single_pair_penalty") or 0.0) > args.max_single_pair_penalty:
                        continue
                model_ids = common_model_ids(rows_by_id, selected + [(candidate, feature_name)])
                if len(model_ids) < args.min_rows:
                    continue
                base_panel = fit_panel(
                    rows_by_id=rows_by_id,
                    selection=selected,
                    model_ids=model_ids,
                    alpha=args.alpha,
                )
                panel = fit_panel(
                    rows_by_id=rows_by_id,
                    selection=selected + [(candidate, feature_name)],
                    model_ids=model_ids,
                    alpha=args.alpha,
                )
                base_objective = panel_objective(
                    base_panel,
                    pair_gap=args.pair_gap,
                    rmse_weight=args.rmse_weight,
                    mae_weight=args.mae_weight,
                    pair_weight=args.pair_weight,
                    anchor_weight=args.anchor_weight,
                    tail_weight=args.tail_weight,
                )
                panel_objective_stats = panel_objective(
                    panel,
                    pair_gap=args.pair_gap,
                    rmse_weight=args.rmse_weight,
                    mae_weight=args.mae_weight,
                    pair_weight=args.pair_weight,
                    anchor_weight=args.anchor_weight,
                    tail_weight=args.tail_weight,
                )
                trial = {
                    "candidate": candidate,
                    "feature_name": feature_name,
                    "n_rows": panel["n_rows"],
                    "base_rmse": base_panel["rmse"],
                    "panel_rmse": panel["rmse"],
                    "base_mae": base_panel["mae"],
                    "panel_mae": panel["mae"],
                    "panel_r2": panel["r2"],
                    "delta_rmse": base_panel["rmse"] - panel["rmse"],
                    "base_objective": base_objective["objective"],
                    "panel_objective": panel_objective_stats["objective"],
                    "delta_objective": base_objective["objective"] - panel_objective_stats["objective"],
                    "base_pair_penalty": base_objective["pair_penalty"],
                    "panel_pair_penalty": panel_objective_stats["pair_penalty"],
                    "base_anchor_penalty": base_objective["anchor_penalty"],
                    "panel_anchor_penalty": panel_objective_stats["anchor_penalty"],
                    "base_p90_abs_err": base_panel["p90_abs_err"],
                    "panel_p90_abs_err": panel["p90_abs_err"],
                    "feature_stats": feature_stats,
                }
                if candidate_best is None or (
                    trial["delta_objective"],
                    trial["n_rows"],
                    candidate["coverage"],
                ) > (
                    candidate_best["delta_objective"],
                    candidate_best["n_rows"],
                    candidate_best["candidate"]["coverage"],
                ):
                    candidate_best = trial

            if candidate_best is None:
                continue

            rankings.append(
                {
                    "position_idx": candidate["position_idx"],
                    "bucket": candidate["bucket"],
                    "feature": candidate_best["feature_name"],
                    "coverage": candidate["coverage"],
                    "opening_key": candidate["opening_key"],
                    "game_id": candidate["game_id"],
                    "delta_rmse": candidate_best["delta_rmse"],
                    "delta_objective": candidate_best["delta_objective"],
                    "base_rmse": candidate_best["base_rmse"],
                    "panel_rmse": candidate_best["panel_rmse"],
                    "n_rows": candidate_best["n_rows"],
                    "feature_stats": candidate_best["feature_stats"],
                }
            )

            if best_pick is None or (
                candidate_best["delta_objective"],
                candidate_best["n_rows"],
                candidate["coverage"],
            ) > (
                best_pick["delta_objective"],
                best_pick["n_rows"],
                best_pick["candidate"]["coverage"],
            ):
                best_pick = candidate_best

        if best_pick is None:
            break

        if best_pick["delta_objective"] <= 0.0:
            break

        best_pick["candidate"]["selected_stats"] = {
            "base_rmse": best_pick["base_rmse"],
            "panel_rmse": best_pick["panel_rmse"],
            "delta_rmse": best_pick["delta_rmse"],
            "base_mae": best_pick["base_mae"],
            "panel_mae": best_pick["panel_mae"],
            "base_objective": best_pick["base_objective"],
            "panel_objective": best_pick["panel_objective"],
            "delta_objective": best_pick["delta_objective"],
            "base_pair_penalty": best_pick["base_pair_penalty"],
            "panel_pair_penalty": best_pick["panel_pair_penalty"],
            "base_anchor_penalty": best_pick["base_anchor_penalty"],
            "panel_anchor_penalty": best_pick["panel_anchor_penalty"],
            "base_p90_abs_err": best_pick["base_p90_abs_err"],
            "panel_p90_abs_err": best_pick["panel_p90_abs_err"],
            "n_rows": best_pick["n_rows"],
            "feature": best_pick["feature_name"],
            "feature_stats": best_pick["feature_stats"],
        }
        selected.append((best_pick["candidate"], best_pick["feature_name"]))
        selected_ids.add(int(best_pick["candidate"]["position_idx"]))
        selected_buckets[best_pick["candidate"]["bucket"]] += 1
        selected_openings[best_pick["candidate"]["opening_key"]] += 1
        if best_pick["candidate"]["game_id"]:
            selected_games[best_pick["candidate"]["game_id"]] += 1

        print(
            f"Selected {len(selected)}/{args.panel_size}: "
            f"idx={best_pick['candidate']['position_idx']} "
            f"{best_pick['candidate']['bucket']} "
            f"{best_pick['feature_name']} "
            f"delta_obj={best_pick['delta_objective']:+.1f} "
            f"delta_rmse={best_pick['delta_rmse']:+.1f} "
            f"rows={best_pick['n_rows']}",
            flush=True,
        )

    if not selected:
        raise SystemExit("No positions selected.")

    final_model_ids = common_model_ids(rows_by_id, selected)
    current_subset = fit_panel(
        rows_by_id=rows_by_id,
        selection=[],
        model_ids=final_model_ids,
        alpha=args.alpha,
    )
    final_panel = fit_panel(
        rows_by_id=rows_by_id,
        selection=selected,
        model_ids=final_model_ids,
        alpha=args.alpha,
    )

    selected_positions = []
    for rank, (candidate, feature_name) in enumerate(selected):
        position = dict(candidate["position"])
        position["position_id"] = f"selected-panel-{rank:04d}"
        position["selection_feature"] = feature_name
        position["selection_lofo_base_rmse"] = round(float(candidate["selected_stats"]["base_rmse"]), 2)
        position["selection_lofo_panel_rmse"] = round(float(candidate["selected_stats"]["panel_rmse"]), 2)
        position["selection_lofo_delta_rmse"] = round(float(candidate["selected_stats"]["delta_rmse"]), 2)
        position["selection_rows"] = int(candidate["selected_stats"]["n_rows"])
        selected_positions.append(position)

    metadata = {
        "description": "Residual panel selected from cheap-model screening results",
        "source_positions": str(args.positions),
        "source_results": [str(path) for path in args.results],
        "panel_size": len(selected_positions),
        "bucket_limits": bucket_limits,
        "selected_bucket_counts": dict(selected_buckets),
        "max_same_opening": args.max_same_opening,
        "max_same_game": args.max_same_game,
        "alpha": args.alpha,
        "min_rows": args.min_rows,
        "pair_gap": args.pair_gap,
        "rmse_weight": args.rmse_weight,
        "mae_weight": args.mae_weight,
        "pair_weight": args.pair_weight,
        "anchor_weight": args.anchor_weight,
        "tail_weight": args.tail_weight,
        "low_rating_threshold": args.low_rating_threshold,
        "high_rating_threshold": args.high_rating_threshold,
        "min_high_low_gap": args.min_high_low_gap,
        "max_repeat_abs_delta": args.max_repeat_abs_delta,
        "min_repeat_models": args.min_repeat_models,
        "max_single_pair_penalty": args.max_single_pair_penalty,
        "baseline_rows": len(final_model_ids),
        "baseline_rmse": current_subset["rmse"],
        "panel_rmse": final_panel["rmse"],
        "baseline_mae": current_subset["mae"],
        "panel_mae": final_panel["mae"],
        "baseline_r2": current_subset["r2"],
        "panel_r2": final_panel["r2"],
    }

    baseline_objective = panel_objective(
        current_subset,
        pair_gap=args.pair_gap,
        rmse_weight=args.rmse_weight,
        mae_weight=args.mae_weight,
        pair_weight=args.pair_weight,
        anchor_weight=args.anchor_weight,
        tail_weight=args.tail_weight,
    )
    final_objective = panel_objective(
        final_panel,
        pair_gap=args.pair_gap,
        rmse_weight=args.rmse_weight,
        mae_weight=args.mae_weight,
        pair_weight=args.pair_weight,
        anchor_weight=args.anchor_weight,
        tail_weight=args.tail_weight,
    )
    metadata.update(
        {
            "baseline_objective": baseline_objective["objective"],
            "panel_objective": final_objective["objective"],
            "baseline_pair_penalty": baseline_objective["pair_penalty"],
            "panel_pair_penalty": final_objective["pair_penalty"],
            "baseline_anchor_penalty": baseline_objective["anchor_penalty"],
            "panel_anchor_penalty": final_objective["anchor_penalty"],
            "baseline_p90_abs_err": current_subset["p90_abs_err"],
            "panel_p90_abs_err": final_panel["p90_abs_err"],
        }
    )

    args.output_positions.write_text(json.dumps({"metadata": metadata, "positions": selected_positions}, indent=2))
    args.output_report.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "selected": [
                    {
                        "position_idx": candidate["position_idx"],
                        "bucket": candidate["bucket"],
                        "feature": feature_name,
                        "coverage": candidate["coverage"],
                        "opening_key": candidate["opening_key"],
                        "game_id": candidate["game_id"],
                        "selected_stats": candidate["selected_stats"],
                        "feature_stats": candidate["feature_stats"].get(feature_name),
                    }
                    for candidate, feature_name in selected
                ],
                "top_rankings": sorted(
                    rankings,
                    key=lambda item: (item["delta_rmse"], item["n_rows"], item["coverage"]),
                    reverse=True,
                )[:50],
            },
            indent=2,
        )
    )

    print()
    print(
        f"Current benchmark on subset: RMSE={current_subset['rmse']:.1f} "
        f"MAE={current_subset['mae']:.1f} R2={current_subset['r2']:.3f}"
    )
    print(
        f"Selected panel on subset: RMSE={final_panel['rmse']:.1f} "
        f"MAE={final_panel['mae']:.1f} R2={final_panel['r2']:.3f}"
    )
    print(f"Wrote {len(selected_positions)} positions to {args.output_positions}")
    print(f"Wrote report to {args.output_report}")


if __name__ == "__main__":
    main()
