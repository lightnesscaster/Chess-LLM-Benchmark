#!/usr/bin/env python3
"""Analyze position benchmark results with item-level Regan-lite scoring.

This script is intentionally offline/local: it joins a positions file, a
results file, and local ratings, then compares aggregate CPL formulas against
position-by-position scoring. The goal is to test whether the signal is lost in
the aggregate summary rather than in the positions themselves.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


ANCHOR_IDS = {"random-bot", "eubos", "maia-1900", "maia-1100", "survival-bot"}


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_results(path: Path, results_glob: str | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if path.exists():
        data = load_json(path)
        if isinstance(data, dict):
            merged.update(data)
    if results_glob:
        for file_path in sorted(glob.glob(results_glob)):
            data = load_json(Path(file_path))
            if isinstance(data, dict):
                merged.update(data)
    return merged


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


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y - pred)))


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    if float(np.std(x)) == 0 or float(np.std(y)) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


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
    ]
    for prefix in prefixes:
        if player.startswith(prefix):
            return prefix
    return player.split()[0].split("-")[0]


def standardize_train_test(
    train_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_x = np.nan_to_num(np.asarray(train_x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    test_x = np.nan_to_num(np.asarray(test_x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    mu = train_x.mean(axis=0)
    sd = train_x.std(axis=0)
    sd[sd < 1e-9] = 1.0
    train_z = np.clip((train_x - mu) / sd, -20.0, 20.0)
    test_z = np.clip((test_x - mu) / sd, -20.0, 20.0)
    return (
        np.nan_to_num(train_z, nan=0.0, posinf=20.0, neginf=-20.0),
        np.nan_to_num(test_z, nan=0.0, posinf=20.0, neginf=-20.0),
    )


def fit_ridge_predict(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alpha: float,
) -> np.ndarray:
    train_z, test_z = standardize_train_test(train_x, test_x)
    train_design = np.nan_to_num(
        np.column_stack([np.ones(len(train_z)), train_z]),
        nan=0.0,
        posinf=20.0,
        neginf=-20.0,
    )
    test_design = np.nan_to_num(
        np.column_stack([np.ones(len(test_z)), test_z]),
        nan=0.0,
        posinf=20.0,
        neginf=-20.0,
    )
    if alpha <= 0:
        coef = np.linalg.lstsq(train_design, train_y, rcond=None)[0]
        return test_design @ coef
    penalty = np.eye(train_design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    with np.errstate(all="ignore"):
        lhs = np.nan_to_num(
            train_design.T @ train_design + penalty,
            nan=0.0,
            posinf=1e9,
            neginf=-1e9,
        )
        rhs = np.nan_to_num(
            train_design.T @ train_y,
            nan=0.0,
            posinf=1e9,
            neginf=-1e9,
        )
    coef = np.linalg.solve(lhs, rhs)
    return test_design @ coef


def loo_ridge_predict(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    all_idx = np.arange(len(y))
    for i in all_idx:
        train = all_idx[all_idx != i]
        pred[[i]] = fit_ridge_predict(X[train], y[train], X[[i]], alpha)
    return pred


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


def linear_loo_from_score(score: np.ndarray, y: np.ndarray) -> np.ndarray:
    return loo_ridge_predict(score.reshape(-1, 1), y, alpha=0.0)


def linear_lofo_from_score(
    score: np.ndarray,
    y: np.ndarray,
    families: list[str],
) -> np.ndarray:
    return lofo_ridge_predict(score.reshape(-1, 1), y, families, alpha=0.0)


def train_weighted_item_score(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
) -> np.ndarray:
    train_z, test_z = standardize_train_test(train_x, test_x)
    weights = np.asarray(
        [pearson(train_z[:, i], train_y) for i in range(train_z.shape[1])],
        dtype=float,
    )
    weights[~np.isfinite(weights)] = 0.0
    denom = float(np.sum(np.abs(weights)))
    if denom < 1e-9:
        return np.zeros(len(test_x), dtype=float)
    with np.errstate(all="ignore"):
        score = test_z @ weights / denom
    return np.nan_to_num(score, nan=0.0, posinf=20.0, neginf=-20.0)


def loo_weighted_item_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    all_idx = np.arange(len(y))
    for i in all_idx:
        train = all_idx[all_idx != i]
        train_score = train_weighted_item_score(X[train], y[train], X[train])
        test_score = train_weighted_item_score(X[train], y[train], X[[i]])
        pred[[i]] = fit_ridge_predict(
            train_score.reshape(-1, 1),
            y[train],
            test_score.reshape(-1, 1),
            alpha=0.0,
        )
    return pred


def lofo_weighted_item_predict(
    X: np.ndarray,
    y: np.ndarray,
    families: list[str],
) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    family_arr = np.asarray(families)
    all_idx = np.arange(len(y))
    for family in sorted(set(families)):
        test = all_idx[family_arr == family]
        train = all_idx[family_arr != family]
        train_score = train_weighted_item_score(X[train], y[train], X[train])
        test_score = train_weighted_item_score(X[train], y[train], X[test])
        pred[test] = fit_ridge_predict(
            train_score.reshape(-1, 1),
            y[train],
            test_score.reshape(-1, 1),
            alpha=0.0,
        )
    return pred


def unsupervised_item_score(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd).mean(axis=1)


def summarize_prediction(name: str, y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    out = {"rmse": rmse(y, pred), "mae": mae(y, pred), "r2": r2(y, pred)}
    print(
        f"  {name:<34} RMSE={out['rmse']:>6.1f} "
        f"MAE={out['mae']:>6.1f} R2={out['r2']:>6.3f}"
    )
    return out


def collect_rows(
    *,
    positions: list[dict[str, Any]],
    results: dict[str, Any],
    ratings: dict[str, Any],
    max_rd: float,
    min_position_coverage: float,
    include_anchors: bool,
    extra_models: set[str],
    focus_families: set[str],
    focus_models: set[str],
) -> tuple[list[dict[str, Any]], list[int]]:
    family_prefixes = {value.lower() for value in focus_families}
    model_prefixes = {value.lower() for value in focus_models}

    def matches_focus(player_id: str) -> bool:
        if not family_prefixes and not model_prefixes:
            return True
        player_lower = player_id.lower()
        family_lower = model_family(player_id).lower()
        return any(family_lower.startswith(prefix) for prefix in family_prefixes) or any(
            player_lower.startswith(prefix) for prefix in model_prefixes
        )

    covered_by_pos: dict[int, int] = defaultdict(int)
    usable_results: dict[str, dict[int, dict[str, Any]]] = {}

    for player_id, model_data in results.items():
        if not matches_focus(player_id) and not (include_anchors and player_id in ANCHOR_IDS):
            continue
        by_idx: dict[int, dict[str, Any]] = {}
        for item in model_data.get("results", []):
            idx = item.get("position_idx")
            if isinstance(idx, int) and 0 <= idx < len(positions):
                by_idx[idx] = item
        if not by_idx:
            continue
        usable_results[player_id] = by_idx
        for idx in by_idx:
            covered_by_pos[idx] += 1

    candidate_models = set(usable_results) & set(ratings)
    reliable_models: list[str] = []
    for player_id in sorted(candidate_models):
        rating = ratings[player_id]
        games = int(rating.get("games_played") or 0)
        rd = rating.get("games_rd", rating.get("rating_deviation"))
        reliable = (
            player_id in ANCHOR_IDS
            or (games > 0 and rd is not None and float(rd) < max_rd)
            or player_id in extra_models
        )
        if not reliable:
            continue
        if not include_anchors and player_id in ANCHOR_IDS:
            continue
        reliable_models.append(player_id)

    min_models = max(3, math.ceil(len(reliable_models) * min_position_coverage))
    common_positions = [
        idx
        for idx in range(len(positions))
        if covered_by_pos.get(idx, 0) >= min_models
    ]

    rows: list[dict[str, Any]] = []
    for player_id in reliable_models:
        by_idx = usable_results[player_id]
        present = [idx for idx in common_positions if idx in by_idx]
        if len(present) < max(5, math.ceil(len(common_positions) * min_position_coverage)):
            continue
        rating = ratings[player_id]
        rd = rating.get("games_rd", rating.get("rating_deviation"))
        rows.append(
            {
                "model": player_id,
                "actual": float(rating["rating"]),
                "rd": float(rd),
                "games_played": int(rating.get("games_played") or 0),
                "family": model_family(player_id),
                "results_by_idx": by_idx,
            }
        )

    return rows, common_positions


def build_feature_matrices(
    rows: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    position_indices: list[int],
    cap_cpl: float,
) -> dict[str, np.ndarray]:
    n = len(rows)
    m = len(position_indices)
    loss = np.zeros((n, m), dtype=float)
    quality = np.zeros((n, m), dtype=float)
    best = np.zeros((n, m), dtype=float)
    legal = np.zeros((n, m), dtype=float)
    top3 = np.zeros((n, m), dtype=float)
    top5 = np.zeros((n, m), dtype=float)
    reciprocal_rank = np.zeros((n, m), dtype=float)
    good25 = np.zeros((n, m), dtype=float)
    good50 = np.zeros((n, m), dtype=float)
    good100 = np.zeros((n, m), dtype=float)
    good300 = np.zeros((n, m), dtype=float)
    bucket_names = sorted(
        {
            str(positions[idx].get("regan_bucket") or positions[idx].get("bucket") or positions[idx].get("type") or "unknown")
            for idx in position_indices
        }
    )
    bucket_to_col = {bucket: i for i, bucket in enumerate(bucket_names)}
    bucket_loss = np.zeros((n, len(bucket_names)), dtype=float)
    bucket_count = np.zeros((n, len(bucket_names)), dtype=float)

    for row_i, row in enumerate(rows):
        by_idx = row["results_by_idx"]
        for col_i, pos_idx in enumerate(position_indices):
            item = by_idx.get(pos_idx)
            if item is None:
                loss[row_i, col_i] = np.nan
                quality[row_i, col_i] = np.nan
                best[row_i, col_i] = np.nan
                legal[row_i, col_i] = np.nan
                top3[row_i, col_i] = np.nan
                top5[row_i, col_i] = np.nan
                reciprocal_rank[row_i, col_i] = np.nan
                good25[row_i, col_i] = np.nan
                good50[row_i, col_i] = np.nan
                good100[row_i, col_i] = np.nan
                good300[row_i, col_i] = np.nan
                continue
            cpl = float(item.get("cpl") or 0.0)
            capped = min(max(0.0, cpl), cap_cpl)
            transformed_loss = math.log1p(capped)
            move_legal = bool(item.get("is_legal", True))
            loss[row_i, col_i] = -transformed_loss
            quality[row_i, col_i] = 1.0 / (1.0 + transformed_loss / math.log1p(100.0))
            best[row_i, col_i] = 1.0 if item.get("is_best", False) else 0.0
            legal[row_i, col_i] = 1.0 if move_legal else 0.0
            multipv = positions[pos_idx].get("multipv") or []
            move_rank = None
            if multipv:
                model_move = str(item.get("model_move") or "")
                for rank, candidate in enumerate(multipv, start=1):
                    if candidate.get("move") == model_move:
                        move_rank = rank
                        break
                if move_rank is None:
                    move_rank = len(multipv) + 1
                top3[row_i, col_i] = 1.0 if move_rank <= 3 else 0.0
                top5[row_i, col_i] = 1.0 if move_rank <= 5 else 0.0
                reciprocal_rank[row_i, col_i] = 1.0 / move_rank
            else:
                top3[row_i, col_i] = np.nan
                top5[row_i, col_i] = np.nan
                reciprocal_rank[row_i, col_i] = np.nan
            good25[row_i, col_i] = 1.0 if move_legal and cpl <= 25 else 0.0
            good50[row_i, col_i] = 1.0 if move_legal and cpl <= 50 else 0.0
            good100[row_i, col_i] = 1.0 if move_legal and cpl <= 100 else 0.0
            good300[row_i, col_i] = 1.0 if move_legal and cpl <= 300 else 0.0

            bucket = str(positions[pos_idx].get("regan_bucket") or positions[pos_idx].get("bucket") or positions[pos_idx].get("type") or "unknown")
            bucket_col = bucket_to_col[bucket]
            bucket_loss[row_i, bucket_col] += transformed_loss
            bucket_count[row_i, bucket_col] += 1.0

        for matrix in (loss, quality, best, legal, top3, top5, reciprocal_rank, good25, good50, good100, good300):
            row_mean = np.nanmean(matrix[row_i])
            fill_value = float(row_mean) if np.isfinite(row_mean) else 0.0
            matrix[row_i, np.isnan(matrix[row_i])] = fill_value

        for bucket_col in range(len(bucket_names)):
            if bucket_count[row_i, bucket_col] > 0:
                bucket_loss[row_i, bucket_col] = -bucket_loss[row_i, bucket_col] / bucket_count[row_i, bucket_col]
            else:
                bucket_loss[row_i, bucket_col] = np.nan

    for bucket_col in range(len(bucket_names)):
        values = bucket_loss[:, bucket_col]
        fill_value = float(np.nanmean(values)) if np.isfinite(np.nanmean(values)) else 0.0
        values[np.isnan(values)] = fill_value
        bucket_loss[:, bucket_col] = values

    aggregate_cols: list[list[float]] = []
    for row in rows:
        items = [row["results_by_idx"][idx] for idx in position_indices if idx in row["results_by_idx"]]
        cpls = np.asarray([min(float(item.get("cpl") or 0.0), cap_cpl) for item in items], dtype=float)
        legal_arr = np.asarray([1.0 if item.get("is_legal", True) else 0.0 for item in items], dtype=float)
        best_arr = np.asarray([1.0 if item.get("is_best", False) else 0.0 for item in items], dtype=float)
        aggregate_cols.append(
            [
                math.log1p(float(cpls.mean())),
                float(np.median(np.log1p(cpls))),
                float(legal_arr.mean() * 100.0),
                float(best_arr.mean() * 100.0),
                float((cpls <= 25).mean() * 100.0),
                float((cpls <= 50).mean() * 100.0),
                float((cpls <= 100).mean() * 100.0),
                float(np.nanmean(top3[row_i]) * 100.0),
                float(np.nanmean(top5[row_i]) * 100.0),
                float(np.nanmean(reciprocal_rank[row_i]) * 100.0),
                survival_probability(float(legal_arr.mean() * 100.0)),
            ]
        )

    return {
        "loss_items": loss,
        "quality_items": quality,
        "best_items": best,
        "legal_items": legal,
        "top3_items": top3,
        "top5_items": top5,
        "reciprocal_rank_items": reciprocal_rank,
        "good25_items": good25,
        "good50_items": good50,
        "good100_items": good100,
        "good300_items": good300,
        "bucket_loss": bucket_loss,
        "aggregate": np.asarray(aggregate_cols, dtype=float),
    }


def logsumexp(values: list[float]) -> float:
    if not values:
        return 0.0
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def regan_likelihood_scores(
    rows: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    position_indices: list[int],
    cap_cpl: float,
) -> dict[str, np.ndarray]:
    """Average move-choice log likelihood from MultiPV deltas.

    This is a Regan-lite version of "players choose among plausible moves with
    probabilities shaped by engine-eval gaps." It uses the MultiPV list for the
    denominator and the model move's delta for the numerator. If the chosen move
    is outside MultiPV, its measured CPL is included as an extra low-probability
    option.
    """
    if not any(positions[idx].get("multipv") for idx in position_indices):
        return {}

    scores: dict[str, list[float]] = {}
    for scale_cp in (25.0, 50.0, 100.0, 200.0, 400.0):
        for consistency in (1.0, 1.5, 2.0):
            scores[f"regan_like_s{scale_cp:g}_c{consistency:g}"] = []

    for row in rows:
        by_idx = row["results_by_idx"]
        per_param_totals = {name: 0.0 for name in scores}
        used = 0
        for pos_idx in position_indices:
            position = positions[pos_idx]
            multipv = position.get("multipv") or []
            item = by_idx.get(pos_idx)
            if not multipv or item is None:
                continue

            model_move = str(item.get("model_move") or "")
            move_delta = None
            candidate_deltas: list[float] = []
            for candidate in multipv:
                delta = min(max(0.0, float(candidate.get("delta_cp") or 0.0)), cap_cpl)
                candidate_deltas.append(delta)
                if candidate.get("move") == model_move:
                    move_delta = delta

            if move_delta is None:
                move_delta = min(max(0.0, float(item.get("cpl") or cap_cpl)), cap_cpl)
                candidate_deltas.append(move_delta)
            if not item.get("is_legal", True):
                move_delta = cap_cpl
                candidate_deltas.append(move_delta)

            for scale_cp in (25.0, 50.0, 100.0, 200.0, 400.0):
                scale = math.log1p(scale_cp)
                for consistency in (1.0, 1.5, 2.0):
                    name = f"regan_like_s{scale_cp:g}_c{consistency:g}"

                    def move_logit(delta: float) -> float:
                        return -((math.log1p(delta) / scale) ** consistency)

                    numerator = move_logit(move_delta)
                    denominator = logsumexp([move_logit(delta) for delta in candidate_deltas])
                    per_param_totals[name] += numerator - denominator
            used += 1

        for name in scores:
            scores[name].append(per_param_totals[name] / used if used else -20.0)

    return {name: np.asarray(values, dtype=float) for name, values in scores.items()}


def current_formula_predictions(
    rows: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    position_indices: list[int],
) -> np.ndarray | None:
    equal_indices = [
        idx
        for idx in position_indices
        if (positions[idx].get("type") == "equal" or positions[idx].get("regan_bucket") == "equal")
    ]
    if len(equal_indices) < 10:
        return None

    preds: list[float] = []
    for row in rows:
        items = [row["results_by_idx"][idx] for idx in equal_indices if idx in row["results_by_idx"]]
        if len(items) < max(5, len(equal_indices) // 2):
            return None
        cpls = [float(item.get("cpl") or 0.0) for item in items]
        legal_pct = 100.0 * sum(1 for item in items if item.get("is_legal", True)) / len(items)
        best_pct = 100.0 * sum(1 for item in items if item.get("is_best", False)) / len(items)
        preds.append(current_equal_formula(sum(cpls) / len(cpls), best_pct, legal_pct))
    return np.asarray(preds, dtype=float)


def print_worst_errors(
    name: str,
    rows: list[dict[str, Any]],
    y: np.ndarray,
    pred: np.ndarray,
    limit: int,
) -> None:
    print()
    print(f"Worst {name} errors:")
    errors = []
    for i, row in enumerate(rows):
        errors.append((abs(float(y[i] - pred[i])), float(y[i] - pred[i]), row["model"], y[i], pred[i]))
    for abs_err, err, model, actual, predicted in sorted(errors, reverse=True)[:limit]:
        print(
            f"  {model:<42} actual={actual:>7.0f} pred={predicted:>7.0f} "
            f"err={err:>+7.0f} abs={abs_err:>6.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, default=Path("position_benchmark/positions.json"))
    parser.add_argument("--results", type=Path, default=Path("position_benchmark/results.json"))
    parser.add_argument("--results-glob", default=None)
    parser.add_argument("--ratings", type=Path, default=Path("data/ratings.json"))
    parser.add_argument("--max-rd", type=float, default=100.0)
    parser.add_argument("--include-anchors", action="store_true")
    parser.add_argument("--focus-family", action="append", default=[])
    parser.add_argument("--focus-model", action="append", default=[])
    parser.add_argument("--extra-model", action="append", default=[])
    parser.add_argument("--min-position-coverage", type=float, default=0.9)
    parser.add_argument("--cap-cpl", type=float, default=5000.0)
    parser.add_argument("--worst", type=int, default=12)
    args = parser.parse_args()

    positions_data = load_json(args.positions)
    positions = positions_data["positions"] if isinstance(positions_data, dict) else positions_data
    results = load_results(args.results, args.results_glob)
    ratings = load_json(args.ratings)

    rows, position_indices = collect_rows(
        positions=positions,
        results=results,
        ratings=ratings,
        max_rd=args.max_rd,
        min_position_coverage=args.min_position_coverage,
        include_anchors=args.include_anchors,
        extra_models=set(args.extra_model),
        focus_families=set(args.focus_family),
        focus_models=set(args.focus_model),
    )
    if len(rows) < 8:
        raise SystemExit(f"Only {len(rows)} usable rows after filters; relax --max-rd or coverage.")

    y = np.asarray([row["actual"] for row in rows], dtype=float)
    families = [row["family"] for row in rows]
    matrices = build_feature_matrices(rows, positions, position_indices, args.cap_cpl)
    likelihood_scores = regan_likelihood_scores(rows, positions, position_indices, args.cap_cpl)

    print(f"Rows: {len(rows)} models")
    print(f"Positions: {len(position_indices)} common of {len(positions)} total")
    print(f"Families: {len(set(families))}")
    print(f"Rating range: {y.min():.0f} to {y.max():.0f}")

    print()
    print("Direct predictions / LOO CV:")
    current_pred = current_formula_predictions(rows, positions, position_indices)
    if current_pred is not None:
        summarize_prediction("current equal formula", y, current_pred)

    best_name = ""
    best_pred: np.ndarray | None = None
    best_rmse = float("inf")

    def track(name: str, pred: np.ndarray) -> None:
        nonlocal best_name, best_pred, best_rmse
        stats = summarize_prediction(name, y, pred)
        if stats["rmse"] < best_rmse:
            best_name = name
            best_pred = pred
            best_rmse = stats["rmse"]

    aggregate = matrices["aggregate"]
    for alpha in (0.0, 1.0, 10.0, 100.0):
        pred = loo_ridge_predict(aggregate, y, alpha)
        track(f"aggregate features alpha={alpha:g}", pred)

    bucket_loss = matrices["bucket_loss"]
    if bucket_loss.shape[1] > 1:
        for alpha in (0.0, 1.0, 10.0, 100.0):
            pred = loo_ridge_predict(np.column_stack([aggregate, bucket_loss]), y, alpha)
            track(f"aggregate + buckets a={alpha:g}", pred)

    for name, score in likelihood_scores.items():
        track(name, linear_loo_from_score(score, y))

    for matrix_name in (
        "loss_items",
        "quality_items",
        "best_items",
        "top3_items",
        "top5_items",
        "reciprocal_rank_items",
        "good25_items",
        "good50_items",
        "good100_items",
        "good300_items",
    ):
        X = matrices[matrix_name]
        score = unsupervised_item_score(X)
        track(f"{matrix_name} item-z", linear_loo_from_score(score, y))
        track(f"{matrix_name} weighted", loo_weighted_item_predict(X, y))
        for alpha in (10.0, 100.0, 1000.0):
            track(f"{matrix_name} ridge a={alpha:g}", loo_ridge_predict(X, y, alpha))

    print()
    print("Leave-one-family-out CV:")
    if current_pred is not None:
        summarize_prediction("current equal formula", y, current_pred)
    for name, X in (
        ("aggregate features", aggregate),
        ("aggregate + buckets", np.column_stack([aggregate, bucket_loss])),
        ("loss_items", matrices["loss_items"]),
        ("quality_items", matrices["quality_items"]),
        ("top3_items", matrices["top3_items"]),
        ("top5_items", matrices["top5_items"]),
        ("reciprocal_rank_items", matrices["reciprocal_rank_items"]),
        ("good50_items", matrices["good50_items"]),
        ("good100_items", matrices["good100_items"]),
    ):
        for alpha in (10.0, 100.0, 1000.0):
            pred = lofo_ridge_predict(X, y, families, alpha)
            summarize_prediction(f"{name} ridge a={alpha:g}", y, pred)
        if name.endswith("items"):
            pred = lofo_weighted_item_predict(X, y, families)
            summarize_prediction(f"{name} weighted", y, pred)

    for name, score in likelihood_scores.items():
        pred = linear_lofo_from_score(score, y, families)
        summarize_prediction(name, y, pred)

    if best_pred is not None:
        print_worst_errors(best_name, rows, y, best_pred, args.worst)


if __name__ == "__main__":
    main()
