#!/usr/bin/env python3
"""Evaluate a fixed selected panel against the current benchmark baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binom


ANCHOR_IDS = {"random-bot", "eubos", "maia-1900", "maia-1100", "survival-bot"}


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--ratings", type=Path, default=Path("data/ratings.json"))
    parser.add_argument("--benchmark-results", type=Path, default=Path("position_benchmark/results.json"))
    parser.add_argument("--max-rd", type=float, default=100.0)
    parser.add_argument("--include-anchors", action="store_true")
    parser.add_argument("--extra-model", action="append", default=[])
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    positions_data = load_json(args.positions)
    positions = positions_data["positions"] if isinstance(positions_data, dict) else positions_data
    panel_results = load_json(args.results)
    ratings = load_json(args.ratings)
    benchmark_results = load_json(args.benchmark_results)

    rows: list[dict[str, Any]] = []
    position_features = {
        idx: str(position.get("selection_feature") or "reciprocal_rank")
        for idx, position in enumerate(positions)
    }
    for player_id, model_data in panel_results.items():
        rating = ratings.get(player_id)
        benchmark = benchmark_results.get(player_id)
        if not rating or not benchmark:
            continue
        prediction = current_position_prediction((benchmark.get("summary") or {}))
        if prediction is None:
            continue
        games = int(rating.get("games_played") or 0)
        rd = rating.get("games_rd", rating.get("rating_deviation"))
        reliable = (
            player_id in ANCHOR_IDS
            or (games > 0 and rd is not None and float(rd) < args.max_rd)
            or player_id in set(args.extra_model)
        )
        if not reliable:
            continue
        if not args.include_anchors and player_id in ANCHOR_IDS:
            continue

        by_idx = {
            int(item["position_idx"]): item
            for item in model_data.get("results", [])
            if isinstance(item.get("position_idx"), int)
        }
        feature_values: list[float] = []
        ok = True
        for idx in range(len(positions)):
            item = by_idx.get(idx)
            if item is None:
                ok = False
                break
            feature = position_features[idx]
            if feature == "top3":
                value = 1.0 if item.get("top3") else 0.0
            elif feature == "top5":
                value = 1.0 if item.get("top5") else 0.0
            elif feature == "best":
                value = 1.0 if item.get("is_best") else 0.0
            else:
                value = float(item.get("reciprocal_rank") or 0.0)
            feature_values.append(value)
        if not ok:
            continue
        rows.append(
            {
                "model": player_id,
                "actual": float(rating["rating"]),
                "current_pred": float(prediction),
                "family": model_family(player_id),
                "panel_features": feature_values,
            }
        )

    if len(rows) < 5:
        raise SystemExit(f"Only {len(rows)} usable rows.")

    y = np.asarray([row["actual"] for row in rows], dtype=float)
    current = np.asarray([row["current_pred"] for row in rows], dtype=float)
    families = [str(row["family"]) for row in rows]
    panel_only_X = np.asarray([row["panel_features"] for row in rows], dtype=float)
    current_plus_panel_X = np.asarray(
        [[row["current_pred"], *row["panel_features"]] for row in rows],
        dtype=float,
    )

    print(f"Rows: {len(rows)}")
    print(f"Families: {sorted(set(families))}")
    print()
    print(
        f"Current benchmark baseline: RMSE={rmse(y, current):.1f} "
        f"MAE={mae(y, current):.1f} R2={r2(y, current):.3f}"
    )

    for alpha in (0.0, 0.1, 1.0, 10.0):
        pred = lofo_ridge_predict(panel_only_X, y, families, alpha)
        print(
            f"Panel only LOFO alpha={alpha:g}: RMSE={rmse(y, pred):.1f} "
            f"MAE={mae(y, pred):.1f} R2={r2(y, pred):.3f}"
        )

    print()
    for alpha in (0.0, 0.1, 1.0, 10.0):
        pred = lofo_ridge_predict(current_plus_panel_X, y, families, alpha)
        print(
            f"Current + panel LOFO alpha={alpha:g}: RMSE={rmse(y, pred):.1f} "
            f"MAE={mae(y, pred):.1f} R2={r2(y, pred):.3f}"
        )
        if alpha == args.alpha:
            print()
            print("Predictions at requested alpha:")
            for row, value in sorted(zip(rows, pred), key=lambda pair: pair[0]["actual"], reverse=True):
                print(
                    f"  {row['model']:<38} actual={row['actual']:>7.0f} "
                    f"current={row['current_pred']:>7.0f} panel={value:>7.0f}"
                )


if __name__ == "__main__":
    main()
