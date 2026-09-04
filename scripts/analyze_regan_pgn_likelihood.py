#!/usr/bin/env python3
"""Evaluate existing benchmark PGNs with a Regan-style move-choice model.

This script uses only existing game PGNs plus Stockfish. It does not call any
model APIs. For each played move it builds a MultiPV move-quality distribution
at depth >= 16, then aggregates Regan-lite log-likelihood features per model
and compares them to actual ratings and to the current position-benchmark
predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
import numpy as np
from scipy.stats import binom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firebase_client import (
    BENCHMARK_RESULTS_COLLECTION,
    GAMES_COLLECTION,
    RATINGS_COLLECTION,
    RESULTS_COLLECTION,
    get_firestore_client,
)


ANCHOR_IDS = {"random-bot", "eubos", "maia-1900", "maia-1100", "survival-bot"}

SCALES_CP = (25.0, 50.0, 100.0, 200.0)
CONSISTENCIES = (1.0, 1.5, 2.0)


def eval_to_cp(info: chess.engine.InfoDict, perspective: chess.Color) -> float:
    score = info.get("score")
    if score is None:
        return 0.0
    pov = score.pov(perspective)
    if pov.is_mate():
        mate = pov.mate()
        if mate is None:
            return 0.0
        if mate > 0:
            return 10000.0 - 10.0 * mate
        return -10000.0 - 10.0 * mate
    cp = pov.score()
    return float(cp if cp is not None else 0.0)


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


def current_position_prediction(summary: dict[str, Any]) -> float | None:
    equal = summary.get("equal") or summary
    try:
        equal_cpl = float(equal["avg_cpl"])
        best_pct = float(equal["best_pct"])
        legal_pct = float(equal["legal_pct"])
    except (KeyError, TypeError, ValueError):
        return None
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


def pearson(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return float("nan")
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if float(np.std(x)) == 0 or float(np.std(y)) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ranks(values: list[float]) -> list[float]:
    ordered = sorted((v, i) for i, v in enumerate(values))
    out = [0.0] * len(values)
    j = 0
    while j < len(ordered):
        k = j
        while k + 1 < len(ordered) and ordered[k + 1][0] == ordered[j][0]:
            k += 1
        avg = (j + k) / 2.0 + 1.0
        for idx in range(j, k + 1):
            out[ordered[idx][1]] = avg
        j = k + 1
    return out


def spearman(a: list[float], b: list[float]) -> float:
    return pearson(ranks(a), ranks(b))


def loo_linear_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    pred = np.empty(len(y), dtype=float)
    idx = np.arange(len(y))
    for i in idx:
        train = idx[idx != i]
        mu = X[train].mean(axis=0)
        sd = X[train].std(axis=0)
        sd[sd < 1e-9] = 1.0
        train_x = np.column_stack([np.ones(len(train)), (X[train] - mu) / sd])
        coef = np.linalg.lstsq(train_x, y[train], rcond=None)[0]
        test_x = np.column_stack([np.ones(1), (X[[i]] - mu) / sd])
        pred[i] = float((test_x @ coef)[0])
    return pred


def fit_linear_raw(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(design, y, rcond=None)[0]


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


@dataclass
class MeanStat:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        if math.isfinite(value):
            self.total += value
            self.count += 1

    def mean(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count


class PlayerAgg:
    def __init__(self) -> None:
        self.stats: dict[str, MeanStat] = defaultdict(MeanStat)
        self.games: set[str] = set()

    def add_value(self, name: str, value: float) -> None:
        self.stats[name].add(value)

    def add_rate_event(self, name: str, hit: bool) -> None:
        self.stats[name].add(1.0 if hit else 0.0)

    def to_features(self) -> dict[str, float]:
        features: dict[str, float] = {"games_analyzed": float(len(self.games))}
        for key, stat in self.stats.items():
            mean = stat.mean()
            if mean is not None:
                features[key] = mean
                features[f"{key}_n"] = float(stat.count)
        return features


def phase_name(fullmove: int, piece_count: int) -> str:
    if piece_count <= 10 or fullmove >= 30:
        return "endgame"
    if fullmove <= 10:
        return "opening"
    return "middlegame"


def pgn_hash(pgn: str) -> str:
    return hashlib.sha256(pgn.encode("utf-8")).hexdigest()


def logsumexp(values: list[float]) -> float:
    if not values:
        return 0.0
    max_value = max(values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def regan_logprob(
    *,
    top_deltas: list[float],
    actual_delta: float,
    scale_cp: float,
    consistency: float,
) -> float:
    scale = math.log1p(scale_cp)

    def move_logit(delta: float) -> float:
        return -((math.log1p(max(0.0, delta)) / scale) ** consistency)

    deltas = [max(0.0, delta) for delta in top_deltas]
    if not any(abs(delta - actual_delta) < 1e-6 for delta in deltas):
        deltas.append(max(0.0, actual_delta))
    logits = [move_logit(delta) for delta in deltas]
    return move_logit(actual_delta) - logsumexp(logits)


def analyze_pgn(
    *,
    game_id: str,
    pgn: str,
    result: dict[str, Any],
    engine: chess.engine.SimpleEngine,
    depth: int,
    multipv: int,
    max_plies: int | None,
    cache_path: Path,
) -> dict[str, Any] | None:
    digest = pgn_hash(pgn)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if (
                cached.get("pgn_sha256") == digest
                and cached.get("depth") == depth
                and cached.get("multipv") == multipv
                and cached.get("max_plies") == max_plies
            ):
                return cached
        except json.JSONDecodeError:
            pass

    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return None

    white_id = result.get("white_id") or game.headers.get("White", "unknown")
    black_id = result.get("black_id") or game.headers.get("Black", "unknown")
    nodes = list(game.mainline())
    if max_plies is not None:
        nodes = nodes[:max_plies]

    board = game.board()
    position_evals: list[dict[str, Any]] = []
    move_meta: list[dict[str, Any]] = []
    side_move_no = {chess.WHITE: 0, chess.BLACK: 0}

    for idx in range(len(nodes) + 1):
        legal_count = board.legal_moves.count()
        mpv = max(1, min(multipv, legal_count))
        info_list = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=mpv,
        )
        if isinstance(info_list, dict):
            info_list = [info_list]
        best_eval = eval_to_cp(info_list[0], board.turn) if info_list else 0.0
        top_moves: list[dict[str, Any]] = []
        for info in info_list:
            pv = info.get("pv") or []
            if not pv:
                continue
            move = pv[0]
            if move not in board.legal_moves:
                continue
            move_eval = eval_to_cp(info, board.turn)
            try:
                san = board.san(move)
            except ValueError:
                san = move.uci()
            top_moves.append(
                {
                    "move": move.uci(),
                    "san": san,
                    "eval": move_eval,
                    "delta_cp": max(0.0, best_eval - move_eval),
                }
            )

        position_evals.append(
            {
                "eval_side_to_move": best_eval,
                "top_moves": top_moves,
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "piece_count": len(board.piece_map()),
                "legal_count": legal_count,
                "in_check": board.is_check(),
            }
        )

        if idx == len(nodes):
            break

        move = nodes[idx].move
        side = board.turn
        side_move_no[side] += 1
        try:
            san = board.san(move)
        except ValueError:
            san = ""
        move_meta.append(
            {
                "ply": idx + 1,
                "fullmove": board.fullmove_number,
                "side": "white" if side == chess.WHITE else "black",
                "player_id": white_id if side == chess.WHITE else black_id,
                "player_move_no": side_move_no[side],
                "played_move": move.uci(),
                "played_san": san,
            }
        )
        board.push(move)

    moves: list[dict[str, Any]] = []
    for idx, meta in enumerate(move_meta):
        before = position_evals[idx]
        after = position_evals[idx + 1]
        best_eval = float(before["eval_side_to_move"])
        actual_eval = -float(after["eval_side_to_move"])
        actual_cpl = max(0.0, best_eval - actual_eval)
        top_moves = before["top_moves"]
        top_deltas = [float(item["delta_cp"]) for item in top_moves]
        move_rank = None
        for rank, item in enumerate(top_moves, start=1):
            if item["move"] == meta["played_move"]:
                move_rank = rank
                break

        moves.append(
            {
                **meta,
                "phase": phase_name(int(meta["fullmove"]), int(before["piece_count"])),
                "piece_count": int(before["piece_count"]),
                "legal_count": int(before["legal_count"]),
                "in_check": bool(before["in_check"]),
                "eval_before": best_eval,
                "eval_after": actual_eval,
                "actual_cpl": actual_cpl,
                "best_move": top_moves[0]["move"] if top_moves else "",
                "is_best": move_rank == 1,
                "move_rank": move_rank or len(top_moves) + 1,
                "top_moves": top_moves,
                "top_deltas": top_deltas,
                "actual_in_topk": move_rank is not None,
            }
        )

    analyzed = {
        "game_id": game_id,
        "pgn_sha256": digest,
        "depth": depth,
        "multipv": multipv,
        "max_plies": max_plies,
        "white_id": white_id,
        "black_id": black_id,
        "moves": moves,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(analyzed))
    return analyzed


def fetch_firestore_data() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    db = get_firestore_client()
    ratings = {doc.id: doc.to_dict() for doc in db.collection(RATINGS_COLLECTION).stream()}
    benchmark = {
        doc.id: doc.to_dict()
        for doc in db.collection(BENCHMARK_RESULTS_COLLECTION).stream()
    }
    results = {doc.id: doc.to_dict() for doc in db.collection(RESULTS_COLLECTION).stream()}
    return ratings, benchmark, results


def fetch_pgn(game_id: str) -> str | None:
    db = get_firestore_client()
    doc = db.collection(GAMES_COLLECTION).document(game_id).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    return data.get("pgn")


def build_model_rows(
    ratings: dict[str, Any],
    benchmark: dict[str, Any],
    include_anchors: bool,
    extra_models: set[str],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for player_id, rating in ratings.items():
        if player_id not in benchmark:
            continue
        summary = benchmark[player_id].get("summary") or {}
        prediction = current_position_prediction(summary)
        if prediction is None:
            continue
        games = int(rating.get("games_played") or 0)
        rd = rating.get("games_rd", rating.get("rating_deviation"))
        reliable = player_id in ANCHOR_IDS or (games > 0 and rd is not None and rd < 100)
        if not reliable and player_id not in extra_models:
            continue
        if not include_anchors and player_id in ANCHOR_IDS:
            continue
        actual = float(rating["rating"])
        rows[player_id] = {
            "model": player_id,
            "actual": actual,
            "current_pred": float(prediction),
            "residual": actual - float(prediction),
            "rd": float(rd),
            "games_played": games,
            "family": model_family(player_id),
        }
    return rows


def filter_model_rows(
    rows: dict[str, dict[str, Any]],
    *,
    focus_families: set[str],
    focus_models: set[str],
    keep_anchors: bool,
) -> dict[str, dict[str, Any]]:
    if not focus_families and not focus_models:
        return rows

    family_prefixes = {value.lower() for value in focus_families}
    model_prefixes = {value.lower() for value in focus_models}
    filtered: dict[str, dict[str, Any]] = {}
    for player_id, row in rows.items():
        player_lower = player_id.lower()
        family_lower = str(row["family"]).lower()
        matches = any(
            family_lower.startswith(prefix) for prefix in family_prefixes
        ) or any(player_lower.startswith(prefix) for prefix in model_prefixes)
        if matches or (keep_anchors and player_id in ANCHOR_IDS):
            filtered[player_id] = row
    return filtered


def select_games(
    results: dict[str, Any],
    target_ids: set[str],
    max_games_per_player: int | None,
    seed: int,
) -> set[str]:
    per_player: dict[str, list[str]] = defaultdict(list)
    for game_id, result in results.items():
        for side in ("white", "black"):
            player_id = result.get(f"{side}_id")
            if player_id in target_ids:
                per_player[player_id].append(game_id)

    selected: set[str] = set()
    rng = random.Random(seed)
    for player_id, game_ids in per_player.items():
        unique_ids = sorted(set(game_ids))
        rng.shuffle(unique_ids)
        if max_games_per_player is not None:
            unique_ids = unique_ids[:max_games_per_player]
        selected.update(unique_ids)
    return selected


def add_move_stats(aggs: dict[str, PlayerAgg], analyzed: dict[str, Any]) -> None:
    for move in analyzed.get("moves", []):
        player_id = move["player_id"]
        if player_id not in aggs:
            continue
        agg = aggs[player_id]

        phase = str(move["phase"])
        cpl = float(move["actual_cpl"])
        eval_before = float(move["eval_before"])
        legal_count = int(move["legal_count"])
        buckets = ["all", phase]
        if int(move["fullmove"]) <= 15:
            buckets.append("early15")
        if abs(eval_before) <= 150:
            buckets.append("quiet_equal")
        if -150 <= eval_before <= 500:
            buckets.append("playable")
        if 150 <= eval_before <= 700:
            buckets.append("conversion")
        if eval_before <= -150:
            buckets.append("defense")
        if legal_count <= 8 or bool(move["in_check"]) or int(move["piece_count"]) <= 8:
            buckets.append("legal_stress")

        for bucket in buckets:
            agg.add_value(f"{bucket}_cpl", cpl)
            agg.add_rate_event(f"{bucket}_best_rate", bool(move["is_best"]))
            agg.add_value(f"{bucket}_move_rank", float(move["move_rank"]))
            agg.add_rate_event(f"{bucket}_topk_rate", bool(move["actual_in_topk"]))
            agg.add_rate_event(f"{bucket}_blunder100", cpl >= 100)
            agg.add_rate_event(f"{bucket}_blunder300", cpl >= 300)
            for scale_cp in SCALES_CP:
                for consistency in CONSISTENCIES:
                    consistency_name = str(consistency).replace(".", "_")
                    feature = f"{bucket}_regan_s{int(scale_cp)}_c{consistency_name}"
                    agg.add_value(
                        feature,
                        regan_logprob(
                            top_deltas=[float(delta) for delta in move["top_deltas"]],
                            actual_delta=cpl,
                            scale_cp=scale_cp,
                            consistency=consistency,
                        ),
                    )


def summarize_features(rows: list[dict[str, Any]], min_feature_n: int) -> None:
    y = np.asarray([row["actual"] for row in rows], dtype=float)
    current = np.asarray([row["current_pred"] for row in rows], dtype=float)

    print()
    print(f"Rows with Regan PGN features: {len(rows)}")
    print(
        "Current static formula: "
        f"RMSE={rmse(y, current):.1f} MAE={mae(y, current):.1f} R2={r2(y, current):.3f}"
    )

    excluded_suffixes = ("_n",)
    feature_names = sorted(
        key
        for key in rows[0]
        if key
        not in {
            "model",
            "actual",
            "current_pred",
            "residual",
            "rd",
            "games_played",
            "family",
        }
        and not key.endswith(excluded_suffixes)
    )

    feature_stats: list[dict[str, Any]] = []
    for feature in feature_names:
        values: list[float] = []
        res: list[float] = []
        ratings: list[float] = []
        current_values: list[float] = []
        for row in rows:
            value = row.get(feature)
            n_value = row.get(f"{feature}_n", min_feature_n)
            if value is None or not math.isfinite(float(value)) or n_value < min_feature_n:
                continue
            values.append(float(value))
            res.append(float(row["residual"]))
            ratings.append(float(row["actual"]))
            current_values.append(float(row["current_pred"]))
        if len(values) < max(10, min_feature_n):
            continue
        feature_stats.append(
            {
                "feature": feature,
                "n": len(values),
                "corr_resid": pearson(values, res),
                "spear_resid": spearman(values, res),
                "corr_rating": pearson(values, ratings),
                "corr_current": pearson(values, current_values),
            }
        )

    print()
    print("Top residual correlations:")
    for item in sorted(
        feature_stats,
        key=lambda x: abs(x["corr_resid"]) if math.isfinite(x["corr_resid"]) else -1,
        reverse=True,
    )[:25]:
        print(
            f"  {item['feature']:<38} n={item['n']:>2} "
            f"pearson={item['corr_resid']:+.3f} spearman={item['spear_resid']:+.3f} "
            f"rating={item['corr_rating']:+.3f} current={item['corr_current']:+.3f}"
        )

    print()
    print("Single-feature LOO residual checks: actual ~ current_pred + feature")
    loo_items: list[tuple[float, float, float, str, int]] = []
    base_rmse = rmse(y, current)
    for item in feature_stats:
        if item["n"] != len(rows):
            continue
        feature = item["feature"]
        X = np.asarray([[row["current_pred"], row[feature]] for row in rows], dtype=float)
        pred = loo_linear_predict(X, y)
        model_rmse = rmse(y, pred)
        loo_items.append((base_rmse - model_rmse, model_rmse, mae(y, pred), feature, item["n"]))
    for delta, model_rmse, model_mae, feature, n in sorted(loo_items, reverse=True)[:20]:
        print(
            f"  {feature:<38} n={n:>2} "
            f"delta_rmse={delta:+.1f} RMSE={model_rmse:.1f} MAE={model_mae:.1f}"
        )

    print()
    print("Subset LOO checks: actual ~ current_pred + feature")
    subset_items: list[tuple[float, float, float, float, str, int]] = []
    for item in feature_stats:
        feature = item["feature"]
        used_rows = [
            row
            for row in rows
            if row.get(feature) is not None
            and math.isfinite(float(row.get(feature)))
            and row.get(f"{feature}_n", min_feature_n) >= min_feature_n
        ]
        if len(used_rows) < max(10, min_feature_n):
            continue
        yy = np.asarray([row["actual"] for row in used_rows], dtype=float)
        base = np.asarray([row["current_pred"] for row in used_rows], dtype=float)
        X = np.asarray(
            [[row["current_pred"], row[feature]] for row in used_rows],
            dtype=float,
        )
        pred = loo_linear_predict(X, yy)
        base_subset_rmse = rmse(yy, base)
        model_rmse = rmse(yy, pred)
        subset_items.append(
            (
                base_subset_rmse - model_rmse,
                base_subset_rmse,
                model_rmse,
                mae(yy, pred),
                feature,
                len(used_rows),
            )
        )
    for delta, base_subset_rmse, model_rmse, model_mae, feature, n in sorted(
        subset_items,
        reverse=True,
    )[:20]:
        print(
            f"  {feature:<38} n={n:>2} "
            f"delta_rmse={delta:+.1f} base_RMSE={base_subset_rmse:.1f} "
            f"RMSE={model_rmse:.1f} MAE={model_mae:.1f}"
        )

    def feature_group(feature: str) -> str:
        if "_regan_" in feature:
            return feature.split("_regan_")[0] + "_regan"
        return feature

    combo_min_n = max(10, min_feature_n)
    best_by_group: dict[str, dict[str, Any]] = {}
    for item in sorted(
        feature_stats,
        key=lambda x: (abs(x["corr_resid"]), x["n"]),
        reverse=True,
    ):
        if item["n"] < combo_min_n:
            continue
        group = feature_group(item["feature"])
        best_by_group.setdefault(group, item)

    candidate_features = [
        item["feature"]
        for item in sorted(
            best_by_group.values(),
            key=lambda x: (abs(x["corr_resid"]), x["n"]),
            reverse=True,
        )[:12]
    ]

    def combo_rows(features: tuple[str, ...]) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if all(
                row.get(feature) is not None
                and math.isfinite(float(row.get(feature)))
                and row.get(f"{feature}_n", min_feature_n) >= min_feature_n
                for feature in features
            )
        ]

    def combo_eval(
        *,
        features: tuple[str, ...],
        include_current: bool,
    ) -> dict[str, Any] | None:
        used_rows = combo_rows(features)
        if len(used_rows) < combo_min_n:
            return None
        yy = np.asarray([row["actual"] for row in used_rows], dtype=float)
        base = np.asarray([row["current_pred"] for row in used_rows], dtype=float)
        columns: list[list[float]] = []
        if include_current:
            columns.append([row["current_pred"] for row in used_rows])
        columns.extend([[row[feature] for row in used_rows] for feature in features])
        X = np.asarray(np.column_stack(columns), dtype=float)
        pred = loo_linear_predict(X, yy)
        return {
            "features": features,
            "include_current": include_current,
            "n": len(used_rows),
            "base_rmse": rmse(yy, base),
            "rmse": rmse(yy, pred),
            "mae": mae(yy, pred),
            "r2": r2(yy, pred),
            "pred": pred,
            "rows": used_rows,
        }

    combo_results: list[dict[str, Any]] = []
    for size in (1, 2, 3):
        for features in itertools.combinations(candidate_features, size):
            for include_current in (True, False):
                result = combo_eval(features=features, include_current=include_current)
                if result is not None:
                    combo_results.append(result)

    print()
    print("Top combo LOO checks:")
    for item in sorted(
        combo_results,
        key=lambda x: (x["base_rmse"] - x["rmse"], x["n"]),
        reverse=True,
    )[:20]:
        prefix = "current+" if item["include_current"] else "pgn-only"
        features = ", ".join(item["features"])
        print(
            f"  {prefix:<8} n={item['n']:>2} "
            f"delta_rmse={item['base_rmse'] - item['rmse']:+.1f} "
            f"RMSE={item['rmse']:.1f} MAE={item['mae']:.1f} R2={item['r2']:.3f} "
            f"features=[{features}]"
        )

    best_current_combo = next(
        (
            item
            for item in sorted(
                combo_results,
                key=lambda x: (x["base_rmse"] - x["rmse"], x["n"]),
                reverse=True,
            )
            if item["include_current"]
        ),
        None,
    )
    if best_current_combo is not None:
        full_columns: list[list[float]] = [[row["current_pred"] for row in best_current_combo["rows"]]]
        full_columns.extend(
            [[row[feature] for row in best_current_combo["rows"]] for feature in best_current_combo["features"]]
        )
        full_X = np.asarray(np.column_stack(full_columns), dtype=float)
        full_y = np.asarray([row["actual"] for row in best_current_combo["rows"]], dtype=float)
        full_coef = fit_linear_raw(full_X, full_y)
        pred_by_model = {
            row["model"]: float(pred)
            for row, pred in zip(best_current_combo["rows"], best_current_combo["pred"])
        }
        print()
        print(
            "Best current+PGN combo: "
            + ", ".join(best_current_combo["features"])
        )
        print(
            f"  subset_n={best_current_combo['n']} "
            f"base_RMSE={best_current_combo['base_rmse']:.1f} "
            f"RMSE={best_current_combo['rmse']:.1f} "
            f"MAE={best_current_combo['mae']:.1f} "
            f"R2={best_current_combo['r2']:.3f}"
        )
        formula_terms = [f"{full_coef[0]:.3f}", f"({full_coef[1]:+.6f} * current_pred)"]
        for idx, feature in enumerate(best_current_combo["features"], start=2):
            formula_terms.append(f"({full_coef[idx]:+.6f} * {feature})")
        print("  fitted_formula:")
        print(f"    rating = {' '.join(formula_terms)}")
        interesting = [
            "gemini-3.1-pro-preview (high)",
            "gemini-3.1-pro-preview (medium)",
            "gemini-3-pro-preview (high)",
            "gemini-3-flash-preview (medium)",
            "grok-4.1-fast",
            "gpt-5.1 (high)",
            "gpt-5.2 (high)",
        ]
        print("  LOO predictions:")
        by_model = {row["model"]: row for row in rows}
        for model in interesting:
            row = by_model.get(model)
            if row is None:
                continue
            pred_value = pred_by_model.get(model)
            if pred_value is None:
                continue
            print(
                f"    {model:<38} actual={row['actual']:>6.0f} "
                f"current={row['current_pred']:>6.0f} combo={pred_value:>6.0f}"
            )

    best_current_single = next(
        (
            item
            for item in sorted(
                combo_results,
                key=lambda x: (x["base_rmse"] - x["rmse"], x["n"]),
                reverse=True,
            )
            if item["include_current"] and len(item["features"]) == 1
        ),
        None,
    )
    if best_current_single is not None:
        full_X = np.asarray(
            [
                [row["current_pred"], row[best_current_single["features"][0]]]
                for row in best_current_single["rows"]
            ],
            dtype=float,
        )
        full_y = np.asarray([row["actual"] for row in best_current_single["rows"]], dtype=float)
        full_coef = fit_linear_raw(full_X, full_y)
        feature = best_current_single["features"][0]
        print()
        print(f"Best simple current+PGN combo: {feature}")
        print(
            f"  subset_n={best_current_single['n']} "
            f"base_RMSE={best_current_single['base_rmse']:.1f} "
            f"RMSE={best_current_single['rmse']:.1f} "
            f"MAE={best_current_single['mae']:.1f} "
            f"R2={best_current_single['r2']:.3f}"
        )
        print("  fitted_formula:")
        print(
            "    rating = "
            f"{full_coef[0]:.3f} "
            f"({full_coef[1]:+.6f} * current_pred) "
            f"({full_coef[2]:+.6f} * {feature})"
        )


def print_interesting_rows(rows: list[dict[str, Any]]) -> None:
    interesting = [
        "gemini-3.1-pro-preview (high)",
        "gemini-3.1-pro-preview (medium)",
        "gemini-3-pro-preview (high)",
        "gemini-3-pro-preview (medium)",
        "gemini-3-flash-preview (medium)",
        "grok-4.1-fast",
        "grok-4-fast",
        "gpt-5.1 (high)",
        "gpt-5.2 (high)",
        "gpt-oss-120b (high)",
    ]
    by_model = {row["model"]: row for row in rows}
    print()
    print("High-interest rows:")
    for model in interesting:
        row = by_model.get(model)
        if row is None:
            print(f"  {model:<40} missing from analyzed reliable rows")
            continue
        print(
            f"  {model:<40} actual={row['actual']:>6.0f} pred={row['current_pred']:>6.0f} "
            f"resid={row['residual']:>+6.0f} games={row.get('games_analyzed', 0):>3.0f} "
            f"openLL={row.get('opening_regan_s100_c1_5', float('nan')):>7.3f} "
            f"allLL={row.get('all_regan_s100_c1_5', float('nan')):>7.3f} "
            f"openCPL={row.get('opening_cpl', float('nan')):>7.1f} "
            f"quietLL={row.get('quiet_equal_regan_s100_c1_5', float('nan')):>7.3f} "
            f"convLL={row.get('conversion_regan_s100_c1_5', float('nan')):>7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish-path", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--max-plies", type=int, default=40)
    parser.add_argument("--max-games-per-player", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-anchors", action="store_true")
    parser.add_argument(
        "--focus-family",
        action="append",
        default=[],
        help="Restrict analysis to model families with this prefix (for example gemini-3.1).",
    )
    parser.add_argument(
        "--focus-model",
        action="append",
        default=[],
        help="Restrict analysis to model IDs with this prefix.",
    )
    parser.add_argument(
        "--extra-model",
        action="append",
        default=[],
        help="Also include this model even if it fails the reliability cutoff.",
    )
    parser.add_argument("--min-feature-n", type=int, default=8)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/stockfish_regan_cache"),
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Print how many selected games are already present in the local cache, then exit.",
    )
    args = parser.parse_args()

    if args.depth < 16:
        raise SystemExit("Use --depth >= 16 for this analysis.")
    if args.multipv < 2:
        raise SystemExit("Use --multipv >= 2 for Regan-style move-choice analysis.")
    max_plies = args.max_plies if args.max_plies > 0 else None

    print("Loading Firestore ratings, benchmark results, and game results...")
    ratings, benchmark, results = fetch_firestore_data()
    model_rows = build_model_rows(
        ratings,
        benchmark,
        include_anchors=args.include_anchors,
        extra_models=set(args.extra_model),
    )
    model_rows = filter_model_rows(
        model_rows,
        focus_families=set(args.focus_family),
        focus_models=set(args.focus_model),
        keep_anchors=args.include_anchors,
    )
    print(
        f"Loaded ratings={len(ratings)} benchmark_results={len(benchmark)} "
        f"results={len(results)} reliable_targets={len(model_rows)}"
    )

    selected_game_ids = select_games(
        results=results,
        target_ids=set(model_rows),
        max_games_per_player=args.max_games_per_player,
        seed=args.seed,
    )
    print(
        f"Selected {len(selected_game_ids)} unique games "
        f"(max_games_per_player={args.max_games_per_player}, max_plies={max_plies}, "
        f"depth={args.depth}, multipv={args.multipv})"
    )

    if args.status_only:
        cache_key = f"depth{args.depth}_mpv{args.multipv}_plies{max_plies or 'all'}"
        cached = sum(
            1
            for game_id in selected_game_ids
            if (args.cache_dir / cache_key / f"{game_id}.json").exists()
        )
        print(f"Cached selected games: {cached}/{len(selected_game_ids)}")
        return

    aggs = {player_id: PlayerAgg() for player_id in model_rows}
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    try:
        try:
            engine.configure({"Threads": args.threads, "Hash": args.hash_mb})
        except chess.engine.EngineError:
            pass

        done = 0
        cache_hits = 0
        failures = 0
        for game_id in sorted(selected_game_ids):
            result = results[game_id]
            cache_key = f"depth{args.depth}_mpv{args.multipv}_plies{max_plies or 'all'}"
            cache_path = args.cache_dir / cache_key / f"{game_id}.json"
            was_cached = cache_path.exists()
            pgn = fetch_pgn(game_id)
            if not pgn:
                failures += 1
                continue
            analyzed = analyze_pgn(
                game_id=game_id,
                pgn=pgn,
                result=result,
                engine=engine,
                depth=args.depth,
                multipv=args.multipv,
                max_plies=max_plies,
                cache_path=cache_path,
            )
            if analyzed is None:
                failures += 1
                continue
            for side in ("white", "black"):
                player_id = result.get(f"{side}_id")
                if player_id in aggs:
                    aggs[player_id].games.add(game_id)
            add_move_stats(aggs, analyzed)
            done += 1
            if was_cached:
                cache_hits += 1
            if done % 10 == 0:
                print(
                    f"Analyzed {done}/{len(selected_game_ids)} games "
                    f"(cache_hits={cache_hits}, failures={failures})",
                    flush=True,
                )
    finally:
        engine.quit()

    rows: list[dict[str, Any]] = []
    for player_id, row in model_rows.items():
        features = aggs[player_id].to_features()
        if features.get("games_analyzed", 0) <= 0:
            continue
        rows.append({**row, **features})
    rows.sort(key=lambda row: row["residual"], reverse=True)

    print(f"Completed game analysis: games={done}, cache_hits={cache_hits}, failures={failures}")
    summarize_features(rows, min_feature_n=args.min_feature_n)
    print_interesting_rows(rows)


if __name__ == "__main__":
    main()
