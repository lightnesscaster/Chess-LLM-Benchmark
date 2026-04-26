#!/usr/bin/env python3
"""Analyze real game PGNs with Stockfish phase features.

This is a read-only Firestore analysis script. It evaluates legal PGN moves
with Stockfish, caches per-game evals locally, and compares game-derived phase
features to the residual of the static position benchmark predictor.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chess
import chess.engine
import chess.pgn
import numpy as np
from scipy.stats import binom

from firebase_client import (
    BENCHMARK_RESULTS_COLLECTION,
    GAMES_COLLECTION,
    RATINGS_COLLECTION,
    RESULTS_COLLECTION,
    get_firestore_client,
)


ANCHOR_IDS = {"random-bot", "eubos", "maia-1900", "maia-1100", "survival-bot"}


def eval_to_cp(info: chess.engine.InfoDict, perspective: chess.Color) -> float:
    """Convert a Stockfish score to centipawns from `perspective`."""
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
    """Probability of at most one illegal move in `game_length` moves."""
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
    """Current production-ish static position formula."""
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
        sd[sd == 0] = 1.0
        train_x = np.column_stack([np.ones(len(train)), (X[train] - mu) / sd])
        coef = np.linalg.lstsq(train_x, y[train], rcond=None)[0]
        test_x = np.column_stack([np.ones(1), (X[[i]] - mu) / sd])
        pred[i] = float((test_x @ coef)[0])
    return pred


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
        self.counts: dict[str, int] = defaultdict(int)
        self.games: set[str] = set()

    def add_value(self, name: str, value: float) -> None:
        self.stats[name].add(value)

    def add_rate_event(self, name: str, hit: bool) -> None:
        self.stats[name].add(1.0 if hit else 0.0)

    def add_count(self, name: str, value: int = 1) -> None:
        self.counts[name] += value

    def to_features(self) -> dict[str, float]:
        features: dict[str, float] = {"games_analyzed": float(len(self.games))}
        for key, stat in self.stats.items():
            mean = stat.mean()
            if mean is not None:
                features[key] = mean
                features[f"{key}_n"] = float(stat.count)
        for key, value in self.counts.items():
            features[key] = float(value)
        return features


def phase_name(fullmove: int) -> str:
    if fullmove <= 10:
        return "opening"
    if fullmove <= 25:
        return "middlegame"
    return "endgame"


def pgn_hash(pgn: str) -> str:
    return hashlib.sha256(pgn.encode("utf-8")).hexdigest()


def analyze_pgn(
    *,
    game_id: str,
    pgn: str,
    result: dict[str, Any],
    engine: chess.engine.SimpleEngine,
    depth: int,
    max_plies: int | None,
    cache_path: Path,
) -> dict[str, Any] | None:
    """Analyze a PGN, with one Stockfish eval per position."""
    digest = pgn_hash(pgn)
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if (
                cached.get("pgn_sha256") == digest
                and cached.get("depth") == depth
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
        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        pv = info.get("pv") or []
        best_move = pv[0].uci() if pv else ""
        position_evals.append(
            {
                "eval_side_to_move": eval_to_cp(info, board.turn),
                "side_to_move": "white" if board.turn == chess.WHITE else "black",
                "best_move": best_move,
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
        eval_before = float(before["eval_side_to_move"])
        eval_after = -float(after["eval_side_to_move"])
        cpl = max(0.0, eval_before - eval_after)
        moves.append(
            {
                **meta,
                "phase": phase_name(int(meta["fullmove"])),
                "eval_before": eval_before,
                "eval_after": eval_after,
                "cpl": cpl,
                "best_move": before["best_move"],
                "is_best": before["best_move"] == meta["played_move"],
            }
        )

    analyzed = {
        "game_id": game_id,
        "pgn_sha256": digest,
        "depth": depth,
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


def add_result_stats(aggs: dict[str, PlayerAgg], game_id: str, result: dict[str, Any]) -> None:
    for side in ("white", "black"):
        player_id = result.get(f"{side}_id")
        if player_id not in aggs:
            continue
        agg = aggs[player_id]
        agg.games.add(game_id)
        illegal = float(result.get(f"illegal_moves_{side}") or 0.0)
        total = float(result.get(f"total_moves_{side}") or 0.0)
        if total > 0:
            agg.add_value("game_illegal_rate", illegal / total)
        winner = result.get("winner")
        score = 0.5 if winner == "draw" else 1.0 if winner == side else 0.0
        agg.add_value("sample_score_rate", score)
        agg.add_rate_event("forfeit_loss_rate", result.get("termination") == "forfeit_illegal_move" and score == 0.0)


def add_move_stats(aggs: dict[str, PlayerAgg], analyzed: dict[str, Any]) -> None:
    for move in analyzed.get("moves", []):
        player_id = move["player_id"]
        if player_id not in aggs:
            continue
        agg = aggs[player_id]
        phase = move["phase"]
        cpl = float(move["cpl"])
        eval_before = float(move["eval_before"])
        eval_after = float(move["eval_after"])
        own_no = int(move["player_move_no"])

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

        for bucket in buckets:
            agg.add_value(f"{bucket}_cpl", cpl)
            agg.add_rate_event(f"{bucket}_best_rate", bool(move["is_best"]))
            agg.add_rate_event(f"{bucket}_blunder100", cpl >= 100)
            agg.add_rate_event(f"{bucket}_blunder300", cpl >= 300)
            agg.add_rate_event(f"{bucket}_blunder500", cpl >= 500)

        agg.add_value("eval_after_all_moves", eval_after)
        if phase == "opening":
            agg.add_value("opening_eval_after", eval_after)
        if eval_before >= 150:
            agg.add_rate_event("advantage_thrown_to_equal_or_worse", eval_after < 100)
        if eval_before >= 300:
            agg.add_rate_event("big_advantage_thrown", eval_after < 150)
        if eval_before > -500:
            agg.add_rate_event("catastrophe_to_losing", eval_after < -500)

        for milestone in (5, 8, 10, 12, 15):
            if own_no == milestone:
                agg.add_value(f"eval_after_own_move_{milestone}", eval_after)


def summarize_features(rows: list[dict[str, Any]], min_feature_n: int) -> None:
    y = np.asarray([row["actual"] for row in rows], dtype=float)
    current = np.asarray([row["current_pred"] for row in rows], dtype=float)
    residuals = [row["residual"] for row in rows]

    print()
    print(f"Rows with phase features: {len(rows)}")
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
        used_rows: list[dict[str, Any]] = []
        for row in rows:
            value = row.get(feature)
            n_value = row.get(f"{feature}_n", min_feature_n)
            if value is None or not math.isfinite(float(value)) or n_value < min_feature_n:
                continue
            values.append(float(value))
            res.append(float(row["residual"]))
            ratings.append(float(row["actual"]))
            current_values.append(float(row["current_pred"]))
            used_rows.append(row)
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
                "values": values,
                "rows": used_rows,
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
            f"  {item['feature']:<36} n={item['n']:>2} "
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
            f"  {feature:<36} n={n:>2} "
            f"delta_rmse={delta:+.1f} RMSE={model_rmse:.1f} MAE={model_mae:.1f}"
        )

    combo_sets = {
        "opening quality": ["current_pred", "opening_cpl", "opening_blunder300", "eval_after_own_move_10"],
        "early blunders": ["current_pred", "early15_cpl", "early15_blunder300", "catastrophe_to_losing"],
        "quiet/conversion": ["current_pred", "quiet_equal_cpl", "conversion_cpl", "advantage_thrown_to_equal_or_worse"],
        "phase cpl": ["current_pred", "opening_cpl", "middlegame_cpl", "endgame_cpl"],
        "legal+opening": ["current_pred", "game_illegal_rate", "opening_cpl", "eval_after_own_move_10"],
    }
    print()
    print("Small combo LOO checks:")
    for name, features in combo_sets.items():
        if all(feature in rows[0] for feature in features):
            complete_rows = [
                row
                for row in rows
                if all(
                    row.get(feature) is not None
                    and math.isfinite(float(row.get(feature)))
                    for feature in features
                )
            ]
            if len(complete_rows) < 12:
                continue
            yy = np.asarray([row["actual"] for row in complete_rows], dtype=float)
            base = np.asarray([row["current_pred"] for row in complete_rows], dtype=float)
            X = np.asarray([[row[feature] for feature in features] for row in complete_rows], dtype=float)
            pred = loo_linear_predict(X, yy)
            print(
                f"  {name:<18} n={len(complete_rows):>2} "
                f"base_RMSE={rmse(yy, base):.1f} RMSE={rmse(yy, pred):.1f} "
                f"MAE={mae(yy, pred):.1f} R2={r2(yy, pred):.3f}"
            )


def print_interesting_rows(rows: list[dict[str, Any]]) -> None:
    interesting = [
        "gemini-3.1-pro-preview (high)",
        "gemini-3.1-pro-preview (medium)",
        "gemini-3-pro-preview (high)",
        "gemini-3-pro-preview (medium)",
        "gemini-3-flash-preview (high)",
        "gemini-3-flash-preview (medium)",
        "grok-4.1-fast",
        "grok-4-fast",
        "gpt-5.1 (high)",
        "gpt-5.1-chat",
        "gpt-5.2 (high)",
        "gpt-5.2 (no thinking)",
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
            f"openBest={row.get('opening_best_rate', float('nan')):>5.1%} "
            f"openCPL={row.get('opening_cpl', float('nan')):>7.1f} "
            f"earlyBest={row.get('early15_best_rate', float('nan')):>5.1%} "
            f"early300={row.get('early15_blunder300', float('nan')):>5.1%} "
            f"eval10={row.get('eval_after_own_move_10', float('nan')):>7.1f} "
            f"quietCPL={row.get('quiet_equal_cpl', float('nan')):>7.1f} "
            f"convCPL={row.get('conversion_cpl', float('nan')):>7.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stockfish-path", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--max-plies", type=int, default=60)
    parser.add_argument("--max-games-per-player", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-anchors", action="store_true")
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
        default=Path("data/stockfish_phase_cache"),
    )
    args = parser.parse_args()

    if args.depth < 16:
        raise SystemExit("Use --depth >= 16 for this analysis.")
    max_plies = args.max_plies if args.max_plies > 0 else None

    print("Loading Firestore ratings, benchmark results, and game results...")
    ratings, benchmark, results = fetch_firestore_data()
    model_rows = build_model_rows(
        ratings,
        benchmark,
        include_anchors=args.include_anchors,
        extra_models=set(args.extra_model),
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
        f"(max_games_per_player={args.max_games_per_player}, max_plies={max_plies}, depth={args.depth})"
    )

    aggs = {player_id: PlayerAgg() for player_id in model_rows}
    for game_id in selected_game_ids:
        add_result_stats(aggs, game_id, results[game_id])

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
            cache_key = f"depth{args.depth}_plies{max_plies or 'all'}"
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
                max_plies=max_plies,
                cache_path=cache_path,
            )
            if analyzed is None:
                failures += 1
                continue
            add_move_stats(aggs, analyzed)
            done += 1
            if was_cached:
                cache_hits += 1
            if done % 25 == 0:
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
