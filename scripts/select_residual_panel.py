#!/usr/bin/env python3
"""Select a tiny PGN-derived residual panel that improves the current benchmark.

This script uses only local artifacts:
- `data/ratings.json`
- `position_benchmark/results.json`
- `position_benchmark/games/_results.json`
- `position_benchmark/games/*.pgn`

It mines shared exact positions from existing games, evaluates them with
Stockfish MultiPV, scores how much each position explains the residual left by
the current position benchmark, and greedily selects a small diverse panel.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn
import numpy as np
from scipy.stats import binom


ANCHOR_IDS = {"random-bot", "eubos", "maia-1900", "maia-1100", "survival-bot"}
FEATURE_OPTIONS = ("reciprocal_rank", "top3", "best", "top5")
DEFAULT_BUCKET_LIMITS = {
    "early_understanding": 3,
    "quiet_middlegame": 4,
    "defense": 2,
}


@dataclass(frozen=True)
class Occurrence:
    game_id: str
    fen: str
    fen_key: str
    move_history: tuple[str, ...]
    move_history_san: tuple[str, ...]
    side_to_move: str
    source_player_id: str
    opponent_id: str
    played_move: str
    played_move_san: str
    fullmove: int
    ply_before: int
    pieces: int
    legal_moves: int
    in_check: bool
    opening_key: str


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def stable_fen_key(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def cache_key(fen_key: str, depth: int, multipv: int) -> str:
    return hashlib.sha256(f"{depth}:{multipv}:{fen_key}".encode("utf-8")).hexdigest()[:24]


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
    max_rd: float,
    include_anchors: bool,
    extra_models: set[str],
    focus_families: set[str],
    focus_models: set[str],
) -> dict[str, dict[str, Any]]:
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

    rows: dict[str, dict[str, Any]] = {}
    for player_id, rating in ratings.items():
        benchmark = benchmark_results.get(player_id)
        if not benchmark:
            continue
        prediction = current_position_prediction(benchmark.get("summary") or {})
        if prediction is None:
            continue

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
        if not matches_focus(player_id) and not (include_anchors and player_id in ANCHOR_IDS):
            continue

        rows[player_id] = {
            "model": player_id,
            "actual": float(rating["rating"]),
            "current_pred": float(prediction),
            "residual": float(rating["rating"]) - float(prediction),
            "rd": float(rd),
            "games_played": games,
            "family": model_family(player_id),
        }
    return rows


def collect_shared_occurrences(
    *,
    games_dir: Path,
    results_json: Path,
    target_ids: set[str],
    max_plies: int,
) -> dict[str, dict[str, Any]]:
    results = load_json(results_json)
    shared: dict[str, dict[str, Any]] = {}

    parsed = 0
    for game_id, meta in results.items():
        white_id = meta.get("white_id")
        black_id = meta.get("black_id")
        if white_id not in target_ids and black_id not in target_ids:
            continue

        pgn_path = games_dir / f"{game_id}.pgn"
        if not pgn_path.exists():
            continue
        game = chess.pgn.read_game(io.StringIO(pgn_path.read_text(errors="replace")))
        if game is None:
            continue

        board = game.board()
        move_history: list[str] = []
        move_history_san: list[str] = []

        for ply_before, node in enumerate(game.mainline(), start=1):
            if ply_before > max_plies:
                break

            if not board.is_game_over(claim_draw=True):
                player_id = white_id if board.turn == chess.WHITE else black_id
                if player_id in target_ids:
                    move = node.move
                    try:
                        san = board.san(move)
                    except ValueError:
                        san = move.uci()

                    fen_key = stable_fen_key(board)
                    occurrence = Occurrence(
                        game_id=game_id,
                        fen=board.fen(),
                        fen_key=fen_key,
                        move_history=tuple(move_history),
                        move_history_san=tuple(move_history_san),
                        side_to_move="white" if board.turn == chess.WHITE else "black",
                        source_player_id=player_id,
                        opponent_id=black_id if board.turn == chess.WHITE else white_id,
                        played_move=move.uci(),
                        played_move_san=san,
                        fullmove=board.fullmove_number,
                        ply_before=ply_before,
                        pieces=len(board.piece_map()),
                        legal_moves=board.legal_moves.count(),
                        in_check=board.is_check(),
                        opening_key=" ".join(move_history[:8]),
                    )
                    item = shared.setdefault(
                        fen_key,
                        {
                            "fen_key": fen_key,
                            "occurrences": [],
                            "models": set(),
                            "games": set(),
                        },
                    )
                    item["occurrences"].append(occurrence)
                    item["models"].add(player_id)
                    item["games"].add(game_id)

            move = node.move
            try:
                move_history_san.append(board.san(move))
            except ValueError:
                move_history_san.append(move.uci())
            move_history.append(move.uci())
            board.push(move)

        parsed += 1
        if parsed % 1000 == 0:
            print(f"Parsed {parsed} local PGNs", flush=True)

    return shared


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
        return -10000.0 - 10.0 * abs(mate)
    cp = pov.score()
    return float(cp if cp is not None else 0.0)


def analyze_position(
    *,
    fen: str,
    fen_key: str,
    engine: chess.engine.SimpleEngine,
    depth: int,
    multipv: int,
    cache_dir: Path,
) -> dict[str, Any] | None:
    cache_path = cache_dir / f"{cache_key(fen_key, depth, multipv)}.json"
    if cache_path.exists():
        cached = load_json(cache_path)
        if (
            cached.get("fen_key") == fen_key
            and cached.get("depth") == depth
            and cached.get("multipv") == multipv
        ):
            return cached

    board = chess.Board(fen)
    width = max(1, min(multipv, board.legal_moves.count()))
    try:
        info_list = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=width,
        )
    except chess.engine.EngineError:
        return None
    if isinstance(info_list, dict):
        info_list = [info_list]

    best_eval = eval_to_cp(info_list[0], board.turn) if info_list else 0.0
    moves: list[dict[str, Any]] = []
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
        moves.append(
            {
                "move": move.uci(),
                "san": san,
                "eval": move_eval,
                "delta_cp": max(0.0, best_eval - move_eval),
            }
        )

    if not moves:
        return None

    analyzed = {
        "fen": fen,
        "fen_key": fen_key,
        "depth": depth,
        "multipv": multipv,
        "eval_before": best_eval,
        "multipv_moves": moves,
        "best_move": moves[0]["move"],
        "best_move_san": moves[0]["san"],
        "second_gap_cp": moves[1]["delta_cp"] if len(moves) > 1 else 10000.0,
        "near_best_25": sum(1 for item in moves if item["delta_cp"] <= 25),
        "near_best_50": sum(1 for item in moves if item["delta_cp"] <= 50),
        "near_best_100": sum(1 for item in moves if item["delta_cp"] <= 100),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(analyzed))
    return analyzed


def choose_canonical_occurrence(occurrences: list[Occurrence]) -> Occurrence:
    opening_counts = Counter(occ.opening_key for occ in occurrences)
    return sorted(
        occurrences,
        key=lambda occ: (
            -opening_counts[occ.opening_key],
            occ.fullmove,
            occ.game_id,
            occ.source_player_id,
        ),
    )[0]


def classify_bucket(occ: Occurrence, analyzed: dict[str, Any]) -> str | None:
    eval_before = float(analyzed["eval_before"])
    second_gap = float(analyzed["second_gap_cp"])
    near_best_100 = int(analyzed["near_best_100"])

    if (
        4 <= occ.fullmove <= 9
        and abs(eval_before) <= 120
        and occ.pieces >= 24
        and occ.legal_moves >= 18
        and not occ.in_check
        and near_best_100 >= 3
        and second_gap <= 110
    ):
        return "early_understanding"

    if (
        10 <= occ.fullmove <= 22
        and abs(eval_before) <= 150
        and occ.pieces >= 14
        and occ.legal_moves >= 14
        and not occ.in_check
        and near_best_100 >= 3
        and second_gap <= 120
    ):
        return "quiet_middlegame"

    if (
        8 <= occ.fullmove <= 22
        and -300 <= eval_before <= -80
        and occ.pieces >= 14
        and occ.legal_moves >= 10
        and near_best_100 >= 2
    ):
        return "defense"

    return None


def aggregate_occurrence_scores(
    occurrences: list[Occurrence],
    multipv_moves: list[dict[str, Any]],
    model_rows: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float]]:
    move_to_rank = {item["move"]: rank for rank, item in enumerate(multipv_moves, start=1)}
    fallback_rank = len(multipv_moves) + 1
    per_feature: dict[str, dict[str, list[float]]] = {
        "reciprocal_rank": defaultdict(list),
        "top3": defaultdict(list),
        "top5": defaultdict(list),
        "best": defaultdict(list),
    }

    for occ in occurrences:
        if occ.source_player_id not in model_rows:
            continue
        rank = move_to_rank.get(occ.played_move, fallback_rank)
        per_feature["reciprocal_rank"][occ.source_player_id].append(1.0 / rank)
        per_feature["top3"][occ.source_player_id].append(1.0 if rank <= 3 else 0.0)
        per_feature["top5"][occ.source_player_id].append(1.0 if rank <= 5 else 0.0)
        per_feature["best"][occ.source_player_id].append(1.0 if rank == 1 else 0.0)

    aggregated: dict[str, dict[str, float]] = {}
    for feature_name, by_model in per_feature.items():
        aggregated[feature_name] = {
            player_id: float(sum(values) / len(values))
            for player_id, values in by_model.items()
            if values
        }
    return aggregated


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
    train_x = np.asarray(train_x, dtype=float)
    test_x = np.asarray(test_x, dtype=float)
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
    penalty = np.eye(train_design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(train_design.T @ train_design + penalty, train_design.T @ train_y)
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

    return {
        "n_rows": len(used_rows),
        "rmse": rmse(y, pred),
        "mae": mae(y, pred),
        "r2": r2(y, pred),
        "actual": y,
        "pred": pred,
        "current": current,
    }


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


def build_position(candidate: dict[str, Any], feature_name: str, rank: int) -> dict[str, Any]:
    occ: Occurrence = candidate["canonical_occurrence"]
    analyzed = candidate["analyzed"]
    return {
        "position_id": f"residual-panel-{rank:04d}",
        "fen": occ.fen,
        "move_history": list(occ.move_history),
        "move_history_san": list(occ.move_history_san),
        "side_to_move": occ.side_to_move,
        "type": "equal",
        "regan_bucket": candidate["bucket"],
        "best_move": analyzed["best_move"],
        "best_move_san": analyzed["best_move_san"],
        "eval_before": round(float(analyzed["eval_before"]), 2),
        "multipv": analyzed["multipv_moves"],
        "second_gap_cp": round(float(analyzed["second_gap_cp"]), 2),
        "near_best_25": int(analyzed["near_best_25"]),
        "near_best_50": int(analyzed["near_best_50"]),
        "near_best_100": int(analyzed["near_best_100"]),
        "legal_moves": occ.legal_moves,
        "pieces": occ.pieces,
        "in_check": occ.in_check,
        "game_id": occ.game_id,
        "ply": occ.ply_before,
        "move_number": occ.fullmove,
        "source_player_id": occ.source_player_id,
        "opponent_id": occ.opponent_id,
        "source_played_move": occ.played_move,
        "source_played_move_san": occ.played_move_san,
        "fen_key": occ.fen_key,
        "opening_key": occ.opening_key,
        "common_model_coverage": candidate["coverage"],
        "common_occurrences": candidate["occurrences_count"],
        "selection_feature": feature_name,
        "selection_bucket": candidate["bucket"],
        "selection_candidate_id": candidate["candidate_id"],
        "selection_lofo_base_rmse": round(float(candidate["selected_stats"]["base_rmse"]), 2),
        "selection_lofo_panel_rmse": round(float(candidate["selected_stats"]["panel_rmse"]), 2),
        "selection_lofo_delta_rmse": round(float(candidate["selected_stats"]["delta_rmse"]), 2),
        "selection_rows": int(candidate["selected_stats"]["n_rows"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=Path("data/ratings.json"))
    parser.add_argument("--benchmark-results", type=Path, default=Path("position_benchmark/results.json"))
    parser.add_argument("--games-dir", type=Path, default=Path("position_benchmark/games"))
    parser.add_argument("--games-results", type=Path, default=Path("position_benchmark/games/_results.json"))
    parser.add_argument("--output-positions", type=Path, default=Path("position_benchmark/residual_panel_8.json"))
    parser.add_argument("--output-report", type=Path, default=Path("position_benchmark/residual_panel_8_report.json"))
    parser.add_argument("--stockfish-path", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/residual_panel_cache"))
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hash-mb", type=int, default=1024)
    parser.add_argument("--max-rd", type=float, default=100.0)
    parser.add_argument("--max-plies", type=int, default=24)
    parser.add_argument("--min-model-coverage", type=int, default=6)
    parser.add_argument("--panel-size", type=int, default=8)
    parser.add_argument("--bucket-limits", default=None)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--min-rows", type=int, default=8)
    parser.add_argument("--max-same-opening", type=int, default=1)
    parser.add_argument("--max-same-game", type=int, default=1)
    parser.add_argument("--include-anchors", action="store_true")
    parser.add_argument("--focus-family", action="append", default=[])
    parser.add_argument("--focus-model", action="append", default=[])
    parser.add_argument("--extra-model", action="append", default=[])
    parser.add_argument("--top-candidates", type=int, default=25)
    args = parser.parse_args()

    ratings = load_json(args.ratings)
    benchmark_results = load_json(args.benchmark_results)
    rows_by_id = build_model_rows(
        ratings=ratings,
        benchmark_results=benchmark_results,
        max_rd=args.max_rd,
        include_anchors=args.include_anchors,
        extra_models=set(args.extra_model),
        focus_families=set(args.focus_family),
        focus_models=set(args.focus_model),
    )
    if len(rows_by_id) < args.min_rows:
        raise SystemExit(f"Only {len(rows_by_id)} usable benchmark rows after filters.")

    print(f"Target model rows: {len(rows_by_id)}")
    print(f"Families: {sorted({row['family'] for row in rows_by_id.values()})}")

    shared = collect_shared_occurrences(
        games_dir=args.games_dir,
        results_json=args.games_results,
        target_ids=set(rows_by_id),
        max_plies=args.max_plies,
    )
    print(f"Shared exact FEN keys in local PGNs: {len(shared)}")

    prefiltered = [
        item
        for item in shared.values()
        if len(item["models"]) >= args.min_model_coverage
    ]
    prefiltered.sort(
        key=lambda item: (
            len(item["models"]),
            len(item["games"]),
            len(item["occurrences"]),
        ),
        reverse=True,
    )
    print(f"Candidates with coverage >= {args.min_model_coverage}: {len(prefiltered)}")

    cache_dir = args.cache_dir / f"depth{args.depth}_multipv{args.multipv}"
    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    candidates: list[dict[str, Any]] = []
    analyzed_count = 0
    bucket_counts: Counter[str] = Counter()
    try:
        try:
            engine.configure({"Threads": args.threads, "Hash": args.hash_mb})
        except chess.engine.EngineError:
            pass

        for item in prefiltered:
            occurrences: list[Occurrence] = item["occurrences"]
            canonical = choose_canonical_occurrence(occurrences)
            analyzed = analyze_position(
                fen=canonical.fen,
                fen_key=canonical.fen_key,
                engine=engine,
                depth=args.depth,
                multipv=args.multipv,
                cache_dir=cache_dir,
            )
            analyzed_count += 1
            if analyzed is None:
                continue
            bucket = classify_bucket(canonical, analyzed)
            if bucket is None:
                continue

            scores = aggregate_occurrence_scores(occurrences, analyzed["multipv_moves"], rows_by_id)
            coverage = len(scores["reciprocal_rank"])
            if coverage < args.min_model_coverage:
                continue

            candidate = {
                "candidate_id": canonical.fen_key,
                "bucket": bucket,
                "coverage": coverage,
                "occurrences_count": len(occurrences),
                "games_count": len(item["games"]),
                "canonical_occurrence": canonical,
                "analyzed": analyzed,
                "scores": scores,
            }
            candidates.append(candidate)
            bucket_counts[bucket] += 1
    finally:
        engine.quit()

    print(f"Analyzed shared candidates: {analyzed_count}")
    print(f"Scored candidates after bucket filters: {len(candidates)}")
    print(f"Bucket counts: {dict(bucket_counts)}")

    if not candidates:
        raise SystemExit("No usable residual-panel candidates found.")

    bucket_limits = parse_limits(args.bucket_limits)
    selected: list[tuple[dict[str, Any], str]] = []
    selected_ids: set[str] = set()
    selected_openings: Counter[str] = Counter()
    selected_games: Counter[str] = Counter()
    selected_buckets: Counter[str] = Counter()
    candidate_rankings: list[dict[str, Any]] = []

    while len(selected) < args.panel_size:
        best_pick: dict[str, Any] | None = None
        round_rankings: list[dict[str, Any]] = []

        for candidate in candidates:
            if candidate["candidate_id"] in selected_ids:
                continue
            if selected_buckets[candidate["bucket"]] >= bucket_limits.get(candidate["bucket"], args.panel_size):
                continue
            opening_key = candidate["canonical_occurrence"].opening_key
            game_id = candidate["canonical_occurrence"].game_id
            if selected_openings[opening_key] >= args.max_same_opening:
                continue
            if selected_games[game_id] >= args.max_same_game:
                continue

            candidate_best: dict[str, Any] | None = None
            for feature_name in FEATURE_OPTIONS:
                if len(candidate["scores"][feature_name]) < args.min_rows:
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
                delta_rmse = base_panel["rmse"] - panel["rmse"]
                trial = {
                    "candidate": candidate,
                    "feature_name": feature_name,
                    "n_rows": panel["n_rows"],
                    "base_rmse": base_panel["rmse"],
                    "panel_rmse": panel["rmse"],
                    "panel_mae": panel["mae"],
                    "panel_r2": panel["r2"],
                    "delta_rmse": delta_rmse,
                }
                if candidate_best is None or (
                    trial["delta_rmse"],
                    trial["n_rows"],
                    candidate["coverage"],
                ) > (
                    candidate_best["delta_rmse"],
                    candidate_best["n_rows"],
                    candidate_best["candidate"]["coverage"],
                ):
                    candidate_best = trial

            if candidate_best is None:
                continue

            round_rankings.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "bucket": candidate["bucket"],
                    "feature": candidate_best["feature_name"],
                    "coverage": candidate["coverage"],
                    "occurrences": candidate["occurrences_count"],
                    "game_id": candidate["canonical_occurrence"].game_id,
                    "opening_key": candidate["canonical_occurrence"].opening_key,
                    "delta_rmse": candidate_best["delta_rmse"],
                    "base_rmse": candidate_best["base_rmse"],
                    "panel_rmse": candidate_best["panel_rmse"],
                    "n_rows": candidate_best["n_rows"],
                    "fullmove": candidate["canonical_occurrence"].fullmove,
                    "eval_before": candidate["analyzed"]["eval_before"],
                }
            )

            if best_pick is None or (
                candidate_best["delta_rmse"],
                candidate_best["n_rows"],
                candidate["coverage"],
            ) > (
                best_pick["delta_rmse"],
                best_pick["n_rows"],
                best_pick["candidate"]["coverage"],
            ):
                best_pick = candidate_best

        if best_pick is None:
            break

        best_pick["candidate"]["selected_stats"] = {
            "base_rmse": best_pick["base_rmse"],
            "panel_rmse": best_pick["panel_rmse"],
            "delta_rmse": best_pick["delta_rmse"],
            "n_rows": best_pick["n_rows"],
        }
        selected.append((best_pick["candidate"], best_pick["feature_name"]))
        selected_ids.add(best_pick["candidate"]["candidate_id"])
        selected_openings[best_pick["candidate"]["canonical_occurrence"].opening_key] += 1
        selected_games[best_pick["candidate"]["canonical_occurrence"].game_id] += 1
        selected_buckets[best_pick["candidate"]["bucket"]] += 1

        candidate_rankings.extend(
            sorted(round_rankings, key=lambda item: (item["delta_rmse"], item["n_rows"], item["coverage"]), reverse=True)[:args.top_candidates]
        )
        print(
            f"Selected {len(selected)}/{args.panel_size}: "
            f"{best_pick['candidate']['bucket']} "
            f"{best_pick['feature_name']} "
            f"delta_rmse={best_pick['delta_rmse']:+.1f} "
            f"rows={best_pick['n_rows']} "
            f"coverage={best_pick['candidate']['coverage']}",
            flush=True,
        )

        if best_pick["delta_rmse"] <= 0.0:
            break

    if not selected:
        raise SystemExit("No panel positions survived greedy selection.")

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

    positions = [
        build_position(candidate, feature_name, rank)
        for rank, (candidate, feature_name) in enumerate(selected)
    ]
    metadata = {
        "description": "Greedy residual panel selected from local shared PGN positions",
        "depth": args.depth,
        "multipv": args.multipv,
        "panel_size": len(positions),
        "bucket_limits": bucket_limits,
        "selected_bucket_counts": dict(selected_buckets),
        "candidate_count": len(candidates),
        "coverage_threshold": args.min_model_coverage,
        "focus_families": args.focus_family,
        "focus_models": args.focus_model,
        "include_anchors": args.include_anchors,
        "alpha": args.alpha,
        "min_rows": args.min_rows,
        "baseline_rows": len(final_model_ids),
        "baseline_rmse": current_subset["rmse"],
        "panel_rmse": final_panel["rmse"],
        "baseline_mae": current_subset["mae"],
        "panel_mae": final_panel["mae"],
        "baseline_r2": current_subset["r2"],
        "panel_r2": final_panel["r2"],
    }

    args.output_positions.write_text(json.dumps({"metadata": metadata, "positions": positions}, indent=2))
    args.output_report.write_text(
        json.dumps(
            {
                "metadata": metadata,
                "selected": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "bucket": candidate["bucket"],
                        "feature": feature_name,
                        "coverage": candidate["coverage"],
                        "occurrences": candidate["occurrences_count"],
                        "game_id": candidate["canonical_occurrence"].game_id,
                        "opening_key": candidate["canonical_occurrence"].opening_key,
                        "fullmove": candidate["canonical_occurrence"].fullmove,
                        "eval_before": candidate["analyzed"]["eval_before"],
                        "selected_stats": candidate["selected_stats"],
                    }
                    for candidate, feature_name in selected
                ],
                "top_rankings": sorted(
                    candidate_rankings,
                    key=lambda item: (item["delta_rmse"], item["n_rows"], item["coverage"]),
                    reverse=True,
                )[: args.top_candidates],
            },
            indent=2,
        )
    )

    print()
    print(f"Final subset rows: {len(final_model_ids)}")
    print(
        f"Current benchmark on subset: RMSE={current_subset['rmse']:.1f} "
        f"MAE={current_subset['mae']:.1f} R2={current_subset['r2']:.3f}"
    )
    print(
        f"Residual panel fit on subset: RMSE={final_panel['rmse']:.1f} "
        f"MAE={final_panel['mae']:.1f} R2={final_panel['r2']:.3f}"
    )
    print(f"Wrote {len(positions)} positions to {args.output_positions}")
    print(f"Wrote report to {args.output_report}")


if __name__ == "__main__":
    main()
