#!/usr/bin/env python3
"""Analyze position-benchmark rating prediction error against current ratings."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from position_benchmark.predictions import (
    DEFAULT_GAME_LIKE_CPL_CAP,
    DEFAULT_MIN_BLUNDER_POSITIONS,
    DEFAULT_MIN_EQUAL_POSITIONS,
    DEFAULT_MIN_GAME_LIKE_POSITIONS,
    DEFAULT_MIN_STABILITY_POSITIONS,
    DEFAULT_MIN_STABILITY_SCORED_MOVES,
    DEFAULT_MIN_STOCKFISH_DEPTH,
    benchmark_result_readiness,
    collect_equal_position_metrics,
    combine_equal_and_game_like_predictions,
    combine_prediction_with_downside_cap,
    predict_rating,
    stability_probe_prediction_cap,
    stability_probe_readiness,
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


ANCHOR_IDS = {"random-bot", "eubos", "maia-1900", "maia-1100", "survival-bot"}
UNAVAILABLE_PLAYER_IDS = {
    "gemini-3-pro-preview (high)": "deprecated; no direct Gemini or OpenRouter endpoint",
    "gemini-2.0-flash-001": "OpenRouter returned no endpoints for google/gemini-2.0-flash-001",
    "grok-4-fast": "deprecated; OpenRouter reports xAI recommends switching to Grok 4.3",
    "grok-4.1-fast": "deprecated; OpenRouter reports xAI recommends switching to Grok 4.3",
}


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def model_family(player_id: str) -> str:
    player = player_id.lower()
    prefixes = [
        "gemini-3.1",
        "gemini-3",
        "gemini-2.5",
        "gemini-2.0",
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.2",
        "gpt-5.1",
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
        "maia",
        "random",
        "survival",
        "eubos",
        "mistral",
        "qwen3",
        "glm",
        "gemma",
        "mimo",
        "o1",
    ]
    for prefix in prefixes:
        if player.startswith(prefix):
            return prefix
    return player.split()[0].split("-")[0]


def reliable_rating(player_id: str, rating: dict[str, Any], max_rd: float) -> bool:
    if player_id in ANCHOR_IDS:
        return True
    games = int(rating.get("games_played") or 0)
    rd = rating.get("games_rd", rating.get("rating_deviation"))
    return games > 0 and rd is not None and float(rd) < max_rd


def rmse(errors: list[float]) -> float:
    return math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def overlay_results(
    base_results: dict[str, Any],
    overlay_paths: list[Path],
) -> dict[str, Any]:
    """Return results with partial rerun rows overlaid by player and position index."""
    merged, _touched = overlay_results_with_touched(base_results, overlay_paths)
    return merged


def overlay_results_with_touched(
    base_results: dict[str, Any],
    overlay_paths: list[Path],
) -> tuple[dict[str, Any], set[str]]:
    """Return overlaid results plus player ids touched by overlays."""
    if not overlay_paths:
        return base_results, set()

    merged = json.loads(json.dumps(base_results))
    touched: set[str] = set()
    for path in overlay_paths:
        overlay = load_json(path)
        for player_id, player_data in overlay.items():
            touched.add(player_id)
            existing = merged.setdefault(player_id, {"summary": {}, "results": []})
            existing_rows = {
                row.get("position_idx"): row
                for row in existing.get("results", [])
                if isinstance(row.get("position_idx"), int)
            }
            for row in player_data.get("results", []):
                idx = row.get("position_idx")
                if isinstance(idx, int):
                    existing_rows[idx] = row
            existing["results"] = [existing_rows[idx] for idx in sorted(existing_rows)]
            if "summary" in player_data:
                existing["overlay_summary"] = player_data["summary"]
    return merged, touched


def print_group(
    label: str,
    rows: list[dict[str, Any]],
    target_floor: float,
    target_max_error: float,
    *,
    include_unavailable_targets: bool,
) -> None:
    if not rows:
        print(f"\n{label}: no rows")
        return
    errors = [row["error"] for row in rows]
    abs_errors = [abs(error) for error in errors]
    target_rows = [
        row
        for row in rows
        if row["actual"] > target_floor
        and (include_unavailable_targets or not row["unavailable"])
    ]
    target_failures = [row for row in target_rows if abs(row["error"]) > target_max_error]
    unavailable_target_rows = [
        row
        for row in rows
        if row["actual"] > target_floor and row["unavailable"] and not include_unavailable_targets
    ]

    print(f"\n{label}: {len(rows)} rows")
    print(
        f"  RMSE={rmse(errors):.1f}  MAE={mean(abs_errors):.1f}  "
        f"MaxAE={max(abs_errors):.1f}"
    )
    print(
        f"  Target actual>{target_floor:.0f}: "
        f"{len(target_failures)}/{len(target_rows)} outside +/-{target_max_error:.0f}"
    )
    if unavailable_target_rows:
        unavailable_names = ", ".join(row["player_id"] for row in unavailable_target_rows)
        print(f"  Unavailable target rows not counted: {unavailable_names}")
    for row in sorted(rows, key=lambda item: abs(item["error"]), reverse=True):
        marker = "FAIL" if row in target_failures else "    "
        blunder_text = (
            f" bl={row['blunder_predicted']:>7.0f}"
            if row.get("blunder_predicted") is not None
            else ""
        )
        game_like_text = (
            f" gl={row['game_like_predicted']:>7.0f}"
            if row.get("game_like_predicted") is not None
            else ""
        )
        stability_text = (
            f" st={row['stability_cap']:>7.0f}"
            if row.get("stability_cap") is not None
            else ""
        )
        print(
            f"  {marker} {row['player_id']:<40} "
            f"actual={row['actual']:>7.0f} pred={row['predicted']:>7.0f} "
            f"err={row['error']:>+7.0f}{blunder_text}{game_like_text}{stability_text} rd={row['rd']:>5.0f} "
            f"n={row['equal_positions']:>2} current={row['current']:<3} "
            f"trusted={row['trusted']:<3} "
            f"available={row['available']:<3} "
            f"family={row['family']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=Path("data/ratings.json"))
    parser.add_argument("--results", type=Path, default=CORE_RESULTS_PATH)
    parser.add_argument("--positions", type=Path, default=CORE_POSITIONS_PATH)
    parser.add_argument("--max-rd", type=float, default=100.0)
    parser.add_argument("--target-floor", type=float, default=1000.0)
    parser.add_argument("--target-max-error", type=float, default=200.0)
    parser.add_argument("--include-anchors", action="store_true")
    parser.add_argument(
        "--include-unavailable-targets",
        action="store_true",
        help="Count retired/unavailable rows in target-failure totals",
    )
    parser.add_argument(
        "--require-current",
        action="store_true",
        help="Only include production-ready history-replay benchmark rows",
    )
    parser.add_argument(
        "--trust-overlays",
        action="store_true",
        help="Include overlay-touched legacy rows in --require-current diagnostics without marking them production-current",
    )
    parser.add_argument("--min-equal-positions", type=int, default=DEFAULT_MIN_EQUAL_POSITIONS)
    parser.add_argument("--min-blunder-positions", type=int, default=DEFAULT_MIN_BLUNDER_POSITIONS)
    parser.add_argument("--min-stockfish-depth", type=int, default=DEFAULT_MIN_STOCKFISH_DEPTH)
    parser.add_argument(
        "--blunder-results",
        type=Path,
        default=BLUNDER_RESULTS_PATH,
        help="Optional historical blunder-panel results",
    )
    parser.add_argument(
        "--blunder-positions",
        type=Path,
        default=BLUNDER_POSITIONS_PATH,
    )
    parser.add_argument(
        "--overlay-results",
        type=Path,
        nargs="*",
        default=[],
        help="Partial result files to overlay by player_id and position_idx before analysis",
    )
    parser.add_argument(
        "--game-like-results",
        type=Path,
        default=GAME_LIKE_RESULTS_PATH,
        help="Optional fresh supplemental game-like results file used as a downside cap",
    )
    parser.add_argument(
        "--game-like-positions",
        type=Path,
        default=GAME_LIKE_POSITIONS_PATH,
        help="Positions file for --game-like-results",
    )
    parser.add_argument("--min-game-like-positions", type=int, default=DEFAULT_MIN_GAME_LIKE_POSITIONS)
    parser.add_argument(
        "--stability-results",
        type=Path,
        default=STABILITY_RESULTS_PATH,
        help="Optional scored continuation-probe summaries used as a weak-play downside cap",
    )
    parser.add_argument("--min-stability-positions", type=int, default=DEFAULT_MIN_STABILITY_POSITIONS)
    parser.add_argument("--min-stability-scored-moves", type=int, default=DEFAULT_MIN_STABILITY_SCORED_MOVES)
    args = parser.parse_args()

    ratings = load_json(args.ratings)
    results, overlay_players = overlay_results_with_touched(load_json(args.results), args.overlay_results)
    positions_data = load_json(args.positions)
    positions = positions_data["positions"] if isinstance(positions_data, dict) else positions_data
    blunder_results = None
    blunder_positions = None
    game_like_results = None
    game_like_positions = None
    stability_results = None
    if args.blunder_results.exists() and args.blunder_positions.exists():
        blunder_results = load_json(args.blunder_results)
        blunder_positions_data = load_json(args.blunder_positions)
        blunder_positions = (
            blunder_positions_data["positions"]
            if isinstance(blunder_positions_data, dict)
            else blunder_positions_data
        )
    if args.game_like_results.exists():
        game_like_results = load_json(args.game_like_results)
        game_like_positions_data = load_json(args.game_like_positions)
        game_like_positions = (
            game_like_positions_data["positions"]
            if isinstance(game_like_positions_data, dict)
            else game_like_positions_data
        )
    if args.stability_results.exists():
        stability_results = load_json(args.stability_results)

    rows: list[dict[str, Any]] = []
    skipped_reasons: dict[str, int] = {}
    for player_id, rating in ratings.items():
        if player_id not in results:
            continue
        if not reliable_rating(player_id, rating, args.max_rd):
            continue
        if not args.include_anchors and player_id in ANCHOR_IDS:
            continue

        model_data = results[player_id]
        readiness = benchmark_result_readiness(
            model_data,
            positions,
            min_equal_positions=args.min_equal_positions,
            min_stockfish_depth=args.min_stockfish_depth,
        )
        trusted_overlay = args.trust_overlays and player_id in overlay_players
        if args.require_current and not (readiness.is_ready or trusted_overlay):
            skipped_reasons[readiness.reason] = skipped_reasons.get(readiness.reason, 0) + 1
            continue

        metrics = collect_equal_position_metrics(model_data.get("results", []), positions)
        if metrics is None:
            continue
        predicted = predict_rating(metrics.equal_cpl, metrics.best_pct, metrics.legal_pct)
        blunder_predicted = None
        blunder_data = (
            blunder_results.get(player_id)
            if isinstance(blunder_results, dict)
            else None
        )
        blunder_readiness = None
        if blunder_data is not None and blunder_positions is not None:
            blunder_readiness = benchmark_result_readiness(
                blunder_data,
                blunder_positions,
                min_equal_positions=args.min_blunder_positions,
                min_stockfish_depth=args.min_stockfish_depth,
                position_type="blunder",
            )
        if blunder_readiness is not None and blunder_readiness.is_ready:
            blunder_metrics = collect_equal_position_metrics(
                blunder_data.get("results", []),
                blunder_positions,
                cpl_cap=DEFAULT_GAME_LIKE_CPL_CAP,
                position_type="blunder",
            )
            if blunder_metrics is not None:
                blunder_predicted = predict_rating(
                    blunder_metrics.equal_cpl,
                    blunder_metrics.best_pct,
                    blunder_metrics.legal_pct,
                )
                predicted = combine_equal_and_game_like_predictions(
                    predicted,
                    blunder_predicted,
                )
        game_like_predicted = None
        game_like_current = False
        if (
            isinstance(game_like_results, dict)
            and game_like_positions is not None
            and player_id in game_like_results
        ):
            game_like_data = game_like_results[player_id]
            game_like_readiness = benchmark_result_readiness(
                game_like_data,
                game_like_positions,
                min_equal_positions=args.min_game_like_positions,
                min_stockfish_depth=args.min_stockfish_depth,
            )
            game_like_current = game_like_readiness.is_ready
            if game_like_readiness.is_ready:
                game_like_metrics = collect_equal_position_metrics(
                    game_like_data.get("results", []),
                    game_like_positions,
                    cpl_cap=DEFAULT_GAME_LIKE_CPL_CAP,
                )
                if game_like_metrics is not None:
                    game_like_predicted = predict_rating(
                        game_like_metrics.equal_cpl,
                        game_like_metrics.best_pct,
                        game_like_metrics.legal_pct,
                    )
                    predicted = combine_equal_and_game_like_predictions(
                        predicted,
                        game_like_predicted,
                    )
        stability_cap = None
        stability_current = False
        if isinstance(stability_results, dict) and player_id in stability_results:
            stability_data = stability_results[player_id]
            stability_readiness = stability_probe_readiness(
                stability_data,
                min_positions=args.min_stability_positions,
                min_scored_moves=args.min_stability_scored_moves,
            )
            stability_current = stability_readiness.is_ready
            if stability_readiness.is_ready:
                stability_cap = stability_probe_prediction_cap(stability_data)
                predicted = combine_prediction_with_downside_cap(
                    predicted,
                    stability_cap,
                )

        rd = rating.get("games_rd", rating.get("rating_deviation", 0.0))
        actual = float(rating["rating"])
        unavailable_reason = UNAVAILABLE_PLAYER_IDS.get(player_id)
        rows.append(
            {
                "player_id": player_id,
                "actual": actual,
                "predicted": predicted,
                "error": actual - predicted,
                "rd": float(rd),
                "family": model_family(player_id),
                "equal_positions": metrics.total,
                "blunder_predicted": blunder_predicted,
                "game_like_predicted": game_like_predicted,
                "game_like_current": game_like_current,
                "stability_cap": stability_cap,
                "stability_current": stability_current,
                "skipped_mismatched_fen": metrics.skipped_mismatched_fen,
                "current": "yes" if readiness.is_ready else "no",
                "trusted": "yes" if trusted_overlay else "no",
                "unavailable": unavailable_reason is not None,
                "available": "no" if unavailable_reason else "yes",
                "unavailable_reason": unavailable_reason,
                "readiness_reason": readiness.reason,
            }
        )

    print_group(
        "Reliable non-anchor models" if not args.include_anchors else "Reliable models",
        rows,
        args.target_floor,
        args.target_max_error,
        include_unavailable_targets=args.include_unavailable_targets,
    )
    if skipped_reasons:
        print(f"\nSkipped by --require-current: {skipped_reasons}")


if __name__ == "__main__":
    main()
