#!/usr/bin/env python3
"""Audit which position benchmark rows are ready for rating prediction."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from position_benchmark.predictions import (  # noqa: E402
    benchmark_result_readiness,
    collect_equal_position_metrics,
    predict_rating,
    predict_rating_from_model_data_with_supplement,
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
from rating.cost_calculator import CostCalculator  # noqa: E402
from utils import resolve_player_id  # noqa: E402


ANCHOR_IDS = {"random-bot", "eubos", "maia-1900", "maia-1100", "survival-bot"}
AD_HOC_PLAYER_HINTS = {
    "gemini-3.1-pro-preview (high)": {
        "model_name": "google/gemini-3.1-pro-preview",
        "api": "gemini",
        "reasoning_effort": "high",
        "timeout": 1200,
    },
    "gemini-3-pro-preview (high)": {
        "model_name": "google/gemini-3-pro-preview",
        "api": "gemini",
        "reasoning_effort": "high",
        "timeout": 1200,
        "unavailable": True,
        "availability_note": "deprecated; no direct Gemini or OpenRouter endpoint",
    },
    "gemini-2.0-flash-001": {
        "model_name": "google/gemini-2.0-flash-001",
        "api": "openrouter",
        "unavailable": True,
        "availability_note": "OpenRouter returned no endpoints for google/gemini-2.0-flash-001",
    },
    "grok-4-fast": {
        "model_name": "x-ai/grok-4-fast",
        "api": "openrouter",
        "unavailable": True,
        "availability_note": "deprecated; OpenRouter reports xAI recommends switching to Grok 4.3",
    },
    "grok-4.1-fast": {
        "model_name": "x-ai/grok-4.1-fast",
        "api": "openrouter",
        "unavailable": True,
        "availability_note": "deprecated; OpenRouter reports xAI recommends switching to Grok 4.3",
    },
}


@dataclass
class AuditRow:
    player_id: str
    actual: float | None
    predicted: float | None
    error: float | None
    rd: float | None
    reliable: bool
    target_failure: bool
    target_model: bool
    equal_rows: int
    missing_equal_rows: int
    fen_mismatches: int
    best_metadata_mismatches: int
    duplicate_indices: int
    current_marker: bool
    overlay_applied: bool
    trusted_overlay: bool
    unavailable: bool = False
    availability_note: str | None = None

    @property
    def needs_rerun(self) -> bool:
        if self.unavailable:
            return False
        return (
            self.target_failure
            or (self.target_model and not self.current_marker)
            or self.missing_equal_rows > 0
            or self.fen_mismatches > 0
            or self.duplicate_indices > 0
        )

    @property
    def priority(self) -> float:
        score = 0.0
        if self.target_failure:
            score += 10000.0
        if self.target_model and not self.current_marker:
            score += 5000.0
        if self.error is not None:
            score += abs(self.error)
        score += 100.0 * self.missing_equal_rows
        score += 50.0 * self.fen_mismatches
        score += 25.0 * self.best_metadata_mismatches
        score += 25.0 * self.duplicate_indices
        return score


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def overlay_results(
    base_results: dict[str, Any],
    overlay_paths: list[Path],
) -> tuple[dict[str, Any], set[str]]:
    """Return results with partial overlays applied, plus players touched by overlays."""
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
            existing["summary"] = player_data.get("summary", existing.get("summary", {}))
    return merged, touched


def reliable_rating(player_id: str, rating: dict[str, Any], max_rd: float) -> bool:
    if player_id in ANCHOR_IDS:
        return True
    games = int(rating.get("games_played") or 0)
    rd = rating.get("games_rd", rating.get("rating_deviation"))
    return games > 0 and rd is not None and float(rd) < max_rd


def collect_configured_players(config_path: Path) -> dict[str, dict[str, Any]]:
    try:
        with config_path.open() as f:
            config = yaml.safe_load(f) or {}
    except (FileNotFoundError, yaml.YAMLError):
        return {}

    players: dict[str, dict[str, Any]] = {}
    for entry in config.get("llms", []):
        player_id = resolve_player_id(entry["player_id"], entry.get("reasoning_effort"))
        players[player_id] = entry
    return players


def safe_slug(player_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", player_id).strip("_").lower()


def uses_reasoning(config: dict[str, Any] | None, hint: dict[str, Any] | None = None) -> bool:
    """Infer whether position calls should use the reasoning completion estimate."""
    source = config or hint or {}
    if source.get("reasoning") is False:
        return False
    return bool(
        source.get("reasoning") is True
        or source.get("reasoning_effort")
        or source.get("reasoning_max_tokens")
    )


def estimate_chunk_cost(
    player_id: str,
    config: dict[str, Any] | None,
    chunk_indices: list[int],
    cost_calculator: CostCalculator,
) -> float | None:
    """Estimate API cost for one rerun chunk."""
    hint = AD_HOC_PLAYER_HINTS.get(player_id)
    model_name = None
    if config is not None:
        model_name = config.get("model_name")
    elif hint is not None:
        model_name = hint.get("model_name")

    return cost_calculator.estimate_position_benchmark_cost(
        player_id,
        model_name=model_name,
        num_positions=len(chunk_indices),
        reasoning=uses_reasoning(config, hint),
        use_budget_overrides=config is not None,
    )


def suggested_command_for_indices(
    player_id: str,
    config: dict[str, Any] | None,
    chunk_indices: list[int],
) -> str:
    suffix = f"{chunk_indices[0]}_{chunk_indices[-1]}"
    if config is None:
        hint = AD_HOC_PLAYER_HINTS.get(player_id)
        if hint is None:
            return "not configured; add or re-enable this player in config/benchmark.yaml before rerunning"

        chunk = " ".join(str(idx) for idx in chunk_indices)
        output = f"/tmp/{safe_slug(player_id)}_equal_{suffix}.json"
        command = (
            f"python position_benchmark/run_benchmark.py "
            f"--player-id {json.dumps(player_id)} "
            f"--model-name {json.dumps(hint['model_name'])} "
            f"--position-indices {chunk} --depth 30 --api {hint['api']} "
            f"--output {output}"
        )
        if hint.get("reasoning_effort"):
            command += f" --reasoning-effort {hint['reasoning_effort']}"
        if hint.get("timeout"):
            command += f" --timeout {hint['timeout']}"
        return command

    api = "openrouter"
    if config and str(config.get("model_name", "")).startswith("google/"):
        api = "gemini"
    elif config and config.get("api") == "codex":
        api = "codex"
    chunk = " ".join(str(idx) for idx in chunk_indices)
    output = f"/tmp/{safe_slug(player_id)}_equal_{suffix}.json"
    return (
        f"python position_benchmark/run_benchmark.py --players {json.dumps(player_id)} "
        f"--position-indices {chunk} --depth 30 --api {api} --output {output}"
    )


def suggested_commands(
    player_id: str,
    config: dict[str, Any] | None,
    equal_indices: list[int],
    chunk_size: int,
    *,
    all_chunks: bool,
) -> list[str]:
    """Build rerun command(s), chunking equal-position indices."""
    chunks = [
        equal_indices[start : start + chunk_size]
        for start in range(0, len(equal_indices), chunk_size)
    ]
    if not all_chunks:
        chunks = chunks[:1]
    return [
        suggested_command_for_indices(player_id, config, chunk)
        for chunk in chunks
        if chunk
    ]


def format_cost(cost: float | None) -> str:
    if cost is None:
        return "unknown"
    return f"${cost:.4f}"


def inspect_player_results(
    player_id: str,
    player_data: dict[str, Any],
    positions: list[dict[str, Any]],
    expected_equal_indices: set[int],
) -> tuple[int, int, int, int, int, bool]:
    rows = player_data.get("results", [])
    seen_equal: set[int] = set()
    duplicate_indices = 0
    fen_mismatches = 0
    best_metadata_mismatches = 0
    seen_indices: set[int] = set()

    for row in rows:
        idx = row.get("position_idx")
        if not isinstance(idx, int) or idx < 0 or idx >= len(positions):
            continue
        if idx in seen_indices:
            duplicate_indices += 1
        seen_indices.add(idx)
        position = positions[idx]
        if position.get("type") != "equal":
            continue
        if row.get("fen") and row.get("fen") != position.get("fen"):
            fen_mismatches += 1
            continue
        seen_equal.add(idx)
        if row.get("best_move") and row.get("best_move") != position.get("best_move"):
            best_metadata_mismatches += 1

    current_marker = benchmark_result_readiness(player_data, positions).is_ready

    return (
        len(seen_equal),
        len(expected_equal_indices - seen_equal),
        fen_mismatches,
        best_metadata_mismatches,
        duplicate_indices,
        current_marker,
    )


def print_audit(
    rows: list[AuditRow],
    configured_players: dict[str, dict[str, Any]],
    equal_indices: list[int],
    chunk_size: int,
    show_all: bool,
    all_chunks: bool,
    cost_calculator: CostCalculator,
    budget: float,
) -> None:
    displayed = [row for row in rows if show_all or row.needs_rerun]
    displayed.sort(key=lambda row: row.priority, reverse=True)

    target_rows = [
        row
        for row in rows
        if row.target_model and row.reliable and row.predicted is not None and not row.unavailable
    ]
    target_failures = [row for row in target_rows if row.target_failure]
    unavailable_target_rows = [
        row
        for row in rows
        if row.target_model and row.reliable and row.predicted is not None and row.unavailable
    ]
    errors = [row.error for row in rows if row.reliable and row.error is not None]
    abs_errors = [abs(error) for error in errors]

    print(
        f"Reliable rows: {len(errors)}  "
        f"RMSE={math.sqrt(sum(e * e for e in errors) / len(errors)):.1f}  "
        f"MAE={sum(abs_errors) / len(abs_errors):.1f}"
        if errors
        else "Reliable rows: 0"
    )
    print(f"Target failures: {len(target_failures)}/{len(target_rows)}")
    if unavailable_target_rows:
        unavailable_names = ", ".join(row.player_id for row in unavailable_target_rows)
        print(f"Unavailable target rows not counted as actionable failures: {unavailable_names}")
    print()
    print(
        "priority  player                                      actual  pred   err  "
        "eq miss fen best dup marker overlay trusted avail"
    )
    printed_cost = 0.0
    unknown_costs = 0
    for row in displayed:
        actual = "" if row.actual is None else f"{row.actual:.0f}"
        predicted = "" if row.predicted is None else f"{row.predicted:.0f}"
        error = "" if row.error is None else f"{row.error:+.0f}"
        marker = "yes" if row.current_marker else "no"
        overlay = "yes" if row.overlay_applied else "no"
        trusted = "yes" if row.trusted_overlay else "no"
        available = "no" if row.unavailable else "yes"
        print(
            f"{row.priority:8.0f}  {row.player_id[:42]:42} "
            f"{actual:>6} {predicted:>5} {error:>6} "
            f"{row.equal_rows:>2} {row.missing_equal_rows:>4} "
            f"{row.fen_mismatches:>3} {row.best_metadata_mismatches:>4} "
            f"{row.duplicate_indices:>3} {marker:>6} {overlay:>7} {trusted:>7} {available:>5}"
        )
        if row.unavailable and row.availability_note:
            print(f"          unavailable: {row.availability_note}")
        if row.target_failure and not row.unavailable:
            config = configured_players.get(row.player_id)
            configured = "configured" if config else "not in active config"
            chunks = [
                equal_indices[start : start + chunk_size]
                for start in range(0, len(equal_indices), chunk_size)
            ]
            if not all_chunks:
                chunks = chunks[:1]
            for chunk in chunks:
                if not chunk:
                    continue
                command = suggested_command_for_indices(row.player_id, config, chunk)
                cost = estimate_chunk_cost(row.player_id, config, chunk, cost_calculator)
                if cost is None:
                    unknown_costs += 1
                else:
                    printed_cost += cost
                print(f"          rerun ({configured}, est {format_cost(cost)}): {command}")

    if printed_cost or unknown_costs:
        suffix = f", {unknown_costs} unknown" if unknown_costs else ""
        print()
        print(f"Estimated cost for printed rerun commands: ${printed_cost:.4f} / ${budget:.2f}{suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=Path("data/ratings.json"))
    parser.add_argument("--results", type=Path, default=CORE_RESULTS_PATH)
    parser.add_argument("--positions", type=Path, default=CORE_POSITIONS_PATH)
    parser.add_argument("--config", type=Path, default=Path("config/benchmark.yaml"))
    parser.add_argument("--max-rd", type=float, default=100.0)
    parser.add_argument("--target-floor", type=float, default=1000.0)
    parser.add_argument("--target-max-error", type=float, default=200.0)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--all-chunks", action="store_true")
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--show-all", action="store_true")
    parser.add_argument("--overlay-results", type=Path, nargs="*", default=[])
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
        "--game-like-results",
        type=Path,
        default=GAME_LIKE_RESULTS_PATH,
        help="Optional supplemental game-like results file used by the production predictor",
    )
    parser.add_argument(
        "--game-like-positions",
        type=Path,
        default=GAME_LIKE_POSITIONS_PATH,
    )
    parser.add_argument(
        "--stability-results",
        type=Path,
        default=STABILITY_RESULTS_PATH,
        help="Optional scored continuation-probe summaries used by the production predictor",
    )
    parser.add_argument(
        "--trust-overlays",
        action="store_true",
        help="Mark overlay-touched legacy rows as trusted for diagnostics without marking them production-current",
    )
    args = parser.parse_args()

    ratings = load_json(args.ratings)
    base_results = load_json(args.results)
    results, overlay_players = overlay_results(base_results, args.overlay_results)
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
    if args.game_like_results.exists() and args.game_like_positions.exists():
        game_like_results = load_json(args.game_like_results)
        game_like_positions_data = load_json(args.game_like_positions)
        game_like_positions = (
            game_like_positions_data["positions"]
            if isinstance(game_like_positions_data, dict)
            else game_like_positions_data
        )
    if args.stability_results.exists():
        stability_results = load_json(args.stability_results)
    configured_players = collect_configured_players(args.config)
    cost_calculator = CostCalculator(config_path=args.config)

    equal_indices = [idx for idx, position in enumerate(positions) if position.get("type") == "equal"]
    expected_equal_indices = set(equal_indices)

    audit_rows: list[AuditRow] = []
    for player_id, player_data in results.items():
        if player_id in ANCHOR_IDS:
            continue
        rating = ratings.get(player_id)
        actual = float(rating["rating"]) if rating else None
        rd_value = None
        reliable = False
        target_model = False
        if rating:
            rd = rating.get("games_rd", rating.get("rating_deviation"))
            rd_value = float(rd) if rd is not None else None
            reliable = reliable_rating(player_id, rating, args.max_rd)
            target_model = reliable and actual is not None and actual > args.target_floor

        metrics = collect_equal_position_metrics(player_data.get("results", []), positions)
        predicted = None
        error = None
        if metrics is not None:
            predicted = predict_rating_from_model_data_with_supplement(
                player_data,
                positions,
                blunder_model_data=(
                    blunder_results.get(player_id)
                    if isinstance(blunder_results, dict)
                    else None
                ),
                blunder_positions=blunder_positions,
                game_like_model_data=(
                    game_like_results.get(player_id)
                    if isinstance(game_like_results, dict)
                    else None
                ),
                game_like_positions=game_like_positions,
                stability_probe_model_data=(
                    stability_results.get(player_id)
                    if isinstance(stability_results, dict)
                    else None
                ),
                require_ready=True,
            )
            if predicted is None:
                predicted = predict_rating(metrics.equal_cpl, metrics.best_pct, metrics.legal_pct)
            if actual is not None:
                error = actual - predicted

        (
            equal_rows,
            missing_equal_rows,
            fen_mismatches,
            best_metadata_mismatches,
            duplicate_indices,
            current_marker,
        ) = inspect_player_results(
            player_id,
            player_data,
            positions,
            expected_equal_indices,
        )
        target_failure = (
            target_model
            and error is not None
            and abs(error) > args.target_max_error
        )
        hint = AD_HOC_PLAYER_HINTS.get(player_id, {})

        audit_rows.append(
            AuditRow(
                player_id=player_id,
                actual=actual,
                predicted=predicted,
                error=error,
                rd=rd_value,
                reliable=reliable,
                target_failure=target_failure,
                target_model=target_model,
                equal_rows=equal_rows,
                missing_equal_rows=missing_equal_rows,
                fen_mismatches=fen_mismatches,
                best_metadata_mismatches=best_metadata_mismatches,
                duplicate_indices=duplicate_indices,
                current_marker=current_marker,
                overlay_applied=player_id in overlay_players,
                trusted_overlay=args.trust_overlays and player_id in overlay_players,
                unavailable=bool(hint.get("unavailable")),
                availability_note=hint.get("availability_note"),
            )
        )

    print_audit(
        audit_rows,
        configured_players,
        equal_indices,
        max(1, args.chunk_size),
        args.show_all,
        args.all_chunks,
        cost_calculator,
        args.budget,
    )


if __name__ == "__main__":
    main()
