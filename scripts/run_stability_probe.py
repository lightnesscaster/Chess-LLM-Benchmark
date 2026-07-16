#!/usr/bin/env python3
"""Run short continuation probes to measure chess-output stability."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
from pathlib import Path
import sys
from typing import Any

import chess
import chess.engine
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines.random_engine import RandomEngine  # noqa: E402
from game.game_runner import GameRunner  # noqa: E402
from llm import CodexSubagentPlayer, GeminiPlayer, OpenRouterPlayer  # noqa: E402
from llm.openrouter_completion_client import OpenRouterCompletionPlayer  # noqa: E402
from position_benchmark.run_benchmark import (  # noqa: E402
    config_is_engine,
    config_uses_reasoning,
    eval_to_cp,
    load_player_configs,
    replay_position_board,
)
from position_benchmark.layout import (  # noqa: E402
    GAME_LIKE_POSITIONS_PATH,
    PROTOCOL_SEQUENCE_RESULTS_PATH,
    STABILITY_RESULTS_PATH,
)
from position_benchmark.retry_protocol import (  # noqa: E402
    CONDITIONAL_RETRY_PROTOCOL_VERSION,
    derive_two_strike_retry_metrics,
)
from rating.cost_calculator import CostCalculator  # noqa: E402


DEFAULT_POSITION_LIMIT = 8
DEFAULT_PROBE_PLIES = 8
DEFAULT_SCORE_DEPTH = 10
STABILITY_PROBE_VERSION = "stratified-v2"
STABILITY_SELECTION_POLICY = "category-round-robin-v1"
PROTOCOL_SEQUENCE_VERSION = "protocol-sequence-v1"
PROTOCOL_SEQUENCE_SELECTION_POLICY = "category-round-robin-one-per-category-v1"
PROTOCOL_SEQUENCE_POSITION_LIMIT = 4
PROTOCOL_SEQUENCE_PROBE_PLIES = 16
EXPLICIT_SELECTION_VERSION = "explicit-selection-v1"


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def selected_positions(
    positions: list[dict[str, Any]],
    *,
    position_indices: list[int] | None,
    limit: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(positions))
    if position_indices is not None:
        requested = set(position_indices)
        indexed = [(idx, pos) for idx, pos in indexed if idx in requested]
        return indexed[:limit] if limit is not None else indexed

    if limit is None or limit >= len(indexed):
        return indexed

    buckets: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for item in indexed:
        position = item[1]
        bucket = str(
            position.get("regan_bucket")
            or position.get("screening_bucket")
            or position.get("category")
            or "uncategorized"
        )
        buckets.setdefault(bucket, []).append(item)

    # The active game-like panel is stored in category blocks. Round-robin
    # sampling prevents a short default probe from selecting only the first
    # block while preserving deterministic, reproducible position choices.
    selected: list[tuple[int, dict[str, Any]]] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for bucket_positions in buckets.values():
            if offset < len(bucket_positions):
                selected.append(bucket_positions[offset])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        offset += 1
    return selected


def pre_moves_from_position(position: dict[str, Any]) -> list[str]:
    """Return UCI pre-moves whose move stack reconstructs the position."""
    board = replay_position_board(position)
    return [move.uci() for move in board.move_stack]


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    rank = int(round((len(sorted_values) - 1) * percentile))
    return sorted_values[max(0, min(len(sorted_values) - 1, rank))]


def score_continuation_moves_from_pgn(
    pgn: str,
    *,
    model_side: chess.Color,
    pre_move_count: int,
    stockfish: chess.engine.SimpleEngine,
    depth: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score model and random-opponent continuation moves with Stockfish CPL."""
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return [], []

    board = chess.Board()
    model_scores: list[dict[str, Any]] = []
    opponent_scores: list[dict[str, Any]] = []
    model_turn_index = 0
    opponent_turn_index = 0
    preceding_opponent_cpl: float | None = None
    for ply, move in enumerate(game.mainline_moves(), start=1):
        should_score = ply > pre_move_count
        if should_score:
            perspective = board.turn
            is_model_move = perspective == model_side
            info_before = stockfish.analyse(board, chess.engine.Limit(depth=depth))
            eval_before = eval_to_cp(info_before, perspective)
            move_san = board.san(move)
        board.push(move)
        if should_score:
            info_after = stockfish.analyse(board, chess.engine.Limit(depth=depth))
            eval_model = -eval_to_cp(info_after, not perspective)
            cpl = max(0.0, eval_before - eval_model)
            score = {
                "ply": ply,
                "continuation_ply": ply - pre_move_count,
                "move": move.uci(),
                "move_san": move_san,
                "eval_before": eval_before,
                "eval_model": eval_model,
                "cpl": cpl,
            }
            if is_model_move:
                model_turn_index += 1
                score["model_turn_index"] = model_turn_index
                score["preceding_opponent_cpl"] = preceding_opponent_cpl
                model_scores.append(score)
            else:
                opponent_turn_index += 1
                score["opponent_turn_index"] = opponent_turn_index
                opponent_scores.append(score)
                preceding_opponent_cpl = cpl
    return model_scores, opponent_scores


def score_model_moves_from_pgn(
    pgn: str,
    *,
    model_side: chess.Color,
    pre_move_count: int,
    stockfish: chess.engine.SimpleEngine,
    depth: int,
) -> list[dict[str, Any]]:
    """Return model-only scores for callers using the original interface."""
    model_scores, _ = score_continuation_moves_from_pgn(
        pgn,
        model_side=model_side,
        pre_move_count=pre_move_count,
        stockfish=stockfish,
        depth=depth,
    )
    return model_scores


def _pgn_move_count(pgn: str) -> int:
    """Return the number of legal plies saved in a continuation PGN."""
    try:
        game = chess.pgn.read_game(io.StringIO(pgn or ""))
    except (ValueError, TypeError):
        return 0
    return sum(1 for _ in game.mainline_moves()) if game is not None else 0


def derive_retry_metrics(row: dict[str, Any]) -> dict[str, int]:
    """Derive the one eligible production retry outcome from a probe row.

    The game policy gives a player a retry after its first illegal move in a
    game. A later second strike forfeits immediately and is not another retry
    opportunity. A legal PGN ply at the first illegal event's move number proves
    that the retry recovered; two illegal events at that same move number prove
    that it failed. Provider errors without a saved legal ply remain unknown.
    """
    details = row.get("illegal_move_details") or []
    illegal_attempts = row.get("model_illegal_attempts")
    if illegal_attempts is None:
        illegal_attempts = len(details)
    metrics = derive_two_strike_retry_metrics(
        illegal_attempts=int(illegal_attempts or 0),
        illegal_events=details,
        saved_legal_plies=_pgn_move_count(str(row.get("pgn") or "")),
    )
    return {
        "model_retry_attempts": int(metrics["retry_attempts"]),
        "model_retry_recoveries": int(metrics["retry_recoveries"]),
        "model_retry_failures": int(metrics["retry_failures"]),
        "model_retry_unknown": int(metrics["retry_unknown"]),
    }


def retry_evidence_is_available(row: dict[str, Any]) -> bool:
    """Return whether a row can support exact retry-outcome reconstruction."""
    illegal_attempts = int(row.get("model_illegal_attempts", 0) or 0)
    return illegal_attempts == 0 or bool(row.get("illegal_move_details"))


def annotate_retry_metrics(row: dict[str, Any]) -> None:
    """Stamp one continuation row with deterministic production-retry evidence."""
    if not retry_evidence_is_available(row):
        row.pop("conditional_retry_protocol_version", None)
        for key in (
            "model_retry_attempts",
            "model_retry_recoveries",
            "model_retry_failures",
            "model_retry_unknown",
        ):
            row.pop(key, None)
        return
    row.update(derive_retry_metrics(row))
    row["conditional_retry_protocol_version"] = CONDITIONAL_RETRY_PROTOCOL_VERSION


def create_llm_player(
    player_id: str,
    player_config: dict[str, Any],
    *,
    api_backend: str,
    respect_config_api: bool,
) -> Any:
    """Create one LLM player using the same client choices as the benchmark runner."""
    common_kwargs = {
        "player_id": player_id,
        "temperature": player_config.get("temperature", 0.0),
        "reasoning": player_config.get("reasoning"),
        "reasoning_effort": player_config.get("reasoning_effort"),
        "timeout": player_config.get("timeout", 600),
    }

    if (respect_config_api and player_config.get("api") == "codex") or api_backend == "codex":
        return CodexSubagentPlayer(
            player_id=common_kwargs["player_id"],
            model_name=player_config.get("codex_model_name") or player_config.get("model_name"),
            reasoning_effort=common_kwargs["reasoning_effort"] or player_config.get("codex_reasoning_effort", "medium"),
            timeout=common_kwargs["timeout"],
            max_retries=player_config.get("codex_max_retries", 2),
            max_concurrent=player_config.get("codex_max_concurrent"),
            sandbox=player_config.get("codex_sandbox", "read-only"),
            ignore_rules=player_config.get("codex_ignore_rules", True),
            ephemeral=player_config.get("codex_ephemeral", True),
            include_legal_moves=player_config.get("codex_include_legal_moves", False),
            extra_args=player_config.get("codex_extra_args"),
            working_dir=player_config.get("codex_working_dir"),
        )

    if api_backend == "gemini":
        model_name = player_config.get("model_name", "")
        if model_name.startswith("google/"):
            model_name = model_name[len("google/") :]
        return GeminiPlayer(model_name=model_name, **common_kwargs)

    if respect_config_api and player_config.get("api") == "completion":
        return OpenRouterCompletionPlayer(
            player_id=common_kwargs["player_id"],
            model_name=player_config.get("model_name"),
            temperature=common_kwargs["temperature"],
            provider_order=player_config.get("provider_order"),
            provider_ignore=player_config.get("provider_ignore"),
            timeout=common_kwargs["timeout"],
        )

    return OpenRouterPlayer(
        model_name=player_config.get("model_name"),
        reasoning_max_tokens=player_config.get("reasoning_max_tokens"),
        max_tokens=player_config.get("max_tokens", 0),
        provider_order=player_config.get("provider_order"),
        provider_ignore=player_config.get("provider_ignore"),
        **common_kwargs,
    )


def summarize_player(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = len(rows)
    model_legal = sum(int(row["model_legal_moves"]) for row in rows)
    model_illegal = sum(int(row["model_illegal_attempts"]) for row in rows)
    model_attempts = model_legal + model_illegal
    own_forfeits = sum(1 for row in rows if row["model_forfeited"])
    api_errors = sum(1 for row in rows if row["termination"] == "api_error")
    probe_plies = sum(int(row["probe_plies_played"]) for row in rows)
    move_cpls = [
        float(score["cpl"])
        for row in rows
        for score in row.get("model_move_scores", [])
        if score.get("cpl") is not None
    ]
    first_move_cpls = [
        float(scores[0]["cpl"])
        for row in rows
        if (scores := row.get("model_move_scores", []))
        and scores[0].get("cpl") is not None
    ]
    later_move_cpls = [
        float(score["cpl"])
        for row in rows
        for score in row.get("model_move_scores", [])[1:]
        if score.get("cpl") is not None
    ]
    opponent_cpls = [
        float(score["cpl"])
        for row in rows
        for score in row.get("opponent_move_scores", [])
        if score.get("cpl") is not None
    ]
    blunder_moves = sum(1 for cpl in move_cpls if cpl >= 300)
    catastrophe_moves = sum(1 for cpl in move_cpls if cpl >= 1000)
    retry_source_rows = [row for row in rows if retry_evidence_is_available(row)]
    retry_rows = [derive_retry_metrics(row) for row in retry_source_rows]
    incomplete_retry_games = len(rows) - len(retry_source_rows)
    retry_attempts = sum(row["model_retry_attempts"] for row in retry_rows)
    retry_recoveries = sum(row["model_retry_recoveries"] for row in retry_rows)
    retry_failures = sum(row["model_retry_failures"] for row in retry_rows)
    retry_unknown = sum(row["model_retry_unknown"] for row in retry_rows)
    known_retry_outcomes = retry_recoveries + retry_failures
    first_attempt_illegals = sum(
        len(
            {
                int(detail["move_number"])
                for detail in row.get("illegal_move_details", [])
                if detail.get("move_number") is not None
            }
        )
        for row in retry_source_rows
    )
    first_attempt_turns = sum(
        int(row["model_legal_moves"]) + int(bool(row["model_forfeited"]))
        for row in retry_source_rows
    )

    summary = {
        "attempted_positions": attempted,
        "model_legal_moves": model_legal,
        "model_illegal_attempts": model_illegal,
        "model_attempts": model_attempts,
        "model_legal_pct": 100.0 * model_legal / model_attempts if model_attempts else 0.0,
        "model_first_attempt_turns": (
            first_attempt_turns if incomplete_retry_games == 0 else None
        ),
        "model_first_attempt_illegals": (
            first_attempt_illegals if incomplete_retry_games == 0 else None
        ),
        "model_first_attempt_illegal_pct": (
            100.0 * first_attempt_illegals / first_attempt_turns
            if first_attempt_turns and incomplete_retry_games == 0
            else None
        ),
        "illegal_attempts_per_model_move": model_illegal / model_legal if model_legal else float(model_illegal),
        "model_forfeits": own_forfeits,
        "model_forfeit_pct": 100.0 * own_forfeits / attempted if attempted else 0.0,
        "api_errors": api_errors,
        "probe_plies_played": probe_plies,
        "avg_probe_plies_played": probe_plies / attempted if attempted else 0.0,
        "model_scored_moves": len(move_cpls),
        "model_avg_cpl": sum(move_cpls) / len(move_cpls) if move_cpls else None,
        "model_first_move_avg_cpl": (
            sum(first_move_cpls) / len(first_move_cpls) if first_move_cpls else None
        ),
        "model_later_move_avg_cpl": (
            sum(later_move_cpls) / len(later_move_cpls) if later_move_cpls else None
        ),
        "model_median_cpl": _median(move_cpls),
        "model_p90_cpl": _percentile(move_cpls, 0.9),
        "model_300cp_blunders": blunder_moves,
        "model_300cp_blunder_pct": 100.0 * blunder_moves / len(move_cpls) if move_cpls else None,
        "model_1000cp_catastrophes": catastrophe_moves,
        "model_1000cp_catastrophe_pct": 100.0 * catastrophe_moves / len(move_cpls) if move_cpls else None,
        "opponent_scored_moves": len(opponent_cpls),
        "opponent_avg_cpl": (
            sum(opponent_cpls) / len(opponent_cpls) if opponent_cpls else None
        ),
        "model_retry_attempts": retry_attempts,
        "model_retry_recoveries": retry_recoveries,
        "model_retry_failures": retry_failures,
        "model_retry_unknown": retry_unknown,
        "model_retry_recovery_pct": (
            100.0 * retry_recoveries / known_retry_outcomes
            if known_retry_outcomes
            else None
        ),
    }
    if rows and incomplete_retry_games == 0:
        summary["conditional_retry_protocol_version"] = (
            CONDITIONAL_RETRY_PROTOCOL_VERSION
        )
    summary["conditional_retry"] = {
        "protocol_version": CONDITIONAL_RETRY_PROTOCOL_VERSION,
        "source": "continuation-game-events",
        "measured_games": len(retry_source_rows),
        "incomplete_evidence_games": incomplete_retry_games,
        "retry_attempts": retry_attempts,
        "retry_recoveries": retry_recoveries,
        "retry_failures": retry_failures,
        "retry_unknown": retry_unknown,
        "known_outcomes": known_retry_outcomes,
        "recovery_pct": summary["model_retry_recovery_pct"],
    }
    return summary


def backfill_retry_metrics(results: dict[str, Any]) -> tuple[int, int]:
    """Backfill saved row/summary retry evidence without making model calls."""
    updated_players = 0
    updated_rows = 0
    for player_data in results.values():
        rows = player_data.get("results") or []
        if not rows:
            continue
        for row in rows:
            annotate_retry_metrics(row)
            updated_rows += 1
        player_data.setdefault("summary", {}).update(summarize_player(rows))
        updated_players += 1
    return updated_players, updated_rows


def estimate_cost(
    players_to_test: dict[str, dict[str, Any]],
    *,
    selected_count: int,
    estimated_model_moves_per_position: int,
    api_backend: str,
    respect_config_api: bool,
    cost_calculator: CostCalculator,
) -> tuple[float, list[str]]:
    known = 0.0
    unknown: list[str] = []
    planned_calls = selected_count * estimated_model_moves_per_position
    for player_id, config in players_to_test.items():
        if config_is_engine(config):
            continue
        if (respect_config_api and config.get("api") == "codex") or api_backend == "codex":
            unknown.append(player_id)
            continue
        cost = cost_calculator.estimate_position_benchmark_cost(
            player_id,
            model_name=config.get("model_name"),
            num_positions=planned_calls,
            reasoning=config_uses_reasoning(config),
            use_budget_overrides=respect_config_api,
        )
        if cost is None:
            unknown.append(player_id)
        else:
            known += cost
    return known, unknown


async def run_probe_for_player(
    player_id: str,
    player_config: dict[str, Any],
    indexed_positions: list[tuple[int, dict[str, Any]]],
    *,
    probe_plies: int,
    api_backend: str,
    random_seed: int,
    verbose: bool,
    respect_config_api: bool,
    stockfish: chess.engine.SimpleEngine | None = None,
    score_depth: int = 0,
) -> dict[str, Any]:
    player = create_llm_player(
        player_id,
        player_config,
        api_backend=api_backend,
        respect_config_api=respect_config_api,
    )
    rows: list[dict[str, Any]] = []
    try:
        for ordinal, (position_idx, position) in enumerate(indexed_positions, start=1):
            board = replay_position_board(position)
            model_side = board.turn
            pre_moves = pre_moves_from_position(position)
            opponent = RandomEngine(
                player_id="stability-random",
                rating=400,
                seed=random_seed + position_idx,
            )
            if model_side == chess.WHITE:
                white, black = player, opponent
            else:
                white, black = opponent, player

            runner = GameRunner(
                white,
                black,
                max_moves=len(pre_moves) + probe_plies,
                verbose=verbose,
                pre_moves=pre_moves,
            )
            game_result, pgn = await runner.play_game()
            side_name = "white" if model_side == chess.WHITE else "black"
            opponent_side = "black" if side_name == "white" else "white"
            model_illegal = int(getattr(game_result, f"illegal_moves_{side_name}"))
            model_legal = int(getattr(game_result, f"total_moves_{side_name}"))
            opponent_illegal = int(getattr(game_result, f"illegal_moves_{opponent_side}"))
            model_forfeited = (
                game_result.termination == "forfeit_illegal_move"
                and game_result.winner not in (side_name, "draw")
            )
            model_move_scores = []
            opponent_move_scores = []
            if stockfish is not None and score_depth > 0:
                model_move_scores, opponent_move_scores = score_continuation_moves_from_pgn(
                    pgn,
                    model_side=model_side,
                    pre_move_count=len(pre_moves),
                    stockfish=stockfish,
                    depth=score_depth,
                )
            row = {
                    "position_idx": position_idx,
                    "position_id": position.get("position_id"),
                    "fen": position.get("fen"),
                    "model_side": side_name,
                    "pre_moves": len(pre_moves),
                    "probe_plies_requested": probe_plies,
                    "probe_plies_played": max(0, game_result.moves - len(pre_moves)),
                    "termination": game_result.termination,
                    "winner": game_result.winner,
                    "model_legal_moves": model_legal,
                    "model_illegal_attempts": model_illegal,
                    "opponent_illegal_attempts": opponent_illegal,
                    "model_forfeited": model_forfeited,
                    "tokens": (
                        game_result.tokens_white
                        if side_name == "white"
                        else game_result.tokens_black
                    ),
                    "illegal_move_details": [
                        item
                        for item in (game_result.illegal_move_details or [])
                        if item.get("side") == side_name
                    ],
                    "model_move_scores": model_move_scores,
                    "opponent_move_scores": opponent_move_scores,
                    "pgn": pgn,
                }
            annotate_retry_metrics(row)
            rows.append(row)
            score_suffix = ""
            if model_move_scores:
                avg_cpl = sum(float(score["cpl"]) for score in model_move_scores) / len(model_move_scores)
                score_suffix = f" avg_cpl={avg_cpl:.0f}"
            print(
                f"  [{ordinal}/{len(indexed_positions)}] {player_id}: "
                f"legal={model_legal} illegal={model_illegal} "
                f"term={game_result.termination}{score_suffix}",
                flush=True,
            )
    finally:
        close = getattr(player, "close", None)
        if close is not None:
            await close()

    return {
        "summary": summarize_player(rows),
        "results": rows,
    }


async def main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, default=GAME_LIKE_POSITIONS_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--players", nargs="+")
    parser.add_argument("--position-indices", nargs="+", type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--probe-plies", type=int)
    parser.add_argument("--score-depth", type=int, default=DEFAULT_SCORE_DEPTH)
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--api", choices=["openrouter", "gemini", "codex"], default="openrouter")
    parser.add_argument(
        "--ignore-config-api",
        action="store_true",
        help="Use --api even when a player config sets api: codex/completion",
    )
    parser.add_argument("--timeout", type=int, help="Override per-player API timeout in seconds")
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument("--max-estimated-cost", type=float)
    parser.add_argument("--allow-unknown-cost", action="store_true")
    parser.add_argument("--verbose-games", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate selection and print the planned workload without model calls",
    )
    parser.add_argument(
        "--protocol-sequence-v1",
        action="store_true",
        help="Use four category-balanced starts with 16 continuation plies",
    )
    parser.add_argument(
        "--backfill-retry-metrics",
        action="store_true",
        help="Backfill retry outcomes in --output without engines or model calls",
    )
    args = parser.parse_args()

    output_path = args.output or (
        PROTOCOL_SEQUENCE_RESULTS_PATH
        if args.protocol_sequence_v1
        else STABILITY_RESULTS_PATH
    )
    position_limit = args.limit if args.limit is not None else (
        PROTOCOL_SEQUENCE_POSITION_LIMIT
        if args.protocol_sequence_v1
        else DEFAULT_POSITION_LIMIT
    )
    probe_plies = args.probe_plies if args.probe_plies is not None else (
        PROTOCOL_SEQUENCE_PROBE_PLIES
        if args.protocol_sequence_v1
        else DEFAULT_PROBE_PLIES
    )

    if args.backfill_retry_metrics:
        if not output_path.exists():
            raise SystemExit(f"No continuation results found at {output_path}")
        output = load_json(output_path)
        updated_players, updated_rows = backfill_retry_metrics(output)
        output_path.write_text(json.dumps(output, indent=2) + "\n")
        print(
            f"Backfilled retry metrics for {updated_rows} rows across "
            f"{updated_players} players in {output_path}"
        )
        return

    if not args.players:
        raise SystemExit("--players is required unless --backfill-retry-metrics is used")

    positions_data = load_json(args.positions)
    positions = positions_data["positions"] if isinstance(positions_data, dict) else positions_data
    indexed_positions = selected_positions(
        positions,
        position_indices=args.position_indices,
        limit=position_limit,
    )
    if not indexed_positions:
        raise SystemExit("No positions selected.")

    all_players = load_player_configs()
    missing = [player_id for player_id in args.players if player_id not in all_players]
    if missing:
        raise SystemExit(f"Unknown player(s): {', '.join(missing)}")
    players_to_test = {player_id: dict(all_players[player_id]) for player_id in args.players}
    if args.timeout is not None:
        for config in players_to_test.values():
            config["timeout"] = args.timeout

    cost_calculator = CostCalculator()
    estimated_cost, unknown_cost_players = estimate_cost(
        players_to_test,
        selected_count=len(indexed_positions),
        estimated_model_moves_per_position=max(1, (probe_plies + 1) // 2),
        api_backend=args.api,
        respect_config_api=not args.ignore_config_api,
        cost_calculator=cost_calculator,
    )
    if estimated_cost or unknown_cost_players:
        suffix = f"; unknown for {', '.join(unknown_cost_players)}" if unknown_cost_players else ""
        print(f"Estimated API cost: ${estimated_cost:.4f}{suffix}")
    if args.max_estimated_cost is not None:
        if estimated_cost > args.max_estimated_cost:
            raise SystemExit(
                f"Estimated cost ${estimated_cost:.4f} exceeds --max-estimated-cost ${args.max_estimated_cost:.4f}"
            )
        if unknown_cost_players and not args.allow_unknown_cost:
            raise SystemExit(
                "Unknown API cost for: "
                + ", ".join(unknown_cost_players)
                + " (pass --allow-unknown-cost to run anyway)"
            )

    if args.dry_run:
        version = (
            EXPLICIT_SELECTION_VERSION
            if args.position_indices is not None
            else PROTOCOL_SEQUENCE_VERSION
            if args.protocol_sequence_v1
            else STABILITY_PROBE_VERSION
        )
        planned_calls = len(indexed_positions) * max(1, (probe_plies + 1) // 2)
        print(f"Probe version: {version}")
        print(f"Selected indices: {[index for index, _ in indexed_positions]}")
        print(f"Probe plies per position: {probe_plies}")
        print(f"Planned first-attempt calls per player: {planned_calls}")
        print(f"Output: {output_path}")
        return

    output: dict[str, Any] = {}
    if output_path.exists():
        output = load_json(output_path)

    stockfish = None
    if args.score_depth > 0:
        stockfish = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)

    try:
        for player_id, config in players_to_test.items():
            print(f"\n=== {player_id} ===", flush=True)
            output[player_id] = await run_probe_for_player(
                player_id,
                config,
                indexed_positions,
                probe_plies=probe_plies,
                api_backend=args.api,
                random_seed=args.random_seed,
                verbose=args.verbose_games,
                respect_config_api=not args.ignore_config_api,
                stockfish=stockfish,
                score_depth=args.score_depth,
            )
            output[player_id]["summary"]["player_id"] = player_id
            output[player_id]["summary"]["positions_file"] = str(args.positions)
            output[player_id]["summary"]["probe_plies"] = probe_plies
            output[player_id]["summary"]["score_depth"] = args.score_depth
            output[player_id]["summary"]["stability_probe_version"] = (
                EXPLICIT_SELECTION_VERSION
                if args.position_indices is not None
                else PROTOCOL_SEQUENCE_VERSION
                if args.protocol_sequence_v1
                else STABILITY_PROBE_VERSION
            )
            output[player_id]["summary"]["position_selection_policy"] = (
                "explicit-indices"
                if args.position_indices is not None
                else PROTOCOL_SEQUENCE_SELECTION_POLICY
                if args.protocol_sequence_v1
                else STABILITY_SELECTION_POLICY
            )
            output[player_id]["summary"]["selected_position_indices"] = [
                position_idx for position_idx, _ in indexed_positions
            ]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(output, indent=2) + "\n")
            print(f"Saved partial results to {output_path}", flush=True)
    finally:
        if stockfish is not None:
            stockfish.quit()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
