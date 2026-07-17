#!/usr/bin/env python3
"""Merge and re-evaluate position benchmark result overlays with Stockfish."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from position_benchmark.run_benchmark import eval_to_cp  # noqa: E402
from position_benchmark.predictions import CURRENT_BENCHMARK_VERSION, result_row_is_current  # noqa: E402
from position_benchmark.layout import CORE_POSITIONS_PATH  # noqa: E402
from position_benchmark.retry_protocol import attach_conditional_retry_summary  # noqa: E402
from position_benchmark.scoring import ILLEGAL_MOVE_EVAL, illegal_move_cpl  # noqa: E402
from position_benchmark.token_accounting import refresh_result_token_usage  # noqa: E402


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def merge_result_files(paths: list[Path]) -> dict[str, Any]:
    """Merge result files by player_id and position_idx."""
    summary_keys_to_preserve = (
        "position_benchmark_version",
        "prompt_history_replay",
        "stockfish_depth",
    )
    merged: dict[str, Any] = {}
    for path in paths:
        data = load_json(path)
        for player_id, player_data in data.items():
            target = merged.setdefault(
                player_id,
                {
                    "summary": {},
                    "results": [],
                    "token_usage": {"prompt": 0, "completion": 0},
                },
            )
            rows_by_idx = {
                row.get("position_idx"): row
                for row in target.get("results", [])
                if isinstance(row.get("position_idx"), int)
            }
            for row in player_data.get("results", []):
                idx = row.get("position_idx")
                if isinstance(idx, int):
                    rows_by_idx[idx] = deepcopy(row)
            target["results"] = [rows_by_idx[idx] for idx in sorted(rows_by_idx)]

            refresh_result_token_usage(target)

            source_summary = player_data.get("summary") or {}
            target_summary = target.setdefault("summary", {})
            for key in summary_keys_to_preserve:
                value = source_summary.get(key)
                if value is None:
                    continue
                existing = target_summary.get(key)
                if existing is None:
                    target_summary[key] = value
                elif existing != value:
                    conflicts = target_summary.setdefault("summary_conflicts", {})
                    conflicts[key] = sorted({str(existing), str(value)})
    return merged


def filter_results(
    results: dict[str, Any],
    *,
    players: set[str] | None,
    position_indices: set[int] | None,
) -> dict[str, Any]:
    """Return a filtered copy of merged result data."""
    filtered: dict[str, Any] = {}
    for player_id, player_data in results.items():
        if players is not None and player_id not in players:
            continue

        rows = [
            deepcopy(row)
            for row in player_data.get("results", [])
            if position_indices is None or row.get("position_idx") in position_indices
        ]
        if not rows:
            continue

        filtered[player_id] = {
            "summary": deepcopy(player_data.get("summary", {})),
            "results": rows,
        }
        refresh_result_token_usage(filtered[player_id])
    return filtered


def evaluate_row(
    row: dict[str, Any],
    *,
    stockfish_path: str,
    depth: int,
    hash_mb: int,
) -> dict[str, Any]:
    """Return a re-evaluated copy of one position result row."""
    result = deepcopy(row)
    board = chess.Board(result["fen"])
    perspective = board.turn

    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        try:
            engine.configure({"Hash": hash_mb})
        except chess.engine.EngineError:
            pass

        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        eval_before = int(eval_to_cp(info, perspective))
        best_move = info["pv"][0]
        result["eval_before"] = eval_before
        result["eval_best"] = eval_before
        result["best_move"] = best_move.uci()
        result["best_move_san"] = board.san(best_move)

        model_move_text = result.get("model_move", "")
        try:
            model_move = chess.Move.from_uci(model_move_text)
            if model_move not in board.legal_moves:
                raise ValueError("model move is not legal")

            result["is_legal"] = True
            result["model_move_san"] = board.san(model_move)
            board.push(model_move)
            info_after = engine.analyse(board, chess.engine.Limit(depth=depth))
            eval_model = int(-eval_to_cp(info_after, not perspective))
            result["eval_model"] = eval_model
            result["cpl"] = max(0, eval_before - eval_model)
        except (ValueError, chess.InvalidMoveError):
            result["is_legal"] = False
            result["model_move_san"] = ""
            result["eval_model"] = ILLEGAL_MOVE_EVAL
            result["cpl"] = illegal_move_cpl(eval_before)

        result["is_best"] = result.get("model_move") == result.get("best_move")
        return result
    finally:
        engine.quit()


def stamp_reevaluated_row(
    row: dict[str, Any],
    *,
    source_summary: dict[str, Any],
    depth: int,
) -> None:
    """Record re-evaluation depth, preserving current-prompt provenance only when present."""
    source_is_current = (
        source_summary.get("position_benchmark_version") == CURRENT_BENCHMARK_VERSION
        and source_summary.get("prompt_history_replay") is True
    )
    if result_row_is_current(row, min_stockfish_depth=1) or source_is_current:
        row["position_benchmark_version"] = CURRENT_BENCHMARK_VERSION
        row["prompt_history_replay"] = True
    row["reevaluated_depth"] = depth
    row["stockfish_depth"] = depth


def recalculate_summary(
    player_id: str,
    results: list[dict[str, Any]],
    positions: list[dict[str, Any]],
) -> dict[str, Any]:
    pos_type_by_idx = {idx: position.get("type", "") for idx, position in enumerate(positions)}

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        total = len(rows)
        if total == 0:
            return {
                "total_positions": 0,
                "legal_moves": 0,
                "legal_pct": 0.0,
                "best_moves": 0,
                "best_pct": 0.0,
                "avg_cpl": 10000,
                "median_cpl": 10000,
            }
        cpls = [float(row.get("cpl", 0.0)) for row in rows]
        legal_rows = [row for row in rows if row.get("is_legal", True)]
        legal_cpls = [float(row.get("cpl", 0.0)) for row in legal_rows]
        sorted_cpls = sorted(cpls)
        mid = total // 2
        median = (
            (sorted_cpls[mid - 1] + sorted_cpls[mid]) / 2.0
            if total % 2 == 0
            else sorted_cpls[mid]
        )
        return {
            "total_positions": total,
            "legal_moves": len(legal_rows),
            "legal_pct": 100.0 * len(legal_rows) / total,
            "best_moves": sum(1 for row in rows if row.get("is_best", False)),
            "best_pct": 100.0 * sum(1 for row in rows if row.get("is_best", False)) / total,
            "avoided_blunders": sum(1 for row in rows if row.get("avoided_blunder", True)),
            "avoided_pct": 100.0 * sum(1 for row in rows if row.get("avoided_blunder", True)) / total,
            "avg_cpl": sum(cpls) / total,
            "avg_cpl_legal": sum(legal_cpls) / len(legal_cpls) if legal_cpls else 10000,
            "median_cpl": median,
        }

    summary = summarize(results)
    summary["player_id"] = player_id
    attach_conditional_retry_summary(summary, results)
    for type_name in ("blunder", "equal"):
        type_rows = [
            row
            for row in results
            if pos_type_by_idx.get(row.get("position_idx", -1)) == type_name
        ]
        if type_rows:
            summary[type_name] = summarize(type_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_results", type=Path, nargs="+")
    parser.add_argument("--positions", type=Path, default=CORE_POSITIONS_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stockfish-path", default="stockfish")
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hash-mb", type=int, default=128)
    parser.add_argument("--players", nargs="+", help="Only re-evaluate these player ids")
    parser.add_argument(
        "--position-indices",
        type=int,
        nargs="+",
        help="Only re-evaluate these original position indices",
    )
    args = parser.parse_args()

    positions_data = load_json(args.positions)
    positions = positions_data["positions"] if isinstance(positions_data, dict) else positions_data
    merged = filter_results(
        merge_result_files(args.input_results),
        players=set(args.players) if args.players else None,
        position_indices=set(args.position_indices) if args.position_indices else None,
    )

    for player_id, player_data in merged.items():
        rows = player_data.get("results", [])
        print(f"Re-evaluating {player_id}: {len(rows)} rows at depth {args.depth}", flush=True)
        reevaluated: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [
                executor.submit(
                    evaluate_row,
                    row,
                    stockfish_path=args.stockfish_path,
                    depth=args.depth,
                    hash_mb=args.hash_mb,
                )
                for row in rows
            ]
            for completed, future in enumerate(as_completed(futures), start=1):
                reevaluated.append(future.result())
                if completed % 10 == 0 or completed == len(futures):
                    print(f"  {completed}/{len(futures)} done", flush=True)

        reevaluated.sort(key=lambda row: row.get("position_idx", -1))
        source_summary = player_data.get("summary") or {}
        for row in reevaluated:
            stamp_reevaluated_row(row, source_summary=source_summary, depth=args.depth)
        player_data["results"] = reevaluated
        player_data["summary"] = recalculate_summary(player_id, reevaluated, positions)
        current_rows = sum(1 for row in reevaluated if result_row_is_current(row, min_stockfish_depth=args.depth))
        if reevaluated and current_rows == len(reevaluated):
            player_data["summary"]["position_benchmark_version"] = CURRENT_BENCHMARK_VERSION
            player_data["summary"]["prompt_history_replay"] = True
        else:
            player_data["summary"]["position_benchmark_version"] = "mixed-or-legacy"
            player_data["summary"]["prompt_history_replay"] = False
        player_data["summary"]["current_result_rows"] = current_rows
        player_data["summary"]["legacy_or_mixed_result_rows"] = len(reevaluated) - current_rows
        if "stockfish_depth" in source_summary:
            player_data["summary"]["source_stockfish_depth"] = source_summary["stockfish_depth"]
        player_data["summary"]["reevaluated_depth"] = args.depth
        player_data["summary"]["stockfish_depth"] = args.depth

    args.output.write_text(json.dumps(merged, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
