#!/usr/bin/env python3
"""Promote verified position-benchmark overlays into a results file."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from position_benchmark.predictions import (  # noqa: E402
    CURRENT_BENCHMARK_VERSION,
    benchmark_result_readiness,
    result_row_is_current,
)
from scripts.reevaluate_position_result_overlays import recalculate_summary  # noqa: E402


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def merge_overlays(
    base_results: dict[str, Any],
    overlay_paths: list[Path],
) -> tuple[dict[str, Any], set[str]]:
    """Merge overlay rows by player and original position index."""
    merged = deepcopy(base_results)
    touched: set[str] = set()

    for path in overlay_paths:
        overlay = load_json(path)
        for player_id, player_data in overlay.items():
            touched.add(player_id)
            target = merged.setdefault(player_id, {"summary": {}, "results": []})
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

            usage = player_data.get("token_usage") or {}
            if usage:
                target_usage = target.setdefault("token_usage", {"prompt": 0, "completion": 0})
                target_usage["prompt"] = target_usage.get("prompt", 0) + usage.get("prompt", 0)
                target_usage["completion"] = target_usage.get("completion", 0) + usage.get("completion", 0)

    return merged, touched


def refresh_summary(
    player_id: str,
    player_data: dict[str, Any],
    positions: list[dict[str, Any]],
    min_stockfish_depth: int,
) -> None:
    """Recalculate summary and attach equal-row freshness counts."""
    rows = player_data.get("results", [])
    summary = recalculate_summary(player_id, rows, positions)
    equal_rows = [
        row
        for row in rows
        if isinstance(row.get("position_idx"), int)
        and 0 <= row["position_idx"] < len(positions)
        and positions[row["position_idx"]].get("type") == "equal"
        and not (row.get("fen") and row.get("fen") != positions[row["position_idx"]].get("fen"))
    ]
    current_equal_rows = sum(
        1 for row in equal_rows if result_row_is_current(row, min_stockfish_depth=min_stockfish_depth)
    )

    summary["current_equal_rows"] = current_equal_rows
    summary["legacy_or_mixed_equal_rows"] = len(equal_rows) - current_equal_rows
    if equal_rows and current_equal_rows == len(equal_rows):
        summary["position_benchmark_version"] = CURRENT_BENCHMARK_VERSION
        summary["prompt_history_replay"] = True
        summary["stockfish_depth"] = min(int(row["stockfish_depth"]) for row in equal_rows)
    else:
        summary["position_benchmark_version"] = "mixed-or-legacy"
        summary["prompt_history_replay"] = False

    player_data["summary"] = summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("overlay_results", type=Path, nargs="+")
    parser.add_argument("--base", type=Path, default=Path("position_benchmark/results.json"))
    parser.add_argument("--positions", type=Path, default=Path("position_benchmark/positions.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-equal-positions", type=int, default=50)
    parser.add_argument("--min-stockfish-depth", type=int, default=30)
    args = parser.parse_args()

    if args.output.resolve() == args.base.resolve():
        raise SystemExit("Refusing to overwrite --base in place; choose a separate --output path")

    base_results = load_json(args.base)
    positions_data = load_json(args.positions)
    positions = positions_data["positions"] if isinstance(positions_data, dict) else positions_data
    merged, touched = merge_overlays(base_results, args.overlay_results)

    failures: list[str] = []
    for player_id in sorted(touched):
        player_data = merged[player_id]
        refresh_summary(player_id, player_data, positions, args.min_stockfish_depth)
        readiness = benchmark_result_readiness(
            player_data,
            positions,
            min_equal_positions=args.min_equal_positions,
            min_stockfish_depth=args.min_stockfish_depth,
        )
        if readiness.is_ready:
            print(f"ready: {player_id}")
        else:
            failures.append(f"{player_id}: {readiness.reason}")

    if failures:
        print("Not writing output; overlays are not production-ready:")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)

    args.output.write_text(json.dumps(merged, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
