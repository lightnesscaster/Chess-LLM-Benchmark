#!/usr/bin/env python3
"""Validate and merge completed continuation-probe shard files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from position_benchmark.predictions import stability_probe_readiness  # noqa: E402


def validate_record(
    player_id: str,
    record: dict[str, Any],
    *,
    expected_indices: list[int],
) -> None:
    """Raise ValueError unless one shard record satisfies the production contract."""
    readiness = stability_probe_readiness(record)
    if not readiness.is_ready:
        raise ValueError(f"{player_id}: {readiness.reason}")

    rows = record.get("results", [])
    row_indices = [row.get("position_idx") for row in rows]
    summary_indices = record.get("summary", {}).get("selected_position_indices")
    if row_indices != expected_indices:
        raise ValueError(f"{player_id}: unexpected row indices {row_indices}")
    if summary_indices != expected_indices:
        raise ValueError(f"{player_id}: unexpected summary indices {summary_indices}")


def merge_shards(
    existing: dict[str, Any],
    shards: list[dict[str, Any]],
    *,
    expected_indices: list[int],
) -> tuple[dict[str, Any], list[str]]:
    """Return existing results with validated shard records replaced by player ID."""
    merged = dict(existing)
    seen: set[str] = set()
    for shard in shards:
        for player_id, record in shard.items():
            if player_id in seen:
                raise ValueError(f"duplicate player across shards: {player_id}")
            if not isinstance(record, dict):
                raise ValueError(f"{player_id}: result record is not an object")
            validate_record(player_id, record, expected_indices=expected_indices)
            seen.add(player_id)
            merged[player_id] = record
    return merged, sorted(seen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--expected-indices", type=int, nargs="+", required=True)
    args = parser.parse_args()

    existing = json.loads(args.output.read_text()) if args.output.exists() else {}
    shards = [json.loads(path.read_text()) for path in args.inputs]
    merged, player_ids = merge_shards(
        existing,
        shards,
        expected_indices=args.expected_indices,
    )

    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(merged, indent=2) + "\n")
    temporary.replace(args.output)
    print(f"Merged {len(player_ids)} validated player records into {args.output}")
    for player_id in player_ids:
        print(f"  {player_id}")


if __name__ == "__main__":
    main()
