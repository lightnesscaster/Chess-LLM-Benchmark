#!/usr/bin/env python3
"""Replace failed continuation-probe rows with validated targeted reruns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from position_benchmark.predictions import stability_probe_readiness  # noqa: E402
from scripts.run_stability_probe import summarize_player  # noqa: E402


CONTRACT_FIELDS = (
    "player_id",
    "positions_file",
    "probe_plies",
    "score_depth",
    "stability_probe_version",
    "position_selection_policy",
    "selected_position_indices",
)


def merge_repair_rows(
    player_id: str,
    existing_record: dict[str, Any],
    repair_record: dict[str, Any],
) -> dict[str, Any]:
    """Return a canonical record with failed rows replaced by complete reruns."""
    existing_summary = existing_record.get("summary", {})
    repair_summary = repair_record.get("summary", {})
    if existing_summary.get("player_id") != player_id:
        raise ValueError(f"{player_id}: canonical player ID does not match")
    if repair_summary.get("player_id") != player_id:
        raise ValueError(f"{player_id}: repair player ID does not match")
    if repair_summary.get("position_selection_policy") != "explicit-indices":
        raise ValueError(f"{player_id}: repair must be an explicit-indices rerun")

    rows = list(existing_record.get("results") or [])
    repair_rows = list(repair_record.get("results") or [])
    if not rows or not repair_rows:
        raise ValueError(f"{player_id}: canonical and repair rows are required")

    row_by_index = {row.get("position_idx"): row for row in rows}
    if len(row_by_index) != len(rows):
        raise ValueError(f"{player_id}: canonical rows contain duplicate indices")

    expected_plies = int(existing_summary.get("probe_plies") or 0)
    seen: set[int] = set()
    for repair_row in repair_rows:
        position_idx = repair_row.get("position_idx")
        if position_idx in seen:
            raise ValueError(f"{player_id}: duplicate repair index {position_idx}")
        seen.add(position_idx)
        original = row_by_index.get(position_idx)
        if original is None:
            raise ValueError(f"{player_id}: repair index {position_idx} is not canonical")
        if (
            original.get("termination") != "api_error"
            and int(original.get("probe_plies_played") or 0) >= expected_plies
        ):
            raise ValueError(f"{player_id}: position {position_idx} is not failed or incomplete")
        if repair_row.get("position_id") != original.get("position_id"):
            raise ValueError(f"{player_id}: position ID mismatch at index {position_idx}")
        if repair_row.get("termination") == "api_error":
            raise ValueError(f"{player_id}: repair index {position_idx} still has an API error")
        if int(repair_row.get("probe_plies_played") or 0) != expected_plies:
            raise ValueError(f"{player_id}: repair index {position_idx} is incomplete")
        row_by_index[position_idx] = repair_row

    selected_indices = list(existing_summary.get("selected_position_indices") or [])
    merged_rows = [row_by_index[position_idx] for position_idx in selected_indices]
    merged_summary = summarize_player(merged_rows)
    for field in CONTRACT_FIELDS:
        merged_summary[field] = existing_summary.get(field)

    merged_record = dict(existing_record)
    merged_record["summary"] = merged_summary
    merged_record["results"] = merged_rows
    readiness = stability_probe_readiness(merged_record)
    if not readiness.is_ready:
        raise ValueError(f"{player_id}: merged record is not ready: {readiness.reason}")
    return merged_record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repair", type=Path, required=True)
    parser.add_argument("--player", required=True)
    args = parser.parse_args()

    output = json.loads(args.output.read_text())
    repair = json.loads(args.repair.read_text())
    if args.player not in output or args.player not in repair:
        raise ValueError(f"{args.player}: player missing from canonical or repair file")

    output[args.player] = merge_repair_rows(
        args.player,
        output[args.player],
        repair[args.player],
    )
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2) + "\n")
    temporary.replace(args.output)
    print(f"Repaired {len(repair[args.player]['results'])} row(s) for {args.player}")


if __name__ == "__main__":
    main()
