#!/usr/bin/env python3
"""Combine multiple positions/results sources into one merged dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def normalize_positions(data: Any) -> list[dict[str, Any]]:
    return data["positions"] if isinstance(data, dict) else data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", action="append", required=True)
    parser.add_argument("--results", action="append", required=True)
    parser.add_argument("--labels", action="append", default=[])
    parser.add_argument("--output-positions", type=Path, required=True)
    parser.add_argument("--output-results", type=Path, required=True)
    args = parser.parse_args()

    if len(args.positions) != len(args.results):
        raise SystemExit("--positions and --results must have the same count")
    if args.labels and len(args.labels) != len(args.positions):
        raise SystemExit("--labels must be omitted or match the number of sources")

    merged_positions: list[dict[str, Any]] = []
    merged_results: dict[str, dict[str, Any]] = {}

    offset = 0
    for source_idx, (positions_path_str, results_path_str) in enumerate(zip(args.positions, args.results)):
        positions_path = Path(positions_path_str)
        results_path = Path(results_path_str)
        label = args.labels[source_idx] if args.labels else f"source_{source_idx + 1}"

        positions_data = load_json(positions_path)
        positions = normalize_positions(positions_data)
        results_data = load_json(results_path)

        for position in positions:
            item = dict(position)
            item["source_label"] = label
            merged_positions.append(item)

        for player_id, model_data in results_data.items():
            dest = merged_results.setdefault(player_id, {"results": []})
            for item in model_data.get("results", []):
                copied = dict(item)
                copied["position_idx"] = int(copied["position_idx"]) + offset
                dest["results"].append(copied)

        offset += len(positions)

    for player_id, model_data in merged_results.items():
        model_data["results"].sort(key=lambda item: int(item["position_idx"]))
        model_data["summary"] = {
            "player_id": player_id,
            "total_positions": len(model_data["results"]),
        }

    args.output_positions.write_text(
        json.dumps(
            {
                "metadata": {
                    "source_count": len(args.positions),
                    "position_count": len(merged_positions),
                },
                "positions": merged_positions,
            },
            indent=2,
        )
    )
    args.output_results.write_text(json.dumps(merged_results, indent=2))


if __name__ == "__main__":
    main()
