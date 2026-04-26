#!/usr/bin/env python3
"""Build a small non-opening screening set from a PGN-mined position pool."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TARGETS = {
    "quiet_equal": 12,
    "tactical_equal": 12,
    "advantage_conversion": 12,
    "defense": 12,
}


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def parse_targets(value: str | None) -> dict[str, int]:
    if not value:
        return dict(DEFAULT_TARGETS)
    targets: dict[str, int] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        name, raw_count = part.split("=", 1)
        targets[name.strip()] = int(raw_count)
    return targets


def opening_key(position: dict[str, Any], plies: int) -> str:
    history = position.get("move_history") or []
    return " ".join(history[:plies])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("position_benchmark/regan_lite_positions.json"))
    parser.add_argument("--output", type=Path, default=Path("position_benchmark/nonopening_screening_positions.json"))
    parser.add_argument("--targets", default=None)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max-same-opening", type=int, default=2)
    parser.add_argument("--max-same-game", type=int, default=1)
    parser.add_argument("--opening-plies", type=int, default=8)
    args = parser.parse_args()

    data = load_json(args.input)
    positions = data["positions"] if isinstance(data, dict) else data
    targets = parse_targets(args.targets)

    grouped: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in targets}
    for position in positions:
        bucket = str(position.get("regan_bucket") or position.get("bucket") or "")
        if bucket in grouped:
            grouped[bucket].append(dict(position))

    rng = random.Random(args.seed)
    selected: list[dict[str, Any]] = []
    selected_counts = {bucket: 0 for bucket in targets}
    opening_counts: Counter[str] = Counter()
    game_counts: Counter[str] = Counter()

    for bucket, bucket_positions in grouped.items():
        rng.shuffle(bucket_positions)
        for position in bucket_positions:
            if selected_counts[bucket] >= targets[bucket]:
                break
            open_key = opening_key(position, args.opening_plies)
            game_id = str(position.get("game_id") or "")
            if opening_counts[open_key] >= args.max_same_opening:
                continue
            if game_id and game_counts[game_id] >= args.max_same_game:
                continue
            position["screening_bucket"] = bucket
            position["type"] = "equal"
            selected.append(position)
            selected_counts[bucket] += 1
            opening_counts[open_key] += 1
            if game_id:
                game_counts[game_id] += 1

    selected.sort(key=lambda item: (item.get("screening_bucket", ""), item.get("move_number", 0), item.get("game_id", "")))
    for idx, position in enumerate(selected):
        position["position_id"] = f"screen-{idx:04d}"

    metadata = {
        "description": "Non-opening screening set sampled from PGN-mined Regan-lite positions",
        "source": str(args.input),
        "targets": targets,
        "selected_counts": selected_counts,
        "positions": len(selected),
        "seed": args.seed,
        "max_same_opening": args.max_same_opening,
        "max_same_game": args.max_same_game,
        "opening_plies": args.opening_plies,
    }
    args.output.write_text(json.dumps({"metadata": metadata, "positions": selected}, indent=2))
    print(f"Wrote {len(selected)} screening positions to {args.output}")
    print(f"Selected counts: {selected_counts}")


if __name__ == "__main__":
    main()
