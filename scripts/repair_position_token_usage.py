#!/usr/bin/env python3
"""Audit or repair position-result token totals from retained result rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from position_benchmark.token_accounting import sum_result_row_tokens  # noqa: E402


def audit_file(path: Path, *, repair: bool) -> int:
    """Report mismatches in one result file and optionally repair them."""
    with path.open() as handle:
        data: Any = json.load(handle)
    if not isinstance(data, dict):
        return 0

    mismatches = 0
    changed = False
    for player_id, player_data in data.items():
        if not isinstance(player_data, dict) or not isinstance(player_data.get("results"), list):
            continue
        if not player_data["results"] or not all(
            isinstance(row, dict) for row in player_data["results"]
        ):
            continue
        expected = sum_result_row_tokens(player_data["results"])
        actual = player_data.get("token_usage") or {"prompt": 0, "completion": 0}
        actual_pair = (int(actual.get("prompt", 0) or 0), int(actual.get("completion", 0) or 0))
        expected_pair = (expected["prompt"], expected["completion"])
        if actual_pair == expected_pair:
            continue
        mismatches += 1
        print(f"{path}: {player_id}: stored={actual_pair}, rows={expected_pair}")
        if repair:
            player_data["token_usage"] = expected
            changed = True

    if changed:
        path.write_text(json.dumps(data, indent=2) + "\n")
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args()

    mismatches = sum(audit_file(path, repair=args.repair) for path in args.paths)
    if mismatches and not args.repair:
        print(f"Found {mismatches} token-usage mismatches")
        return 1
    if args.repair:
        print(f"Repaired {mismatches} token-usage mismatches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
