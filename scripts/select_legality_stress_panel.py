#!/usr/bin/env python3
"""Select a compact, family-balanced legality stress panel without target leakage."""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from position_benchmark.layout import (  # noqa: E402
    CORE_POSITIONS_PATH,
    CORE_RESULTS_PATH,
    repo_relative,
)
from position_benchmark.predictions import result_row_is_current  # noqa: E402
from scripts.analyze_puzzle_predictions import ANCHOR_IDS, model_family  # noqa: E402


DEFAULT_OUTPUT = Path("position_benchmark/candidates/legality_stress_6.json")
DEFAULT_GAME_AUDIT = Path(
    "position_benchmark/validation/2026-07-14-gpt56-game-retry-audit.json"
)
ENGINE_MARKERS = ("random-bot", "maia-", "eubos", "survival-bot", "stockfish")


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor
        while (
            end + 1 < len(ordered)
            and values[ordered[end + 1]] == values[ordered[cursor]]
        ):
            end += 1
        rank = (cursor + end) / 2.0 + 1.0
        for offset in range(cursor, end + 1):
            ranks[ordered[offset]] = rank
        cursor = end + 1
    return ranks


def _spearman(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    x = _ranks(left)
    y = _ranks(right)
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    denominator = math.sqrt(
        sum((a - mean_x) ** 2 for a in x)
        * sum((b - mean_y) ** 2 for b in y)
    )
    return numerator / denominator if denominator else None


def _eligible_models(
    results: dict[str, Any],
    position_count: int,
) -> dict[str, list[dict[str, Any]]]:
    eligible: dict[str, list[dict[str, Any]]] = {}
    for player_id, player_data in results.items():
        lowered = player_id.lower()
        if player_id in ANCHOR_IDS or any(marker in lowered for marker in ENGINE_MARKERS):
            continue
        rows = player_data.get("results") or []
        if len(rows) != position_count or not all(result_row_is_current(row) for row in rows):
            continue
        rows_by_idx = {row.get("position_idx"): row for row in rows}
        if set(rows_by_idx) != set(range(position_count)):
            continue
        eligible[player_id] = [rows_by_idx[index] for index in range(position_count)]
    return eligible


def select_panel(
    positions: list[dict[str, Any]],
    results: dict[str, Any],
    *,
    holdout_prefix: str,
    min_family_illegal_rate: float,
    live_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    """Select only from the non-holdout cohort, then report held-out diagnostics."""
    eligible = _eligible_models(results, len(positions))
    selection = {
        player_id: rows
        for player_id, rows in eligible.items()
        if not player_id.startswith(holdout_prefix)
    }
    holdout = {
        player_id: rows
        for player_id, rows in eligible.items()
        if player_id.startswith(holdout_prefix)
    }
    families: dict[str, list[str]] = defaultdict(list)
    for player_id in selection:
        families[model_family(player_id)].append(player_id)
    if not families:
        raise ValueError("No eligible non-holdout model families")

    position_scores: list[dict[str, Any]] = []
    for index, position in enumerate(positions):
        family_rates = [
            sum(not selection[player_id][index].get("is_legal", True) for player_id in members)
            / len(members)
            for members in families.values()
        ]
        raw_rate = sum(
            not rows[index].get("is_legal", True) for rows in selection.values()
        ) / len(selection)
        position_scores.append(
            {
                "position_idx": index,
                "position_id": position.get("position_id"),
                "family_balanced_illegal_rate": sum(family_rates) / len(family_rates),
                "raw_selection_illegal_rate": raw_rate,
            }
        )

    selected_scores = [
        score
        for score in position_scores
        if score["family_balanced_illegal_rate"] >= min_family_illegal_rate
    ]
    selected_scores.sort(
        key=lambda score: (
            -score["family_balanced_illegal_rate"],
            score["position_id"] or "",
        )
    )
    selected_indices = [score["position_idx"] for score in selected_scores]
    selected_positions = []
    for local_index, source_index in enumerate(selected_indices):
        position = deepcopy(positions[source_index])
        position["source_panel"] = position.get("panel", "core-equal")
        position["source_position_idx"] = source_index
        position["panel"] = "legality-stress"
        position["stress_position_idx"] = local_index
        selected_positions.append(position)

    validation: dict[str, Any] = {
        "holdout_prefix": holdout_prefix,
        "holdout_models": sorted(holdout),
    }
    if holdout and selected_indices:
        selected_illegals = sum(
            not rows[index].get("is_legal", True)
            for rows in holdout.values()
            for index in selected_indices
        )
        selected_attempts = len(holdout) * len(selected_indices)
        full_illegals = sum(
            not row.get("is_legal", True)
            for rows in holdout.values()
            for row in rows
        )
        full_attempts = len(holdout) * len(positions)
        validation.update(
            {
                "selected_illegals": selected_illegals,
                "selected_attempts": selected_attempts,
                "selected_illegal_pct": 100.0 * selected_illegals / selected_attempts,
                "full_core_illegals": full_illegals,
                "full_core_attempts": full_attempts,
                "full_core_illegal_pct": 100.0 * full_illegals / full_attempts,
                "illegal_incidence_lift": (
                    (selected_illegals / selected_attempts)
                    / (full_illegals / full_attempts)
                    if full_illegals
                    else None
                ),
            }
        )

        audit_players = (live_audit or {}).get("players", {})
        common = [player_id for player_id in sorted(holdout) if player_id in audit_players]
        if common:
            stress_rates = [
                100.0
                * sum(
                    not holdout[player_id][index].get("is_legal", True)
                    for index in selected_indices
                )
                / len(selected_indices)
                for player_id in common
            ]
            core_rates = [
                100.0
                * sum(not row.get("is_legal", True) for row in holdout[player_id])
                / len(positions)
                for player_id in common
            ]
            live_rates = [
                float(audit_players[player_id]["first_attempt_illegal_pct"])
                for player_id in common
            ]
            validation.update(
                {
                    "live_game_models": common,
                    "stress_vs_live_spearman": _spearman(stress_rates, live_rates),
                    "full_core_vs_live_spearman": _spearman(core_rates, live_rates),
                }
            )

    return {
        "metadata": {
            "panel": "legality-stress",
            "status": "research-candidate",
            "description": (
                "Compact high-illegality core subset for efficient production-retry "
                "measurement; not a replacement rating panel"
            ),
            "selection_policy": "pre-holdout-family-balanced-illegal-rate-v1",
            "selection_uses_holdout": False,
            "source_positions": repo_relative(CORE_POSITIONS_PATH),
            "source_results": repo_relative(CORE_RESULTS_PATH),
            "minimum_family_balanced_illegal_rate": min_family_illegal_rate,
            "selection_models": sorted(selection),
            "selection_families": {
                family: sorted(members) for family, members in sorted(families.items())
            },
            "position_count": len(selected_positions),
            "position_scores": selected_scores,
            "held_out_validation": validation,
        },
        "positions": selected_positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=Path, default=CORE_POSITIONS_PATH)
    parser.add_argument("--results", type=Path, default=CORE_RESULTS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--holdout-prefix", default="gpt-5.6-")
    parser.add_argument("--min-family-illegal-rate", type=float, default=0.25)
    parser.add_argument("--live-audit", type=Path, default=DEFAULT_GAME_AUDIT)
    args = parser.parse_args()

    positions_data = json.loads(args.positions.read_text())
    positions = positions_data["positions"]
    results = json.loads(args.results.read_text())
    live_audit = json.loads(args.live_audit.read_text()) if args.live_audit.exists() else None
    panel = select_panel(
        positions,
        results,
        holdout_prefix=args.holdout_prefix,
        min_family_illegal_rate=args.min_family_illegal_rate,
        live_audit=live_audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(panel, indent=2) + "\n")

    metadata = panel["metadata"]
    validation = metadata["held_out_validation"]
    print(
        f"Selected {metadata['position_count']} positions from "
        f"{len(metadata['selection_models'])} models / "
        f"{len(metadata['selection_families'])} families"
    )
    if validation.get("selected_illegal_pct") is not None:
        print(
            f"Held-out illegality: {validation['selected_illegal_pct']:.2f}% vs "
            f"{validation['full_core_illegal_pct']:.2f}% full core "
            f"({validation['illegal_incidence_lift']:.2f}x lift)"
        )
    if validation.get("stress_vs_live_spearman") is not None:
        print(
            f"Held-out Spearman vs live p: stress="
            f"{validation['stress_vs_live_spearman']:.3f}, full-core="
            f"{validation['full_core_vs_live_spearman']:.3f}"
        )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
