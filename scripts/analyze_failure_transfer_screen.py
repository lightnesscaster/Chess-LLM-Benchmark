#!/usr/bin/env python3
"""Validate, merge, and analyze failure-transfer-screen-v1 result shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = (
    ROOT / "position_benchmark/candidates/failure_transfer_screen_v1/matrix.json"
)
DEFAULT_RESULTS = ROOT / "position_benchmark/results/failure_transfer_screen.json"
DEFAULT_ANALYSIS = (
    ROOT
    / "position_benchmark/validation/2026-07-15-failure-transfer-screen.json"
)
DEFAULT_SHORTLIST = (
    ROOT / "position_benchmark/candidates/failure_transfer_positive_3.json"
)
TARGET_FILES = {
    "gpt-5.6-luna (medium)": ROOT
    / "position_benchmark/candidates/failure_transfer_screen_v1/luna.json",
    "gpt-5.6-terra (low)": ROOT
    / "position_benchmark/candidates/failure_transfer_screen_v1/terra.json",
    "gpt-5.6-sol (high)": ROOT
    / "position_benchmark/candidates/failure_transfer_screen_v1/sol.json",
}
DEFAULT_SHARDS = [
    Path("/tmp/failure_transfer_luna_d30.json"),
    Path("/tmp/failure_transfer_terra_d30.json"),
    Path("/tmp/failure_transfer_sol_d30.json"),
]


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def wilson(successes: int, trials: int) -> list[float] | None:
    if not trials:
        return None
    z = 1.96
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / trials
            + z * z / (4 * trials * trials)
        )
        / denominator
    )
    return [100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half)]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cpls = [float(row["cpl"]) for row in rows]
    legal_cpls = [float(row["cpl"]) for row in rows if row["is_legal"]]
    illegals = sum(row["is_legal"] is False for row in rows)
    retry_rows = [row for row in rows if row.get("retry_attempted")]
    return {
        "attempts": len(rows),
        "illegals": illegals,
        "illegal_pct": 100.0 * illegals / len(rows) if rows else None,
        "illegal_wilson_95pct": wilson(illegals, len(rows)),
        "mean_cpl": statistics.mean(cpls) if cpls else None,
        "capped_5000_mean_cpl": (
            statistics.mean(min(value, 5000.0) for value in cpls) if cpls else None
        ),
        "median_cpl": statistics.median(cpls) if cpls else None,
        "legal_only_mean_cpl": statistics.mean(legal_cpls) if legal_cpls else None,
        "retry_attempts": len(retry_rows),
        "retry_recoveries": sum(row.get("retry_is_legal") is True for row in retry_rows),
        "retry_failures": sum(row.get("retry_is_legal") is False for row in retry_rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--shards", type=Path, nargs="+", default=DEFAULT_SHARDS)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--analysis-output", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--shortlist-output", type=Path, default=DEFAULT_SHORTLIST)
    args = parser.parse_args()

    matrix = load(args.matrix)
    position_metadata: dict[str, dict[str, Any]] = {}
    for player_id, path in TARGET_FILES.items():
        data = load(path)
        if data["metadata"]["target_player"] != player_id:
            raise ValueError(f"Target file mismatch for {player_id}")
        for position in data["positions"]:
            position_metadata[position["position_id"]] = position

    merged: dict[str, Any] = {}
    for shard_path in args.shards:
        shard = load(shard_path)
        for player_id, record in shard.items():
            if player_id in merged:
                raise ValueError(f"Duplicate shard player {player_id}")
            expected_ids = matrix["test_matrix"].get(player_id)
            if expected_ids is None:
                raise ValueError(f"Unexpected shard player {player_id}")
            rows = record.get("results", [])
            if [row.get("position_id") for row in rows] != expected_ids:
                raise ValueError(f"{player_id}: result IDs do not match frozen matrix")
            if int(record.get("summary", {}).get("positions_skipped", 0)) != 0:
                raise ValueError(f"{player_id}: incomplete shard")
            if any(int(row.get("stockfish_depth", 0)) != 30 for row in rows):
                raise ValueError(f"{player_id}: non-depth-30 result")
            merged[player_id] = record

    if set(merged) != set(matrix["test_matrix"]):
        raise ValueError("Result players do not match frozen matrix")

    annotated: dict[str, list[dict[str, Any]]] = {}
    all_rows: list[dict[str, Any]] = []
    pair_outcomes: list[dict[str, Any]] = []
    candidate_tests: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for player_id, record in merged.items():
        player_rows: list[dict[str, Any]] = []
        by_id = {row["position_id"]: row for row in record["results"]}
        for row in record["results"]:
            metadata = position_metadata[row["position_id"]]
            item = {
                **row,
                "target_player_id": player_id,
                "candidate_kind": (
                    "source_failure"
                    if row["position_id"].startswith("failure-")
                    else "matched_control"
                ),
                "source_base_model": metadata["base_model"],
                "source_player_id": metadata["player_id"],
                "source_failure_primary_class": metadata.get(
                    "source_failure_primary_class"
                ),
                "matched_failure_id": metadata.get("matched_failure_id"),
            }
            player_rows.append(item)
            all_rows.append(item)
            if item["candidate_kind"] == "source_failure":
                candidate_tests[item["position_id"]].append(item)
        annotated[player_id] = player_rows

        for failure_id in [
            position_id
            for position_id in matrix["test_matrix"][player_id]
            if position_id.startswith("failure-")
        ]:
            control_id = failure_id.replace("failure-", "control-")
            pair_outcomes.append(
                {
                    "target_player_id": player_id,
                    "failure_id": failure_id,
                    "failure_illegal": by_id[failure_id]["is_legal"] is False,
                    "control_illegal": by_id[control_id]["is_legal"] is False,
                }
            )

    failure_rows = [row for row in all_rows if row["candidate_kind"] == "source_failure"]
    control_rows = [row for row in all_rows if row["candidate_kind"] == "matched_control"]
    failure_only_discordant = sum(
        pair["failure_illegal"] and not pair["control_illegal"] for pair in pair_outcomes
    )
    control_only_discordant = sum(
        pair["control_illegal"] and not pair["failure_illegal"] for pair in pair_outcomes
    )
    discordant = failure_only_discordant + control_only_discordant
    one_sided_mcnemar = (
        sum(
            math.comb(discordant, count) * 0.5**discordant
            for count in range(failure_only_discordant, discordant + 1)
        )
        if discordant
        else 1.0
    )

    per_target = {
        player_id: {
            "failure_states": summarize(
                [row for row in rows if row["candidate_kind"] == "source_failure"]
            ),
            "matched_controls": summarize(
                [row for row in rows if row["candidate_kind"] == "matched_control"]
            ),
        }
        for player_id, rows in annotated.items()
    }
    per_source: dict[str, dict[str, Any]] = {}
    for source in ("luna", "terra", "sol"):
        per_source[source] = summarize(
            [row for row in failure_rows if row["source_base_model"] == source]
        )
    candidate_transfer = []
    for candidate_id, rows in sorted(candidate_tests.items()):
        metadata = position_metadata[candidate_id]
        summary = summarize(rows)
        candidate_transfer.append(
            {
                "candidate_id": candidate_id,
                "source_base_model": metadata["base_model"],
                "source_player_id": metadata["player_id"],
                "source_failure_primary_class": metadata.get(
                    "source_failure_primary_class"
                ),
                **summary,
                "target_players": [row["target_player_id"] for row in rows],
            }
        )

    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "screen_version": "failure-transfer-screen-v1",
        "stockfish_depth": 30,
        "production_effect": "none",
        "aggregate": {
            "failure_states": summarize(failure_rows),
            "matched_controls": summarize(control_rows),
            "illegal_rate_difference_points": (
                summarize(failure_rows)["illegal_pct"]
                - summarize(control_rows)["illegal_pct"]
            ),
            "matched_pairs": len(pair_outcomes),
            "failure_only_discordant": failure_only_discordant,
            "control_only_discordant": control_only_discordant,
            "one_sided_exact_mcnemar_p": one_sided_mcnemar,
        },
        "per_target": per_target,
        "per_source_base_model": per_source,
        "candidate_transfer": candidate_transfer,
        "pair_outcomes": pair_outcomes,
        "selection_warning": (
            "Candidates positive in this screen require validation on model families "
            "that supplied neither discovery nor selection evidence."
        ),
    }
    for player_id, record in merged.items():
        record["summary"]["screen_version"] = "failure-transfer-screen-v1"
        record["summary"]["production_effect"] = "none"
        record["summary"]["failure_states"] = per_target[player_id]["failure_states"]
        record["summary"]["matched_controls"] = per_target[player_id]["matched_controls"]

    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    args.results_output.write_text(json.dumps(merged, indent=2) + "\n")
    args.analysis_output.parent.mkdir(parents=True, exist_ok=True)
    args.analysis_output.write_text(json.dumps(analysis, indent=2) + "\n")
    positive_ids = [
        row["candidate_id"] for row in candidate_transfer if row["illegals"] > 0
    ]
    shortlist_positions: list[dict[str, Any]] = []
    for failure_id in positive_ids:
        for position_id in (
            failure_id,
            failure_id.replace("failure-", "control-"),
        ):
            position = dict(position_metadata[position_id])
            position["panel"] = "failure-transfer-positive-v1"
            position["panel_index"] = len(shortlist_positions)
            position["candidate_kind"] = (
                "source_failure"
                if position_id.startswith("failure-")
                else "matched_control"
            )
            shortlist_positions.append(position)
    shortlist = {
        "metadata": {
            "panel": "failure-transfer-positive-v1",
            "status": "research-candidate",
            "production_effect": "none",
            "selection_uses_gpt56_transfer_results": True,
            "selection_rule": "at-least-one-illegal-on-two-foreign-base-model-targets-v1",
            "source_screen": "failure-transfer-screen-v1",
            "source_analysis": str(args.analysis_output.relative_to(ROOT)),
            "failure_position_count": len(positive_ids),
            "matched_control_count": len(positive_ids),
            "position_count": len(shortlist_positions),
            "stockfish_depth": 30,
            "required_next_gate": "held-out-non-gpt56-model-families",
        },
        "positions": shortlist_positions,
    }
    args.shortlist_output.parent.mkdir(parents=True, exist_ok=True)
    args.shortlist_output.write_text(json.dumps(shortlist, indent=2) + "\n")
    print(f"Saved {args.results_output}")
    print(f"Saved {args.analysis_output}")
    print(f"Saved {args.shortlist_output}")


if __name__ == "__main__":
    main()
