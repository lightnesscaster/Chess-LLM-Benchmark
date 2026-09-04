#!/usr/bin/env python3
"""Compare protocol-sequence-v1 with static panels, prior probes, and live games."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORE = ROOT / "position_benchmark/results/core.json"
DEFAULT_GAME_LIKE = ROOT / "position_benchmark/results/game_like.json"
DEFAULT_PRIOR = ROOT / "position_benchmark/results/stability.json"
DEFAULT_SEQUENCE = ROOT / "position_benchmark/results/protocol_sequence.json"
DEFAULT_LIVE = (
    ROOT
    / "position_benchmark/validation/2026-07-15-gpt56-illegal-forensics.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "position_benchmark/validation/2026-07-15-protocol-sequence-pilot.json"
)


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def rate(illegals: int, attempts: int) -> dict[str, Any]:
    return {
        "attempts": attempts,
        "illegals": illegals,
        "illegal_pct": 100.0 * illegals / attempts if attempts else None,
        "wilson_95pct": wilson_interval(illegals, attempts),
    }


def wilson_interval(successes: int, trials: int) -> list[float] | None:
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


def static_rate(record: dict[str, Any]) -> dict[str, Any]:
    rows = record.get("results", [])
    illegals = sum(row.get("is_legal") is False for row in rows)
    return rate(illegals, len(rows))


def continuation_rate(record: dict[str, Any]) -> dict[str, Any]:
    rows = record.get("results", [])
    illegals = sum(
        len(
            {
                int(detail["move_number"])
                for detail in row.get("illegal_move_details", [])
                if detail.get("move_number") is not None
            }
        )
        for row in rows
    )
    attempts = sum(
        int(row.get("model_legal_moves", 0)) + int(bool(row.get("model_forfeited")))
        for row in rows
    )
    return rate(illegals, attempts)


def average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sequence_cpl(record: dict[str, Any]) -> dict[str, Any]:
    scores = [
        score
        for row in record.get("results", [])
        for score in row.get("model_move_scores", [])
    ]
    early = [float(score["cpl"]) for score in scores if score["model_turn_index"] <= 4]
    late = [float(score["cpl"]) for score in scores if score["model_turn_index"] >= 5]
    all_cpl = [float(score["cpl"]) for score in scores]
    return {
        "scored_moves": len(all_cpl),
        "raw_mean_cpl": average(all_cpl),
        "capped_5000_mean_cpl": average([min(value, 5000.0) for value in all_cpl]),
        "turns_1_4": {
            "moves": len(early),
            "raw_mean_cpl": average(early),
            "capped_5000_mean_cpl": average(
                [min(value, 5000.0) for value in early]
            ),
        },
        "turns_5_8": {
            "moves": len(late),
            "raw_mean_cpl": average(late),
            "capped_5000_mean_cpl": average(
                [min(value, 5000.0) for value in late]
            ),
        },
        "median_cpl": record["summary"].get("model_median_cpl"),
        "p90_cpl": record["summary"].get("model_p90_cpl"),
        "catastrophe_1000cp_pct": record["summary"].get(
            "model_1000cp_catastrophe_pct"
        ),
        "random_opponent_mean_cpl": record["summary"].get("opponent_avg_cpl"),
    }


def failure_events(player_id: str, record: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in record.get("results", []):
        pre_moves = int(row.get("pre_moves", 0))
        for detail in row.get("illegal_move_details", []):
            move_number = int(detail["move_number"])
            continuation_turn = (move_number - (pre_moves + 1)) // 2 + 1
            output.append(
                {
                    "source_player_id": player_id,
                    "starting_position_id": row.get("position_id"),
                    "starting_position_idx": row.get("position_idx"),
                    "continuation_turn": continuation_turn,
                    "prospective_ply": move_number,
                    "fen": detail.get("fen"),
                    "parsed_move": detail.get("parsed_move"),
                    "raw_response": detail.get("raw_response"),
                }
            )
    return output


def aggregate_rates(players: dict[str, dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for source in ("core", "game_like", "prior_continuation", "protocol_sequence", "live"):
        attempts = sum(player[source]["attempts"] for player in players.values())
        illegals = sum(player[source]["illegals"] for player in players.values())
        output[source] = rate(illegals, attempts)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, default=DEFAULT_CORE)
    parser.add_argument("--game-like", type=Path, default=DEFAULT_GAME_LIKE)
    parser.add_argument("--prior", type=Path, default=DEFAULT_PRIOR)
    parser.add_argument("--sequence", type=Path, default=DEFAULT_SEQUENCE)
    parser.add_argument("--live", type=Path, default=DEFAULT_LIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    core = load(args.core)
    game_like = load(args.game_like)
    prior = load(args.prior)
    sequence = load(args.sequence)
    live = load(args.live)
    live_rates = {
        row["player_id"]: rate(int(row["illegals"]), int(row["attempts"]))
        for row in live["rates"]["player_id"]
    }

    players: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for player_id, record in sequence.items():
        players[player_id] = {
            "core": static_rate(core[player_id]),
            "game_like": static_rate(game_like[player_id]),
            "prior_continuation": continuation_rate(prior[player_id]),
            "protocol_sequence": continuation_rate(record),
            "live": live_rates[player_id],
            "protocol_sequence_cpl": sequence_cpl(record),
            "protocol_sequence_forfeits": record["summary"].get("model_forfeits"),
            "protocol_sequence_retry": record["summary"].get("conditional_retry"),
        }
        failures.extend(failure_events(player_id, record))

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "protocol-sequence-v1-pilot-comparison-v1",
        "players": players,
        "aggregate_rates": aggregate_rates(players),
        "failure_events": failures,
        "failure_event_selection_warning": (
            "Discovery events are candidate-pool inputs only. Validate transfer on "
            "models that did not supply each event before selecting a frozen panel."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
