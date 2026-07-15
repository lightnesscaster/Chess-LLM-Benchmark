#!/usr/bin/env python3
"""Audit full-game retry outcomes under the two-strikes-per-game policy.

This script is read-only with respect to Firestore. It can load raw result
documents from Firestore or a local JSON export and optionally save a local audit
artifact. It never calls a model or chess engine.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firebase_client import RESULTS_COLLECTION, get_firestore_client  # noqa: E402
from position_benchmark.retry_protocol import (  # noqa: E402
    CONDITIONAL_RETRY_PROTOCOL_VERSION,
    derive_game_side_retry_metrics,
)


def load_games(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load result documents from a local export or read-only Firestore stream."""
    if path is not None:
        data = json.loads(path.read_text())
        if isinstance(data, list):
            return {
                str(row.get("game_id") or index): row
                for index, row in enumerate(data)
            }
        return {str(key): value for key, value in data.items()}

    db = get_firestore_client()
    return {
        doc.id: doc.to_dict()
        for doc in db.collection(RESULTS_COLLECTION).stream()
    }


def audit_games(
    games: dict[str, dict[str, Any]],
    *,
    player_prefixes: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Aggregate exact or reconstructable retry evidence by player."""
    players: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "games": 0,
            "legal_moves": 0,
            "illegal_forfeits": 0,
            "illegal_responses": 0,
            "retry_attempts": 0,
            "retry_recoveries": 0,
            "retry_failures": 0,
            "retry_unknown": 0,
            "later_second_strikes": 0,
            "incomplete_evidence_games": 0,
        }
    )
    selected_games: list[dict[str, Any]] = []

    for game_id, game in games.items():
        game_selected = False
        game_retry: dict[str, Any] = {
            "game_id": game_id,
            "white_id": game.get("white_id"),
            "black_id": game.get("black_id"),
            "winner": game.get("winner"),
            "termination": game.get("termination"),
            "created_at": game.get("created_at"),
            "moves": game.get("moves"),
        }
        for side in ("white", "black"):
            player_id = str(game.get(f"{side}_id") or "")
            if not player_id:
                continue
            if player_prefixes and not any(
                player_id.startswith(prefix) for prefix in player_prefixes
            ):
                continue

            game_selected = True
            metrics = derive_game_side_retry_metrics(game, side)
            player = players[player_id]
            player["games"] += 1
            player["legal_moves"] += int(game.get(f"total_moves_{side}", 0) or 0)
            player_forfeited = (
                game.get("termination") == "forfeit_illegal_move"
                and game.get("winner") not in (side, "draw")
            )
            player["illegal_forfeits"] += int(player_forfeited)
            player["illegal_responses"] += int(
                game.get(f"illegal_moves_{side}", 0) or 0
            )
            for key in (
                "retry_attempts",
                "retry_recoveries",
                "retry_failures",
                "retry_unknown",
                "later_second_strikes",
            ):
                player[key] += int(metrics[key])
            if not metrics["evidence_available"]:
                player["incomplete_evidence_games"] += 1
            game_retry[side] = {
                "player_id": player_id,
                **metrics,
                "illegal_responses": int(
                    game.get(f"illegal_moves_{side}", 0) or 0
                ),
            }
        if game_selected:
            selected_games.append(game_retry)

    for metrics in players.values():
        known = metrics["retry_recoveries"] + metrics["retry_failures"]
        metrics["known_retry_outcomes"] = known
        metrics["first_attempt_illegals"] = (
            metrics["retry_attempts"] + metrics["later_second_strikes"]
        )
        metrics["first_attempt_turns"] = (
            metrics["legal_moves"] + metrics["illegal_forfeits"]
        )
        metrics["first_attempt_illegal_pct"] = (
            100.0
            * metrics["first_attempt_illegals"]
            / metrics["first_attempt_turns"]
            if metrics["first_attempt_turns"]
            else None
        )
        metrics["retry_recovery_pct"] = (
            100.0 * metrics["retry_recoveries"] / known if known else None
        )
        metrics["retry_failure_pct"] = (
            100.0 * metrics["retry_failures"] / known if known else None
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_version": CONDITIONAL_RETRY_PROTOCOL_VERSION,
        "policy": "second illegal response by a player anywhere in one game forfeits",
        "source_games": len(games),
        "selected_games": len(selected_games),
        "player_prefixes": list(player_prefixes),
        "players": dict(sorted(players.items())),
        "game_evidence": selected_games,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Local result JSON instead of Firestore")
    parser.add_argument(
        "--player-prefix",
        action="append",
        default=[],
        help="Only include player IDs starting with this value; repeat as needed",
    )
    parser.add_argument("--output", type=Path, help="Optional local JSON audit artifact")
    args = parser.parse_args()

    games = load_games(args.input)
    audit = audit_games(games, player_prefixes=tuple(args.player_prefix))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(audit, indent=2) + "\n")

    print(
        f"Audited {audit['selected_games']} selected games from "
        f"{audit['source_games']} result documents"
    )
    for player_id, metrics in audit["players"].items():
        known = metrics["known_retry_outcomes"]
        recovery = metrics["retry_recovery_pct"]
        recovery_text = "n/a" if recovery is None else f"{recovery:.1f}%"
        print(
            f"{player_id}: games={metrics['games']} "
            f"illegal={metrics['illegal_responses']} "
            f"retry={metrics['retry_recoveries']}/{known} recovered "
            f"({recovery_text}), later_second={metrics['later_second_strikes']} "
            f"unknown={metrics['retry_unknown']} "
            f"incomplete={metrics['incomplete_evidence_games']}"
        )
    if args.output is not None:
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
