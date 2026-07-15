"""Shared metadata and aggregation for production-style illegal-move retries."""

from __future__ import annotations

from typing import Any


CONDITIONAL_RETRY_PROTOCOL_VERSION = "production-game-retry-v1"


def derive_two_strike_retry_metrics(
    *,
    illegal_attempts: int,
    illegal_events: list[dict[str, Any]],
    saved_legal_plies: int,
) -> dict[str, int | bool]:
    """Reconstruct retry evidence for one side under the two-strike policy."""
    if illegal_attempts <= 0:
        return {
            "retry_attempts": 0,
            "retry_recoveries": 0,
            "retry_failures": 0,
            "retry_unknown": 0,
            "later_second_strikes": 0,
            "evidence_available": True,
        }
    if not illegal_events:
        return {
            "retry_attempts": 0,
            "retry_recoveries": 0,
            "retry_failures": 0,
            "retry_unknown": 0,
            "later_second_strikes": 0,
            "evidence_available": False,
        }

    try:
        first_move = int(illegal_events[0].get("move_number"))
    except (TypeError, ValueError):
        first_move = None

    recovered = int(first_move is not None and saved_legal_plies >= first_move)
    failed = 0
    later_second_strikes = 0
    if len(illegal_events) >= 2:
        try:
            second_move = int(illegal_events[1].get("move_number"))
        except (TypeError, ValueError):
            second_move = None
        if first_move is not None and second_move == first_move:
            recovered = 0
            failed = 1
        elif first_move is not None and second_move is not None:
            recovered = 1
            later_second_strikes = 1

    return {
        "retry_attempts": 1,
        "retry_recoveries": recovered,
        "retry_failures": failed,
        "retry_unknown": 1 - recovered - failed,
        "later_second_strikes": later_second_strikes,
        "evidence_available": True,
    }


def derive_game_side_retry_metrics(
    game: dict[str, Any],
    side: str,
) -> dict[str, int | bool]:
    """Extract direct or legacy-inferred retry metrics for one game side."""
    direct_key = f"retry_attempts_{side}"
    if direct_key in game:
        attempts = int(game.get(direct_key, 0) or 0)
        recoveries = int(game.get(f"retry_recoveries_{side}", 0) or 0)
        failures = int(game.get(f"retry_failures_{side}", 0) or 0)
        unknown = int(
            game.get(
                f"retry_unknown_{side}",
                attempts - recoveries - failures,
            )
            or 0
        )
        return {
            "retry_attempts": attempts,
            "retry_recoveries": recoveries,
            "retry_failures": failures,
            "retry_unknown": unknown,
            "later_second_strikes": int(
                int(game.get(f"illegal_moves_{side}", 0) or 0) >= 2
                and recoveries > 0
                and failures == 0
            ),
            "evidence_available": True,
        }

    events = [
        event
        for event in (game.get("illegal_move_details") or [])
        if event.get("side") == side
    ]
    return derive_two_strike_retry_metrics(
        illegal_attempts=int(game.get(f"illegal_moves_{side}", 0) or 0),
        illegal_events=events,
        saved_legal_plies=int(game.get("moves", 0) or 0),
    )


def conditional_retry_summary(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Summarize rows measured with the current conditional retry protocol."""
    measured = [
        row
        for row in rows
        if row.get("conditional_retry_protocol_version")
        == CONDITIONAL_RETRY_PROTOCOL_VERSION
    ]
    if not measured:
        return None

    initial_legal = sum(1 for row in measured if row.get("is_legal") is True)
    initial_illegal = len(measured) - initial_legal
    attempted = [row for row in measured if row.get("retry_attempted") is True]
    recovered = sum(1 for row in attempted if row.get("retry_is_legal") is True)
    failed = sum(1 for row in attempted if row.get("retry_is_legal") is False)
    unknown = len(attempted) - recovered - failed
    missing = max(0, initial_illegal - len(attempted))

    return {
        "protocol_version": CONDITIONAL_RETRY_PROTOCOL_VERSION,
        "measured_positions": len(measured),
        "initial_illegal_moves": initial_illegal,
        "retry_attempts": len(attempted),
        "retry_recoveries": recovered,
        "retry_failures": failed,
        "retry_unknown": unknown,
        "initial_illegals_without_retry": missing,
        "recovery_pct": 100.0 * recovered / len(attempted) if attempted else None,
        "conditional_failure_pct": (
            100.0 * failed / len(attempted) if attempted else None
        ),
        "post_retry_legal_moves": initial_legal + recovered,
        "post_retry_legal_pct": 100.0 * (initial_legal + recovered) / len(measured),
        "total_model_calls": len(measured) + len(attempted),
    }


def attach_conditional_retry_summary(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> None:
    """Attach retry evidence when rows contain current-protocol measurements."""
    retry = conditional_retry_summary(rows)
    if retry is not None:
        summary["conditional_retry"] = retry
