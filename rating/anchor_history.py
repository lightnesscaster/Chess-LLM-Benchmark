"""Resolve fixed engine ratings without retroactively changing old games."""

from datetime import datetime, timezone


def _timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def anchor_rating_at(engine_config: dict, created_at: str) -> float:
    """Use each historical rating strictly before its UTC cutoff, current after."""
    history = engine_config.get("rating_history", [])
    if history:
        # Unknown dates must fail closed, not silently reprice historical games.
        game_time = _timestamp(created_at)
        for entry in sorted(history, key=lambda item: _timestamp(item["before"])):
            if game_time < _timestamp(entry["before"]):
                return float(entry["rating"])
    return float(engine_config["rating"])
