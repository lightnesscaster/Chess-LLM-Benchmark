"""
Timeline data for LLM Chess Benchmark.

Builds the payload behind the timeline page: every qualifying model at its
release date, plus the sequence of record-setters (the strongest model
released so far at each point in time). The chart itself is drawn client-side
by ``web/static/js/timeline.js``.
"""

from datetime import datetime, timezone
from typing import Any

from utils import is_reasoning_model
from model_metadata import load_publish_dates

# Labs that get their own color on the chart. A scatter compares every pair of
# colors, and only three hues stay distinguishable for color-blind readers, so
# every other lab shares a neutral gray and is named in the tooltip instead.
FEATURED_LABS = ("google", "openai", "anthropic")

LAB_NAMES = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "deepseek": "DeepSeek",
    "meta-llama": "Meta",
    "x-ai": "xAI",
    "mistralai": "Mistral",
    "qwen": "Qwen",
    "moonshotai": "Moonshot",
    "z-ai": "Z.ai",
    "xiaomi": "Xiaomi",
    "kwaipilot": "Kwaipilot",
}

# Rough skill bands on the Lichess classical scale: (lower bound, name).
SKILL_TIERS = [
    (500, "Novice"),
    (1000, "Beginner"),
    (1500, "Intermediate"),
    (1800, "Advanced"),
    (2100, "Expert"),
]

MAX_RATING_DEVIATION = 150
MIN_RATING = -500


def get_provider(model_id: str) -> str:
    """Extract provider from model_id (e.g., 'anthropic/claude-3.5-sonnet' -> 'anthropic')."""
    if "/" in model_id:
        return model_id.split("/")[0]
    return "unknown"


def _lab_name(provider: str) -> str:
    return LAB_NAMES.get(provider) or provider.replace("-", " ").title()


def _utc(timestamp: float) -> datetime:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def _held_text(days: int) -> str:
    if days < 60:
        return f"{days} day{'' if days == 1 else 's'}"
    months = round(days / 30.44)
    if months < 24:
        return f"{months} months"
    years = months / 12
    return f"{years:.1f}".rstrip("0").rstrip(".") + " years"


def build_timeline_data(
    leaderboard_data: list[dict[str, Any]],
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    Build the timeline payload from leaderboard entries.

    A model is plotted when it has a release date, a settled rating
    (RD <= 150, rating >= -500) and was the strongest model its lab had
    released in its category (reasoning or not) at the time.

    Returns a JSON-serializable dict with ``points``, ``records``, ``tiers``,
    ``labs`` and ``now`` (epoch seconds).
    """
    now = now or datetime.now(timezone.utc)
    providers = {
        player_id: get_provider(info.get("model_id", ""))
        for player_id, info in load_publish_dates().items()
    }

    candidates = []
    for entry in leaderboard_data:
        if (
            not entry.get("publish_timestamp")
            or entry.get("is_anchor")
            or entry.get("rating_deviation", 350) > MAX_RATING_DEVIATION
            or entry.get("rating", 0) < MIN_RATING
        ):
            continue
        candidates.append({
            "entry": entry,
            "provider": providers.get(entry["player_id"], "unknown"),
            "reasoning": is_reasoning_model(entry["player_id"]),
        })

    # Keep a model only if no earlier release from the same lab, in the same
    # category, is rated higher.
    def was_lab_best_at_release(candidate: dict) -> bool:
        entry = candidate["entry"]
        return not any(
            other["provider"] == candidate["provider"]
            and other["reasoning"] == candidate["reasoning"]
            and other["entry"]["publish_timestamp"] < entry["publish_timestamp"]
            and other["entry"]["rating"] > entry["rating"]
            for other in candidates
        )

    shown = [c for c in candidates if was_lab_best_at_release(c)]
    # Same-day releases: strongest first, so only the day's best can set a record.
    shown.sort(key=lambda c: (c["entry"]["publish_timestamp"], -c["entry"]["rating"]))

    points = []
    records: list[dict[str, Any]] = []
    for candidate in shown:
        entry = candidate["entry"]
        provider = candidate["provider"]
        timestamp = entry["publish_timestamp"]
        point = {
            "id": entry["player_id"],
            "rating": entry["rating"],
            "low": entry.get("confidence_low"),
            "high": entry.get("confidence_high"),
            "games": entry.get("games_played"),
            "ts": timestamp,
            "lab": provider if provider in FEATURED_LABS else "other",
            "lab_name": _lab_name(provider),
            "reasoning": candidate["reasoning"],
            "record": False,
        }
        points.append(point)

        if records and entry["rating"] <= records[-1]["rating"]:
            continue
        if records and _utc(records[-1]["ts"]).date() == _utc(timestamp).date():
            continue  # a stronger same-day sibling already took the record
        point["record"] = True
        records.append({
            "id": point["id"],
            "rating": point["rating"],
            "ts": timestamp,
            "lab": point["lab"],
            "lab_name": point["lab_name"],
            "date": _utc(timestamp).strftime("%b %-d, %Y"),
            "month": _utc(timestamp).strftime("%b %Y"),
            "gain": point["rating"] - records[-1]["rating"] if records else None,
        })

    for record, successor in zip(records, records[1:] + [None]):
        end = _utc(successor["ts"]) if successor else now
        record["current"] = successor is None
        record["held"] = _held_text(max((end - _utc(record["ts"])).days, 0))

    return {
        "points": points,
        "records": records,
        "tiers": [{"from": lower, "name": name} for lower, name in SKILL_TIERS],
        "labs": [
            {"key": lab, "name": _lab_name(lab)} for lab in FEATURED_LABS
        ] + [{"key": "other", "name": "Other labs"}],
        "now": now.timestamp(),
    }
