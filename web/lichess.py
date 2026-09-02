"""Lichess profile lookup for rated human challenge games."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from urllib.parse import quote

import requests


class LichessLookupError(ValueError):
    """Raised when a Rapid rating snapshot cannot be obtained."""


@dataclass(frozen=True)
class LichessRapidSnapshot:
    """Immutable values used to score one human challenge."""

    username: str
    rating: int
    rating_deviation: int
    provisional: bool

    def to_dict(self) -> dict:
        return asdict(self)


def fetch_rapid_snapshot(username, session=None) -> LichessRapidSnapshot:
    """Fetch the current Lichess Rapid rating and RD from a public profile."""
    normalized = str(username or "").strip()
    if not normalized or len(normalized) > 64 or any(char.isspace() for char in normalized):
        raise LichessLookupError("Enter a valid Lichess username.")

    http = session or requests
    try:
        response = http.get(
            f"https://lichess.org/api/user/{quote(normalized, safe='')}",
            headers={"Accept": "application/json", "User-Agent": "ChessBenchLLM/1.0"},
            timeout=8,
        )
    except requests.RequestException as error:
        raise LichessLookupError("Lichess could not be reached. Try again shortly.") from error

    if response.status_code == 404:
        raise LichessLookupError("That Lichess account was not found.")
    if response.status_code != 200:
        raise LichessLookupError("Lichess could not verify that account. Try again shortly.")
    try:
        payload = response.json()
    except (TypeError, ValueError) as error:
        raise LichessLookupError("Lichess returned an invalid profile.") from error

    if not isinstance(payload, dict):
        raise LichessLookupError("Lichess returned an invalid profile.")
    perfs = payload.get("perfs")
    rapid = perfs.get("rapid") if isinstance(perfs, dict) else None
    if not isinstance(rapid, dict):
        raise LichessLookupError("This Lichess account has no Rapid rating.")

    rating = rapid.get("rating")
    rating_deviation = rapid.get("rd")
    canonical_username = payload.get("username")
    if (
        not isinstance(canonical_username, str)
        or not canonical_username.strip()
        or isinstance(rating, bool)
        or not isinstance(rating, (int, float))
        or isinstance(rating_deviation, bool)
        or not isinstance(rating_deviation, (int, float))
        or not 0 <= rating <= 4000
        or not 0 < rating_deviation <= 500
    ):
        raise LichessLookupError("Lichess did not return a valid Rapid rating and RD.")

    return LichessRapidSnapshot(
        username=canonical_username.strip(),
        rating=round(rating),
        rating_deviation=round(rating_deviation),
        provisional=bool(rapid.get("prov", False)),
    )
