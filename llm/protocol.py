"""Shared response protocol for LLM chess players."""

from __future__ import annotations


def parse_resignation(response_text: str | None) -> str | None:
    """Return the resignation action only for an exact, unambiguous response."""
    if isinstance(response_text, str) and response_text.strip().casefold() == "resign":
        return "resign"
    return None
