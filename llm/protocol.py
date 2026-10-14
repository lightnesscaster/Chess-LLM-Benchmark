"""Shared response protocol for LLM chess players."""

from __future__ import annotations

import re


def parse_draw_action(response_text: str | None) -> str | None:
    """Recognize only a complete draw action; board legality belongs to the game."""
    if not isinstance(response_text, str):
        return None
    text = response_text.strip().casefold()
    if text == "accept_draw":
        return text
    match = re.fullmatch(r"([a-h][1-8])([a-h][1-8])([qrbn]?) +offer_draw", text)
    if match and match[1] != match[2]:
        return f"{match[1]}{match[2]}{match[3]} offer_draw"
    return None


def split_draw_offer(action: str) -> tuple[str, bool]:
    """Split a canonical move with an attached offer, leaving other actions intact."""
    if action.endswith(" offer_draw") and parse_draw_action(action) == action:
        return action.removesuffix(" offer_draw"), True
    return action, False


def unparsed_response(response_text: str | None, limit: int) -> str:
    """Shorten debug output without turning ambiguous text into a draw action."""
    text = response_text.strip() if response_text else ""
    shortened = text[:limit]
    if parse_draw_action(shortened) and not parse_draw_action(text):
        return shortened.rstrip() + "..."
    return shortened


def parse_resignation(response_text: str | None) -> str | None:
    """Return the resignation action only for an exact, unambiguous response."""
    if isinstance(response_text, str) and response_text.strip().casefold() == "resign":
        return "resign"
    return None
