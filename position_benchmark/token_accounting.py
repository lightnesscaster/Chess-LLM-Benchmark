"""Canonical token accounting for position-benchmark result rows."""

from __future__ import annotations

from typing import Any, Iterable


def sum_result_row_tokens(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    """Return token totals attributable to the retained result rows."""
    prompt = 0
    completion = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt += int(row.get("prompt_tokens", 0) or 0)
        completion += int(row.get("completion_tokens", 0) or 0)
    return {"prompt": prompt, "completion": completion}


def refresh_result_token_usage(player_data: dict[str, Any]) -> dict[str, int]:
    """Replace file-level token totals with the sum of retained result rows."""
    usage = sum_result_row_tokens(player_data.get("results", []))
    player_data["token_usage"] = usage
    return usage
