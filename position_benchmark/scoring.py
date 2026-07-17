"""Shared scoring rules for position-benchmark results."""

from __future__ import annotations


ILLEGAL_MOVE_EVAL = -5000


def illegal_move_cpl(eval_before: float) -> float:
    """Score one illegal strike without rewarding positions already below -5000."""
    return max(0, eval_before - ILLEGAL_MOVE_EVAL)
