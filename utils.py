"""Shared utility functions."""


def is_reasoning_model(player_id: str) -> bool:
    """
    Determine if a player is a reasoning model based on naming conventions.

    This is the single source of truth for reasoning model detection,
    used by both the benchmark scheduler and the web leaderboard.
    """
    player_lower = player_id.lower()
    return (
        '(thinking)' in player_lower or
        '(high)' in player_lower or
        '(medium)' in player_lower or
        '(minimal)' in player_lower or
        '-r1' in player_lower or
        'o3' in player_lower or
        'o4-mini' in player_lower or
        'o1' in player_lower or
        'gemini-3' in player_lower or
        'gemini-2.5-pro' in player_lower or
        'grok-4' in player_lower or
        '-thinking' in player_lower or
        'gpt-5-chat' in player_lower or
        'gpt-5.1-chat' in player_lower or
        'gpt-5.2-chat' in player_lower
    )
