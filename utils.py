"""Shared utility functions."""


def is_reasoning_model(player_id: str) -> bool:
    """
    Determine if a player is a reasoning model based on naming conventions.

    This is the single source of truth for reasoning model detection,
    used by both the benchmark scheduler and the web leaderboard.

    NOTE: This list must be manually updated as new reasoning models are released.
    """
    if not player_id:
        return False

    player_lower = player_id.lower()

    # Check for OpenAI reasoning models (o1, o3, o4-mini) at word boundaries
    # to avoid matching version strings like "v0.1"
    if player_lower in ("o1", "o3") or player_lower.startswith("o1-") or \
       player_lower.startswith("o3-") or player_lower.startswith("o4-mini"):
        return True

    reasoning_indicators = [
        "(thinking)",
        "(high)",
        "(medium)",
        "(minimal)",
        "-r1",          # DeepSeek R1 models
        "gemini-3",     # Gemini 3.x models
        "gemini-2.5-pro",  # Gemini 2.5 Pro (reasoning by default)
        "grok-4",       # Grok 4.x models
        "-thinking",    # Explicit thinking suffix
        "gpt-5-chat",   # GPT-5 chat (reasoning variant)
        "gpt-5.1-chat",
        "gpt-5.2-chat",
    ]
    return any(indicator in player_lower for indicator in reasoning_indicators)
