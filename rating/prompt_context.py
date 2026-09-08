"""Read-only, labeled rating snapshots for game prompts."""


def rating_snapshot(store, player_id: str) -> dict | None:
    """Return a known rating, never create a default rating for prompt context."""
    if not store.has_player(player_id):
        return None
    rating = store.get(player_id)
    anchor = store.is_anchor(player_id)
    return {
        "rating": rating.rating,
        "rating_deviation": rating.rating_deviation,
        "source": "ChessBench fixed anchor" if anchor else "ChessBench",
        "provisional": not anchor and (rating.games_played == 0 or rating.rating_deviation > 110),
    }
