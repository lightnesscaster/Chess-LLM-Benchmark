"""
Rating storage and persistence.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Set

from .glicko2 import PlayerRating, Glicko2System


class RatingStore:
    """
    Stores and manages player ratings.

    Uses a JSON file for persistence.
    """

    def __init__(self, path: str = "data/ratings.json", anchor_ids: Set[str] = None):
        """
        Initialize the rating store.

        Args:
            path: Path to the ratings JSON file
            anchor_ids: Set of player IDs that are anchors (fixed ratings)
        """
        self.path = Path(path)
        self.anchor_ids = anchor_ids or set()
        self._ratings: Dict[str, PlayerRating] = {}
        self._load()

    def _load(self) -> None:
        """Load ratings from file."""
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            for player_id, rating_data in data.items():
                player_rating = PlayerRating.from_dict(rating_data)
                # Enforce rating floor on loaded ratings (anchors exempt)
                if player_id not in self.anchor_ids:
                    if player_rating.rating < Glicko2System.RATING_FLOOR:
                        player_rating.rating = Glicko2System.RATING_FLOOR
                self._ratings[player_id] = player_rating

    def _save(self) -> None:
        """Save ratings to file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            player_id: rating.to_dict()
            for player_id, rating in self._ratings.items()
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def get(self, player_id: str) -> PlayerRating:
        """
        Get a player's rating, creating default if not exists.

        Args:
            player_id: The player ID

        Returns:
            PlayerRating object
        """
        if player_id not in self._ratings:
            self._ratings[player_id] = PlayerRating(player_id=player_id)
        return self._ratings[player_id]

    def set(self, rating: PlayerRating) -> None:
        """
        Set a player's rating.

        Args:
            rating: The PlayerRating to store
        """
        self._ratings[rating.player_id] = rating
        self._save()

    def is_anchor(self, player_id: str) -> bool:
        """Check if a player is an anchor with fixed rating."""
        return player_id in self.anchor_ids

    def set_anchor(
        self,
        player_id: str,
        rating: float,
        rating_deviation: float = 50.0,
    ) -> None:
        """
        Set an anchor player with fixed rating.

        Args:
            player_id: The anchor player ID
            rating: The fixed rating
            rating_deviation: Low RD for anchors (default 50)
        """
        self.anchor_ids.add(player_id)
        self._ratings[player_id] = PlayerRating(
            player_id=player_id,
            rating=rating,
            rating_deviation=rating_deviation,
            volatility=0.03,  # Low volatility for anchors
            games_played=0,
        )
        self._save()

    def get_all(self) -> Dict[str, PlayerRating]:
        """Get all player ratings."""
        return dict(self._ratings)

    def get_sorted_ratings(self, min_games: int = 0) -> list[PlayerRating]:
        """
        Get ratings sorted by rating (descending).

        Args:
            min_games: Minimum games played to include

        Returns:
            List of PlayerRating sorted by rating
        """
        ratings = [
            r for r in self._ratings.values()
            if r.games_played >= min_games
        ]
        return sorted(ratings, key=lambda r: r.rating, reverse=True)

    def reset(self) -> None:
        """Reset all ratings (keeps anchors)."""
        anchors = {
            pid: self._ratings[pid]
            for pid in self.anchor_ids
            if pid in self._ratings
        }
        self._ratings = anchors
        self._save()
