"""
Rating storage and persistence.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Set

from .glicko2 import PlayerRating, Glicko2System

logger = logging.getLogger(__name__)

# Module-level cache for Firestore data
_firestore_cache: Dict[str, "PlayerRating"] = {}
_firestore_cache_time: float = 0
_FIRESTORE_CACHE_TTL = 300  # 5 minutes


class RatingStore:
    """
    Stores and manages player ratings.

    Supports both local JSON file and Firestore backends.
    """

    def __init__(
        self,
        path: str = "data/ratings.json",
        anchor_ids: Set[str] = None,
        use_firestore: bool = None,
    ):
        """
        Initialize the rating store.

        Args:
            path: Path to the ratings JSON file (used for local storage)
            anchor_ids: Set of player IDs that are anchors (fixed ratings)
            use_firestore: If True, use Firestore. If None, auto-detect based on
                          FIREBASE_ENABLED env var or presence of firebase-key.json
        """
        self.path = Path(path)
        self.anchor_ids = anchor_ids or set()
        self._ratings: Dict[str, PlayerRating] = {}

        # Determine storage backend
        if use_firestore is None:
            use_firestore = self._should_use_firestore()
        self._use_firestore = use_firestore

        if self._use_firestore:
            self._init_firestore()
        else:
            self._load()

    def _should_use_firestore(self) -> bool:
        """Check if we should use Firestore."""
        # Check env var
        if os.environ.get("FIREBASE_ENABLED", "").lower() in ("1", "true", "yes"):
            return True

        # Check for credentials file
        from pathlib import Path
        possible_paths = [
            Path(__file__).parent.parent / "firebase-key.json",
            Path.cwd() / "firebase-key.json",
        ]
        for p in possible_paths:
            if p.exists():
                return True

        return False

    def _init_firestore(self) -> None:
        """Initialize Firestore connection and load ratings."""
        from firebase_client import get_firestore_client, RATINGS_COLLECTION
        self._db = get_firestore_client()
        self._collection = RATINGS_COLLECTION
        self._load_from_firestore()

    def _load_from_firestore(self) -> None:
        """Load ratings from Firestore with caching and error handling."""
        global _firestore_cache, _firestore_cache_time

        # Check if we have a valid cache
        cache_age = time.time() - _firestore_cache_time
        if _firestore_cache and cache_age < _FIRESTORE_CACHE_TTL:
            logger.info(f"Using cached Firestore data ({cache_age:.0f}s old)")
            for player_id, player_rating in _firestore_cache.items():
                # Make a copy to avoid shared state issues
                rating_copy = PlayerRating(
                    player_id=player_rating.player_id,
                    rating=player_rating.rating,
                    rating_deviation=player_rating.rating_deviation,
                    volatility=player_rating.volatility,
                    games_played=player_rating.games_played,
                )
                # Enforce rating floor on loaded ratings (anchors exempt)
                if player_id not in self.anchor_ids:
                    if rating_copy.rating < Glicko2System.RATING_FLOOR:
                        rating_copy.rating = Glicko2System.RATING_FLOOR
                self._ratings[player_id] = rating_copy
            return

        try:
            # Fetch fresh data from Firestore with a timeout
            docs = self._db.collection(self._collection).stream(timeout=10)
            new_cache: Dict[str, PlayerRating] = {}

            for doc in docs:
                data = doc.to_dict()
                player_rating = PlayerRating.from_dict(data)
                new_cache[data["player_id"]] = player_rating

                # Enforce rating floor on loaded ratings (anchors exempt)
                if data["player_id"] not in self.anchor_ids:
                    if player_rating.rating < Glicko2System.RATING_FLOOR:
                        player_rating.rating = Glicko2System.RATING_FLOOR
                self._ratings[data["player_id"]] = player_rating

            # Update the cache on success
            _firestore_cache = new_cache
            _firestore_cache_time = time.time()
            logger.info(f"Loaded {len(new_cache)} ratings from Firestore")

        except Exception as e:
            error_name = type(e).__name__
            logger.warning(f"Firestore error ({error_name}): {e}")

            # Fall back to cache if available (even if expired)
            if _firestore_cache:
                logger.info(f"Falling back to cached data ({len(_firestore_cache)} ratings)")
                for player_id, player_rating in _firestore_cache.items():
                    rating_copy = PlayerRating(
                        player_id=player_rating.player_id,
                        rating=player_rating.rating,
                        rating_deviation=player_rating.rating_deviation,
                        volatility=player_rating.volatility,
                        games_played=player_rating.games_played,
                    )
                    if player_id not in self.anchor_ids:
                        if rating_copy.rating < Glicko2System.RATING_FLOOR:
                            rating_copy.rating = Glicko2System.RATING_FLOOR
                    self._ratings[player_id] = rating_copy
            else:
                # No cache available - try loading from local file as last resort
                logger.warning("No cache available, attempting local file fallback")
                if self.path.exists():
                    self._load()
                else:
                    logger.error("No fallback data available, starting with empty ratings")

    def _save_to_firestore(self, player_id: str) -> None:
        """Save a single player's rating to Firestore."""
        if player_id in self._ratings:
            self._db.collection(self._collection).document(player_id).set(
                self._ratings[player_id].to_dict()
            )

    def _save_all_to_firestore(self) -> None:
        """Save all ratings to Firestore."""
        batch = self._db.batch()
        for player_id, rating in self._ratings.items():
            ref = self._db.collection(self._collection).document(player_id)
            batch.set(ref, rating.to_dict())
        batch.commit()

    def _load(self) -> None:
        """Load ratings from local file."""
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
        """Save ratings to local file."""
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

    def set(self, rating: PlayerRating, auto_save: bool = True) -> None:
        """
        Set a player's rating.

        Args:
            rating: The PlayerRating to store
            auto_save: Whether to save to disk immediately (default True)
        """
        self._ratings[rating.player_id] = rating
        if auto_save:
            if self._use_firestore:
                self._save_to_firestore(rating.player_id)
            else:
                self._save()

    def save(self) -> None:
        """Manually save ratings to disk."""
        if self._use_firestore:
            self._save_all_to_firestore()
        else:
            self._save()

    def is_anchor(self, player_id: str) -> bool:
        """Check if a player is an anchor with fixed rating."""
        return player_id in self.anchor_ids

    def has_player(self, player_id: str) -> bool:
        """Check if a player exists in the store."""
        return player_id in self._ratings

    def set_anchor(
        self,
        player_id: str,
        rating: float,
        rating_deviation: float = 30.0,
        auto_save: bool = True,
    ) -> None:
        """
        Set an anchor player with fixed rating.

        Args:
            player_id: The anchor player ID
            rating: The fixed rating
            rating_deviation: Low RD for anchors (default 30)
            auto_save: Whether to save to disk immediately (default True)
        """
        self.anchor_ids.add(player_id)
        self._ratings[player_id] = PlayerRating(
            player_id=player_id,
            rating=rating,
            rating_deviation=rating_deviation,
            volatility=0.03,  # Low volatility for anchors
            games_played=0,
        )
        if auto_save:
            if self._use_firestore:
                self._save_to_firestore(player_id)
            else:
                self._save()

    def get_all(self) -> Dict[str, PlayerRating]:
        """Get all player ratings."""
        return dict(self._ratings)

    def get_sorted_ratings(self, min_games: int = 0) -> list[PlayerRating]:
        """
        Get ratings sorted by rating (descending).

        Args:
            min_games: Minimum games played to include (anchors always included)

        Returns:
            List of PlayerRating sorted by rating
        """
        ratings = [
            r for r in self._ratings.values()
            if r.games_played >= min_games or r.player_id in self.anchor_ids
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
        if self._use_firestore:
            # Delete all non-anchor documents
            docs = self._db.collection(self._collection).stream()
            batch = self._db.batch()
            for doc in docs:
                if doc.id not in self.anchor_ids:
                    batch.delete(doc.reference)
            batch.commit()
            self._save_all_to_firestore()
        else:
            self._save()
