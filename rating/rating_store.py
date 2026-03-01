"""
Rating storage and persistence.
"""

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Set

from .glicko2 import PlayerRating, Glicko2System

logger = logging.getLogger(__name__)

# Module-level cache for Firestore data
_firestore_cache: Dict[str, "PlayerRating"] = {}
_firestore_cache_time: float = 0
_FIRESTORE_CACHE_TTL = 3600  # 1 hour

# Cache invalidation signal file
_CACHE_INVALIDATE_FILE = Path(__file__).parent.parent / "data" / ".cache_invalidate"


def invalidate_cache() -> None:
    """Touch the cache invalidation file to signal all processes to refresh."""
    try:
        _CACHE_INVALIDATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_INVALIDATE_FILE.touch()
        logger.info(f"Cache invalidation signal sent: {_CACHE_INVALIDATE_FILE}")
    except OSError as e:
        logger.warning(f"Failed to touch cache invalidation file: {e}")


def _should_invalidate_cache() -> bool:
    """Check if cache should be invalidated based on signal file."""
    try:
        if not _CACHE_INVALIDATE_FILE.exists():
            return False
        file_mtime = _CACHE_INVALIDATE_FILE.stat().st_mtime
        return file_mtime > _firestore_cache_time
    except OSError as e:
        logger.warning(f"Failed to check cache invalidation file: {e}")
        return False


class RatingStore:
    """
    Stores and manages player ratings.

    Supports both local JSON file and Firestore backends.
    """

    def __init__(
        self,
        path: str = "data/ratings.json",
        anchor_ids: Set[str] = None,
        ghost_ids: Set[str] = None,
        use_firestore: bool = None,
    ):
        """
        Initialize the rating store.

        Args:
            path: Path to the ratings JSON file (used for local storage)
            anchor_ids: Set of player IDs that are anchors (fixed ratings)
            ghost_ids: Set of player IDs that are "ghosts" - their games don't
                      affect opponents' ratings/RD (but their own rating updates)
            use_firestore: If True, use Firestore. If None, auto-detect based on
                          FIREBASE_ENABLED env var or presence of firebase-key.json
        """
        self.path = Path(path)
        self.anchor_ids = anchor_ids or set()
        self.ghost_ids = ghost_ids or set()
        self._ratings: Dict[str, PlayerRating] = {}

        # Load benchmark predictions for seeding new players
        self._benchmark_predictions = self._load_benchmark_predictions()

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

    def _load_benchmark_predictions(self) -> Dict[str, float]:
        """
        Load position benchmark results and compute predicted ratings.

        Uses the 3-feature formula from rating_prediction_formulas.py:
          rating = 1794.14 - 260.55*log(eq_cpl+1) + 21.48*pct_lt10 + 2.09*surv_40

        Returns:
            Dict mapping model name to predicted rating, or empty dict if files missing.
        """
        base = Path(__file__).parent.parent / "position_benchmark"
        results_path = base / "results.json"
        positions_path = base / "positions.json"

        if not results_path.exists() or not positions_path.exists():
            return {}

        try:
            with open(results_path) as f:
                results_data = json.load(f)
            with open(positions_path) as f:
                positions_data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load benchmark data: {e}")
            return {}

        # Build position type lookup
        pos_type = {}
        for i, p in enumerate(positions_data.get("positions", [])):
            pos_type[i] = p.get("type", "")

        predictions = {}
        for model_name, model_data in results_data.items():
            model_results = model_data.get("results", [])
            if not model_results:
                continue

            # Collect equal-position metrics
            eq_cpls = []
            eq_lt10_count = 0
            eq_illegal_count = 0
            eq_total = 0

            for r in model_results:
                idx = r.get("position_idx", -1)
                if pos_type.get(idx) != "equal":
                    continue
                eq_total += 1
                cpl = r.get("cpl", 0)
                eq_cpls.append(cpl)
                if cpl < 10:
                    eq_lt10_count += 1
                if not r.get("is_legal", True):
                    eq_illegal_count += 1

            if eq_total == 0:
                continue

            eq_cpl = sum(eq_cpls) / len(eq_cpls)
            pct_lt10 = 100.0 * eq_lt10_count / eq_total
            p = eq_illegal_count / eq_total  # illegal rate

            # Survival probability: P(0 or 1 illegal in 40 moves) - pure Python binomial
            if p <= 0:
                surv_40 = 100.0
            elif p >= 1:
                surv_40 = 0.0
            else:
                surv_40 = 100.0 * ((1 - p) ** 40 + 40 * p * (1 - p) ** 39)

            predicted = (
                1794.14
                - 260.55 * math.log(eq_cpl + 1)
                + 21.48 * pct_lt10
                + 2.09 * surv_40
            )
            predictions[model_name] = predicted

        if predictions:
            logger.info(f"Loaded benchmark predictions for {len(predictions)} models")

        return predictions

    def _init_firestore(self) -> None:
        """Initialize Firestore connection and load ratings."""
        from firebase_client import get_firestore_client, RATINGS_COLLECTION
        self._db = get_firestore_client()
        self._collection = RATINGS_COLLECTION
        self._load_from_firestore()

    def _load_from_firestore(self) -> None:
        """Load ratings from Firestore with caching and error handling."""
        global _firestore_cache, _firestore_cache_time

        # Check if we have a valid cache (not expired and not invalidated)
        cache_age = time.time() - _firestore_cache_time
        cache_valid = _firestore_cache and cache_age < _FIRESTORE_CACHE_TTL
        if cache_valid and not _should_invalidate_cache():
            logger.info(f"Using cached Firestore data ({cache_age:.0f}s old)")
            # Copy the entire cache dict efficiently
            self._ratings = {
                pid: PlayerRating(
                    player_id=pr.player_id,
                    rating=pr.rating,
                    rating_deviation=pr.rating_deviation,
                    volatility=pr.volatility,
                    games_played=pr.games_played,
                    wins=pr.wins,
                    losses=pr.losses,
                    draws=pr.draws,
                    unclamped_rating=pr.unclamped_rating,
                    games_rd=pr.games_rd,
                )
                for pid, pr in _firestore_cache.items()
            }
            return

        try:
            # Fetch fresh data from Firestore with a timeout
            docs = self._db.collection(self._collection).stream(timeout=10)
            new_cache: Dict[str, PlayerRating] = {}

            for doc in docs:
                data = doc.to_dict()
                player_rating = PlayerRating.from_dict(data)
                new_cache[data["player_id"]] = player_rating
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
                self._ratings = {
                    pid: PlayerRating(
                        player_id=pr.player_id,
                        rating=pr.rating,
                        rating_deviation=pr.rating_deviation,
                        volatility=pr.volatility,
                        games_played=pr.games_played,
                        wins=pr.wins,
                        losses=pr.losses,
                        draws=pr.draws,
                        unclamped_rating=pr.unclamped_rating,
                        games_rd=pr.games_rd,
                    )
                    for pid, pr in _firestore_cache.items()
                }
                # Update cache time to prevent repeated fetch attempts
                _firestore_cache_time = time.time()
            else:
                # No cache available - try loading from local file as last resort
                logger.warning("No cache available, attempting local file fallback")
                if self.path.exists():
                    self._load()
                else:
                    logger.error("No fallback data available, starting with empty ratings")

    def _save_to_firestore(self, player_id: str) -> None:
        """Save a single player's rating to Firestore and update cache."""
        global _firestore_cache
        if player_id in self._ratings:
            self._db.collection(self._collection).document(player_id).set(
                self._ratings[player_id].to_dict()
            )
            # Update cache to maintain consistency
            if _firestore_cache:
                rating = self._ratings[player_id]
                _firestore_cache[player_id] = PlayerRating(
                    player_id=rating.player_id,
                    rating=rating.rating,
                    rating_deviation=rating.rating_deviation,
                    volatility=rating.volatility,
                    games_played=rating.games_played,
                    wins=rating.wins,
                    losses=rating.losses,
                    draws=rating.draws,
                    unclamped_rating=rating.unclamped_rating,
                    games_rd=rating.games_rd,
                )

    def _save_all_to_firestore(self) -> None:
        """Save all ratings to Firestore and update cache."""
        global _firestore_cache, _firestore_cache_time
        batch = self._db.batch()
        for player_id, rating in self._ratings.items():
            ref = self._db.collection(self._collection).document(player_id)
            batch.set(ref, rating.to_dict())
        batch.commit()
        # Update cache with all current ratings
        _firestore_cache = {
            player_id: PlayerRating(
                player_id=rating.player_id,
                rating=rating.rating,
                rating_deviation=rating.rating_deviation,
                volatility=rating.volatility,
                games_played=rating.games_played,
                wins=rating.wins,
                losses=rating.losses,
                draws=rating.draws,
                unclamped_rating=rating.unclamped_rating,
                games_rd=rating.games_rd,
            )
            for player_id, rating in self._ratings.items()
        }
        _firestore_cache_time = time.time()

    def _load(self) -> None:
        """Load ratings from local file."""
        if self.path.exists():
            with open(self.path) as f:
                data = json.load(f)
            for player_id, rating_data in data.items():
                player_rating = PlayerRating.from_dict(rating_data)
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
            if player_id in self._benchmark_predictions:
                predicted = self._benchmark_predictions[player_id]
                self._ratings[player_id] = PlayerRating(
                    player_id=player_id,
                    rating=predicted,
                    rating_deviation=149.0,
                    games_rd=350.0,
                )
            else:
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

    def is_ghost(self, player_id: str) -> bool:
        """Check if a player is a ghost (opponents don't get rating updates)."""
        return player_id in self.ghost_ids

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
            unclamped_rating=rating,  # Anchors have fixed ratings
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
