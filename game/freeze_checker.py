"""
Freeze-checking logic for benchmark models.

Determines whether a model should stop playing games based on
rating stability, age, performance relative to peers, and cost.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from rating.rating_store import RatingStore
from rating.cost_calculator import CostCalculator, filter_results_by_rating_diff
from game.stats_collector import StatsCollector


class FreezeChecker:
    """
    Checks whether models should be frozen (stop playing games).

    Extracted from MatchScheduler so it can be used independently
    (e.g., from recalculate) without needing a full scheduler instance.
    """

    # Base freeze threshold
    FROZEN_RD_THRESHOLD = 60

    # Age-based freezing
    FROZEN_AGE_RD_THRESHOLD_6M = 70
    FROZEN_AGE_MONTHS_6M = 6
    FROZEN_AGE_RD_THRESHOLD_1Y = 80
    FROZEN_AGE_MONTHS_1Y = 12

    # Recent weak model freezing (released after Nov 2025)
    RECENT_WEAK_REASONING_RD_THRESHOLD = 100
    RECENT_WEAK_REASONING_RATING_THRESHOLD = 1000
    RECENT_WEAK_NONREASONING_RD_THRESHOLD = 100
    RECENT_WEAK_NONREASONING_RATING_THRESHOLD = 500
    RECENT_WEAK_CUTOFF = datetime(2025, 11, 1, tzinfo=timezone.utc).timestamp()

    # Within-year weak model freezing (released after Apr 2025)
    WITHIN_YEAR_WEAK_RD_THRESHOLD = 100
    WITHIN_YEAR_WEAK_RATING_THRESHOLD = 0
    WITHIN_YEAR_WEAK_CUTOFF = datetime(2025, 4, 1, tzinfo=timezone.utc).timestamp()

    # Proven worse freezing
    PROVEN_WORSE_RD_THRESHOLD = 80
    PROVEN_WORSE_MIN_LOSSES = 3
    PROVEN_WORSE_TIME_WINDOW = 2

    # Effort variant freezing
    EFFORT_VARIANT_RD_THRESHOLD = 150
    EFFORT_VARIANT_MIN_LOSSES = 3
    EFFORT_LEVELS = {
        "no thinking": 0,
        "minimal": 1,
        "low": 2,
        "medium": 3,
        "thinking": 4,
        "high": 4,
    }

    # Provider inferior freezing
    PROVIDER_INFERIOR_RD_THRESHOLD = 100
    PROVIDER_INFERIOR_MIN_LOSSES = 3
    PROVIDER_INFERIOR_TIME_WINDOW = 3
    PROVIDER_INFERIOR_COST_RATIO = 2.0

    # Expensive inferior freezing
    EXPENSIVE_INFERIOR_RD_THRESHOLD = 150
    EXPENSIVE_INFERIOR_MIN_LOSSES = 3
    EXPENSIVE_INFERIOR_TIME_WINDOW = 3
    EXPENSIVE_INFERIOR_COST_RATIO = 2.0

    # Lost to much weaker model freezing
    LOST_TO_WEAKER_RATING_GAP = 600
    LOST_TO_WEAKER_TIME_WINDOW = 3
    LOST_TO_WEAKER_COST_RATIO = 5.0

    # Cost estimation
    UNKNOWN_MODEL_DEFAULT_COST = 1.0

    def __init__(
        self,
        rating_store: RatingStore,
        stats_collector: StatsCollector,
        reasoning_ids: Optional[set] = None,
        engine_ids: Optional[set] = None,
    ):
        self.rating_store = rating_store
        self.stats_collector = stats_collector
        self.reasoning_ids = reasoning_ids or set()
        self.engine_ids = engine_ids or set()

        # Publish date and provider data
        self._publish_dates: Dict[str, int] = {}
        self._player_providers: Dict[str, str] = {}
        self._models_by_provider: Dict[str, List[str]] = {}
        self._player_model_ids: Dict[str, str] = {}
        self._models_by_model_id: Dict[str, List[str]] = {}
        self._load_publish_dates()

        # Cost estimation
        self._cost_calculator = CostCalculator()
        self._cost_cache: Dict[str, float] = {}
        self._cost_data_cache: Optional[Dict[str, Any]] = None

    def _load_publish_dates(self) -> None:
        publish_dates_path = Path(__file__).parent.parent / "data" / "model_publish_dates.json"
        try:
            with open(publish_dates_path) as f:
                data = json.load(f)
                for player_id, info in data.items():
                    if "created_timestamp" in info:
                        self._publish_dates[player_id] = info["created_timestamp"]
                    model_id = info.get("model_id", "")
                    if "/" in model_id:
                        provider = model_id.split("/")[0]
                        self._player_providers[player_id] = provider
                        if provider not in self._models_by_provider:
                            self._models_by_provider[provider] = []
                        self._models_by_provider[provider].append(player_id)
                    if model_id:
                        self._player_model_ids[player_id] = model_id
                        if model_id not in self._models_by_model_id:
                            self._models_by_model_id[model_id] = []
                        self._models_by_model_id[model_id].append(player_id)
        except (FileNotFoundError, json.JSONDecodeError):
            pass

    def get_player_cost(self, player_id: str) -> float:
        """Get estimated cost per game for a player."""
        if player_id in self._cost_cache:
            return self._cost_cache[player_id]

        if player_id in self.engine_ids:
            self._cost_cache[player_id] = 0.0
            return 0.0

        # Historical cost
        if self._cost_data_cache is None:
            filtered_results = filter_results_by_rating_diff(
                self.stats_collector.results, self.rating_store
            )
            self._cost_data_cache = self._cost_calculator.calculate_player_costs(
                filtered_results
            )

        if player_id in self._cost_data_cache:
            avg_cost = self._cost_data_cache[player_id].get("avg_cost_per_game", 0.0)
            if avg_cost > 0:
                self._cost_cache[player_id] = avg_cost
                return avg_cost

        # Estimate from pricing
        model = self._cost_calculator.get_model_for_player(player_id)
        if not model:
            self._cost_cache[player_id] = self.UNKNOWN_MODEL_DEFAULT_COST
            return self.UNKNOWN_MODEL_DEFAULT_COST

        pricing = self._cost_calculator.get_pricing(model)
        if not pricing:
            self._cost_cache[player_id] = self.UNKNOWN_MODEL_DEFAULT_COST
            return self.UNKNOWN_MODEL_DEFAULT_COST

        # Position benchmark token data
        comp_per_call = None
        try:
            results_path = Path(__file__).parent.parent / "position_benchmark" / "results.json"
            if results_path.exists():
                with open(results_path) as f:
                    bench_data = json.load(f)
                if player_id in bench_data:
                    tu = bench_data[player_id].get("token_usage", {})
                    results = bench_data[player_id].get("results", [])
                    n_positions = len(results) if results else 50
                    bench_comp = tu.get("completion", 0)
                    if bench_comp > 0:
                        comp_per_call = bench_comp / n_positions
        except Exception:
            pass

        if comp_per_call is None:
            comp_per_call = 3000 if player_id in self.reasoning_ids else 10

        estimated_cost = max(0.0, (
            100 * 1500 * pricing.get("prompt", 0) +
            100 * comp_per_call * pricing.get("completion", 0)
        ))
        self._cost_cache[player_id] = estimated_cost
        return estimated_cost

    def invalidate_cost_cache(self) -> None:
        self._cost_data_cache = None
        self._cost_cache.clear()

    def _get_effort_level(self, player_id: str) -> Optional[int]:
        start = player_id.rfind('(')
        end = player_id.rfind(')')
        if start != -1 and end != -1 and end > start:
            tag = player_id[start + 1:end]
            return self.EFFORT_LEVELS.get(tag)
        return None

    def is_proven_worse(self, player_id: str,
                        player_stats: Dict[str, Any]) -> bool:
        """Check if model is statistically significantly worse than another."""
        my_timestamp = self._publish_dates.get(player_id)
        if my_timestamp is None:
            return False

        my_data = self.rating_store.get(player_id)
        my_rating = my_data.rating
        my_rd = my_data.rating_deviation

        two_months = self.PROVEN_WORSE_TIME_WINDOW * 30.44 * 24 * 60 * 60
        is_reasoning = player_id in self.reasoning_ids

        for other_id, other_timestamp in self._publish_dates.items():
            if other_id == player_id:
                continue
            if not is_reasoning and other_id in self.reasoning_ids:
                continue
            if other_timestamp > my_timestamp + two_months:
                continue
            if not self.rating_store.has_player(other_id):
                continue

            other_data = self.rating_store.get(other_id)
            if other_data.rating_deviation >= self.PROVEN_WORSE_RD_THRESHOLD:
                continue
            if other_data.rating <= my_rating:
                continue

            other_losses = player_stats.get(other_id, {}).get("losses", 0)
            if other_losses < self.PROVEN_WORSE_MIN_LOSSES:
                continue

            if other_data.rating - my_rating > my_rd + other_data.rating_deviation:
                return True

        return False

    def is_inferior_effort_variant(self, player_id: str,
                                   player_stats: Dict[str, Any]) -> bool:
        """Check if a higher-effort variant is outperformed by a lower-effort sibling."""
        model_id = self._player_model_ids.get(player_id)
        if not model_id:
            return False

        my_effort = self._get_effort_level(player_id)
        if my_effort is None:
            return False

        my_rating = self.rating_store.get(player_id).rating

        for other_id in self._models_by_model_id.get(model_id, []):
            if other_id == player_id:
                continue

            other_effort = self._get_effort_level(other_id)
            if other_effort is None or other_effort >= my_effort:
                continue

            if not self.rating_store.has_player(other_id):
                continue

            other_data = self.rating_store.get(other_id)
            if other_data.rating_deviation >= self.EFFORT_VARIANT_RD_THRESHOLD:
                continue
            if other_data.rating <= my_rating:
                continue

            other_losses = player_stats.get(other_id, {}).get("losses", 0)
            if other_losses < self.EFFORT_VARIANT_MIN_LOSSES:
                continue

            return True

        return False

    def is_inferior_to_provider_sibling(self, player_id: str) -> bool:
        """Check if model is inferior to another model from the same provider."""
        provider = self._player_providers.get(player_id)
        if not provider:
            return False

        my_timestamp = self._publish_dates.get(player_id)
        if my_timestamp is None:
            return False

        my_rating = self.rating_store.get(player_id).rating
        my_cost = self.get_player_cost(player_id)

        three_months = self.PROVIDER_INFERIOR_TIME_WINDOW * 30.44 * 24 * 60 * 60
        is_reasoning = player_id in self.reasoning_ids

        for other_id in self._models_by_provider.get(provider, []):
            if other_id == player_id:
                continue
            if not is_reasoning and other_id in self.reasoning_ids:
                continue

            other_timestamp = self._publish_dates.get(other_id)
            if other_timestamp is None:
                continue
            if other_timestamp > my_timestamp + three_months:
                continue
            if not self.rating_store.has_player(other_id):
                continue

            other_rating = self.rating_store.get(other_id).rating
            if other_rating <= my_rating:
                continue

            if my_cost == 0:
                continue

            other_cost = self.get_player_cost(other_id)
            if other_cost > 0 and my_cost * self.PROVIDER_INFERIOR_COST_RATIO <= other_cost:
                continue

            return True

        return False

    def lost_to_much_weaker(self, player_id: str) -> bool:
        """Check if model lost to a model far below its peer group."""
        my_timestamp = self._publish_dates.get(player_id)
        if my_timestamp is None:
            return False

        my_cost = self.get_player_cost(player_id)
        three_months = self.LOST_TO_WEAKER_TIME_WINDOW * 30.44 * 24 * 60 * 60

        lost_to = set()
        for result in self.stats_collector.results:
            if result.winner == "white" and result.black_id == player_id:
                lost_to.add(result.white_id)
            elif result.winner == "black" and result.white_id == player_id:
                lost_to.add(result.black_id)

        if not lost_to:
            return False

        lost_to_ratings = {}
        for opp_id in lost_to:
            if self.rating_store.has_player(opp_id):
                lost_to_ratings[opp_id] = self.rating_store.get(opp_id).rating

        if not lost_to_ratings:
            return False

        lowest_loss_rating = min(lost_to_ratings.values())

        for peer_id, peer_timestamp in self._publish_dates.items():
            if peer_id == player_id:
                continue
            if peer_timestamp > my_timestamp + three_months:
                continue
            if not self.rating_store.has_player(peer_id):
                continue

            peer_rating = self.rating_store.get(peer_id).rating
            if peer_rating - lowest_loss_rating < self.LOST_TO_WEAKER_RATING_GAP:
                continue

            if my_cost == 0:
                continue
            peer_cost = self.get_player_cost(peer_id)
            if peer_cost > 0 and my_cost * self.LOST_TO_WEAKER_COST_RATIO <= peer_cost:
                continue

            return True

        return False

    def is_expensive_inferior(self, player_id: str) -> bool:
        """Check if model is outperformed by a cheaper model (any provider)."""
        my_timestamp = self._publish_dates.get(player_id)
        if my_timestamp is None:
            return False

        my_rating = self.rating_store.get(player_id).rating
        my_cost = self.get_player_cost(player_id)

        if my_cost == 0:
            return False

        three_months = self.EXPENSIVE_INFERIOR_TIME_WINDOW * 30.44 * 24 * 60 * 60
        is_reasoning = player_id in self.reasoning_ids

        for other_id, other_timestamp in self._publish_dates.items():
            if other_id == player_id:
                continue
            if not is_reasoning and other_id in self.reasoning_ids:
                continue
            if other_timestamp > my_timestamp + three_months:
                continue
            if not self.rating_store.has_player(other_id):
                continue

            other_rating = self.rating_store.get(other_id).rating
            if other_rating <= my_rating:
                continue

            other_cost = self.get_player_cost(other_id)
            if other_cost == 0 or my_cost >= self.EXPENSIVE_INFERIOR_COST_RATIO * other_cost:
                return True

        return False

    def is_frozen(self, player_id: str, current_rd: float,
                  player_stats: Optional[Dict[str, Any]] = None) -> bool:
        """
        Check if a model is frozen (shouldn't play more games).

        See MatchScheduler._is_frozen for the full rule documentation.
        """
        # Always frozen below base threshold
        if current_rd < self.FROZEN_RD_THRESHOLD:
            return True

        # Age-based freezing
        if player_id in self._publish_dates:
            publish_timestamp = self._publish_dates[player_id]
            now = datetime.now(timezone.utc).timestamp()
            age_months = (now - publish_timestamp) / (30.44 * 24 * 60 * 60)

            if current_rd < self.FROZEN_AGE_RD_THRESHOLD_6M and age_months > self.FROZEN_AGE_MONTHS_6M:
                return True
            if current_rd < self.FROZEN_AGE_RD_THRESHOLD_1Y and age_months > self.FROZEN_AGE_MONTHS_1Y:
                return True

            # Within-year weak model freezing
            if publish_timestamp >= self.WITHIN_YEAR_WEAK_CUTOFF:
                current_rating = self.rating_store.get(player_id).rating
                if (current_rd < self.WITHIN_YEAR_WEAK_RD_THRESHOLD and
                        current_rating < self.WITHIN_YEAR_WEAK_RATING_THRESHOLD):
                    return True

                # Recent weak model freezing
                if publish_timestamp >= self.RECENT_WEAK_CUTOFF:
                    is_reasoning = player_id in self.reasoning_ids
                    if is_reasoning:
                        if (current_rd < self.RECENT_WEAK_REASONING_RD_THRESHOLD and
                                current_rating < self.RECENT_WEAK_REASONING_RATING_THRESHOLD):
                            return True
                    else:
                        if (current_rd < self.RECENT_WEAK_NONREASONING_RD_THRESHOLD and
                                current_rating < self.RECENT_WEAK_NONREASONING_RATING_THRESHOLD):
                            return True

        # Proven worse freezing
        if current_rd < self.PROVEN_WORSE_RD_THRESHOLD:
            if player_stats is None:
                player_stats = self.stats_collector.get_player_stats()
            losses = player_stats.get(player_id, {}).get("losses", 0)
            if losses >= self.PROVEN_WORSE_MIN_LOSSES:
                if self.is_proven_worse(player_id, player_stats):
                    return True

        # Inferior effort variant freezing
        if current_rd < self.EFFORT_VARIANT_RD_THRESHOLD:
            if player_stats is None:
                player_stats = self.stats_collector.get_player_stats()
            losses = player_stats.get(player_id, {}).get("losses", 0)
            if losses >= self.EFFORT_VARIANT_MIN_LOSSES:
                if self.is_inferior_effort_variant(player_id, player_stats):
                    return True

        # Inferior same-provider model freezing
        if current_rd < self.PROVIDER_INFERIOR_RD_THRESHOLD:
            if player_stats is None:
                player_stats = self.stats_collector.get_player_stats()
            losses = player_stats.get(player_id, {}).get("losses", 0)
            if losses >= self.PROVIDER_INFERIOR_MIN_LOSSES:
                if self.is_inferior_to_provider_sibling(player_id):
                    return True

        # Lost to much weaker model freezing (no RD requirement)
        if self.lost_to_much_weaker(player_id):
            return True

        # Expensive inferior model freezing
        if current_rd < self.EXPENSIVE_INFERIOR_RD_THRESHOLD:
            if player_stats is None:
                player_stats = self.stats_collector.get_player_stats()
            losses = player_stats.get(player_id, {}).get("losses", 0)
            if losses >= self.EXPENSIVE_INFERIOR_MIN_LOSSES:
                if self.is_expensive_inferior(player_id):
                    return True

        return False

    def compute_frozen_flags(self) -> Dict[str, bool]:
        """Compute frozen status for all players in the rating store."""
        player_stats = self.stats_collector.get_player_stats()
        flags = {}
        for player_id in list(self.rating_store._ratings.keys()):
            if self.rating_store.is_anchor(player_id):
                flags[player_id] = False
                continue
            data = self.rating_store.get(player_id)
            flags[player_id] = self.is_frozen(player_id, data.rating_deviation, player_stats)
        return flags
