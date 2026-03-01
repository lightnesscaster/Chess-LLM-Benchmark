"""
Match scheduler for running benchmark games.

Handles:
- Pairing generation (LLM vs anchors, LLM vs LLM)
- Parallel game execution
- Rating updates after games
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional
from dataclasses import dataclass

from engines.base_engine import BaseEngine
from llm.base_llm import BaseLLMPlayer
from .game_runner import GameRunner
from .models import GameResult, PlayerConfig
from .pgn_logger import PGNLogger
from .stats_collector import StatsCollector
from rating.glicko2 import Glicko2System, PlayerRating
from rating.rating_store import RatingStore
from rating.cost_calculator import CostCalculator, filter_results_by_rating_diff


Player = Union[BaseEngine, BaseLLMPlayer]


@dataclass
class GameTask:
    """A scheduled game to play."""
    white: Player
    black: Player
    game_num: int
    total_games: int


class MatchScheduler:
    """
    Schedules and runs benchmark games.
    """

    # Game caps for reasoning models
    REASONING_BASE_CAP = 10  # Default cap for reasoning models
    REASONING_HIGH_RATING_CAP = 15  # Cap for reasoning models rated > 1000
    REASONING_HIGH_RATING_THRESHOLD = 1000  # Rating threshold for higher cap

    # Cap for models with low rating deviation (rating is stable)
    LOW_RD_THRESHOLD = 70  # RD below this triggers reduced cap
    LOW_RD_CAP = 10  # Max games for models with RD < threshold

    # Frozen threshold - models below this RD don't initiate games (but can be challenged)
    FROZEN_RD_THRESHOLD = 60

    # Age-based freezing thresholds (for older models with stable ratings)
    # Models with RD < 70 and released > 6 months ago are frozen
    FROZEN_AGE_RD_THRESHOLD_6M = 70
    FROZEN_AGE_MONTHS_6M = 6
    # Models with RD < 80 and released > 1 year ago are frozen
    FROZEN_AGE_RD_THRESHOLD_1Y = 80
    FROZEN_AGE_MONTHS_1Y = 12

    # Recent weak model freezing (released < 6 months ago)
    # Reasoning models: freeze at RD < 100 if rating < 1000
    RECENT_WEAK_REASONING_RD_THRESHOLD = 100
    RECENT_WEAK_REASONING_RATING_THRESHOLD = 1000
    # Non-reasoning models: freeze at RD < 100 if rating < 500
    RECENT_WEAK_NONREASONING_RD_THRESHOLD = 100
    RECENT_WEAK_NONREASONING_RATING_THRESHOLD = 500

    # Within-year weak model freezing (released < 1 year ago)
    # Any model: freeze at RD < 100 if rating < 0
    WITHIN_YEAR_WEAK_RD_THRESHOLD = 100
    WITHIN_YEAR_WEAK_RATING_THRESHOLD = 0

    # Legal move rate threshold - models below this must play random-bot (if rated above -200)
    LEGAL_MOVE_RATE_THRESHOLD = 0.98  # 98% accuracy
    LOW_ACCURACY_RATING_THRESHOLD = -200  # Only enforce accuracy requirement above this rating
    RANDOM_BOT_MIN_GAMES = 5  # Min games vs random-bot to prove competence (must not lose any)
    RANDOM_BOT_ID = "random-bot"

    # Tighter rating threshold for stable models (low RD)
    # When RD is below this, use a tighter rating threshold for pairings
    STABLE_RD_THRESHOLD = 100  # RD below this triggers tighter pairing
    STABLE_RATING_THRESHOLD = 300  # Use 300 point window instead of default (600)

    # Cost-aware scheduling parameters
    # LLM priority: RD / (1 + COST_SENSITIVITY * cost_per_game)
    COST_SENSITIVITY = 0.5  # How much cost affects LLM priority
    # Opponent selection: rating_diff + min(OPPONENT_COST_WEIGHT * cost, OPPONENT_COST_CAP)
    OPPONENT_COST_WEIGHT = 200  # Rating points per $1 of opponent cost
    OPPONENT_COST_CAP = 400  # Max rating points penalty from opponent cost

    def __init__(
        self,
        players: Dict[str, Player],
        rating_store: RatingStore,
        glicko: Glicko2System,
        pgn_logger: PGNLogger,
        stats_collector: StatsCollector,
        max_concurrent: int = 4,
        max_moves: int = 200,
        verbose: bool = False,
        reasoning_ids: Optional[set] = None,
    ):
        """
        Initialize the scheduler.

        Args:
            players: Dict mapping player_id to Player object
            rating_store: Rating storage
            glicko: Glicko-2 system for updates
            pgn_logger: PGN logger
            stats_collector: Stats collector
            max_concurrent: Maximum concurrent games
            max_moves: Maximum moves per game
            verbose: Print verbose output
            reasoning_ids: Set of player IDs that are reasoning models (have game caps)
        """
        self.players = players
        self.rating_store = rating_store
        self.glicko = glicko
        self.pgn_logger = pgn_logger
        self.stats_collector = stats_collector
        self.max_concurrent = max_concurrent
        self.max_moves = max_moves
        self.verbose = verbose
        self.reasoning_ids = reasoning_ids or set()

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Lock for rating updates
        self._rating_lock = asyncio.Lock()

        # Track games played per LLM during benchmark (for reasoning caps and low RD caps)
        # Note: Only used in run_benchmark() flow, protected by scheduler_lock
        self._games_played: Dict[str, int] = {}

        # Load model publish dates for age-based freezing
        self._publish_dates: Dict[str, int] = {}  # player_id -> timestamp
        self._load_publish_dates()

        # Cost calculator for cost-aware scheduling
        self._cost_calculator = CostCalculator()
        self._cost_cache: Dict[str, float] = {}  # Cache estimated costs per player
        self._cost_data_cache: Optional[Dict[str, Dict[str, Any]]] = None  # Filtered cost data
        self._pairwise_cost_cache: Optional[Dict[Tuple[str, str], float]] = None  # Pairwise game costs

    def _load_publish_dates(self) -> None:
        """Load model publish dates from data file."""
        publish_dates_path = Path(__file__).parent.parent / "data" / "model_publish_dates.json"
        try:
            with open(publish_dates_path) as f:
                data = json.load(f)
                for player_id, info in data.items():
                    if "created_timestamp" in info:
                        self._publish_dates[player_id] = info["created_timestamp"]
        except (FileNotFoundError, json.JSONDecodeError):
            pass  # No publish dates available

    # Default cost for models with unknown pricing (conservative estimate)
    UNKNOWN_MODEL_DEFAULT_COST = 1.0  # $1.00 per game

    def invalidate_cost_cache(self) -> None:
        """Invalidate all cost caches. Call this when ratings are updated."""
        self._cost_data_cache = None
        self._cost_cache.clear()
        self._pairwise_cost_cache = None

    def _get_player_cost(self, player_id: str) -> float:
        """
        Get estimated cost per game for a player.

        Uses historical average if available from stats_collector,
        otherwise estimates from pricing data. Only considers games
        against opponents within 600 rating points (DEFAULT_COST_RATING_THRESHOLD).

        Args:
            player_id: The player to get cost for

        Returns:
            Estimated cost per game in dollars (0.0 for engines/free models)
        """
        # Return cached value if available
        if player_id in self._cost_cache:
            return self._cost_cache[player_id]

        # Engines have no cost
        if player_id in self.players and isinstance(self.players[player_id], BaseEngine):
            self._cost_cache[player_id] = 0.0
            return 0.0

        # Try to get historical cost from cached cost_data
        # Calculate cost_data once and cache it (avoid recalculating for every player)
        # Only include games against similarly-rated opponents for accurate cost estimate
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

        # Fall back to estimate from pricing
        model = self._cost_calculator.get_model_for_player(player_id)
        if not model:
            # Unknown model - use conservative default instead of 0
            self._cost_cache[player_id] = self.UNKNOWN_MODEL_DEFAULT_COST
            return self.UNKNOWN_MODEL_DEFAULT_COST

        pricing = self._cost_calculator.get_pricing(model)
        if not pricing:
            # No pricing data - use conservative default
            self._cost_cache[player_id] = self.UNKNOWN_MODEL_DEFAULT_COST
            return self.UNKNOWN_MODEL_DEFAULT_COST

        # Estimate: ~100 LLM calls per game, ~1500 prompt tokens, ~10 completion tokens per call
        estimated_cost = max(0.0, (
            100 * 1500 * pricing.get("prompt", 0) +
            100 * 10 * pricing.get("completion", 0)
        ))
        self._cost_cache[player_id] = estimated_cost
        return estimated_cost

    def _calculate_game_cost(self, result: GameResult) -> float:
        """
        Calculate the cost of a completed game from token usage.

        Args:
            result: The game result with token data

        Returns:
            Total cost in dollars for both players
        """
        total_cost = 0.0

        # Calculate white player cost
        if result.tokens_white:
            white_cost = self._cost_calculator.calculate_game_cost(
                result.tokens_white,
                self._cost_calculator.get_model_for_player(result.white_id) or ""
            )
            if white_cost:
                total_cost += white_cost

        # Calculate black player cost
        if result.tokens_black:
            black_cost = self._cost_calculator.calculate_game_cost(
                result.tokens_black,
                self._cost_calculator.get_model_for_player(result.black_id) or ""
            )
            if black_cost:
                total_cost += black_cost

        return total_cost

    def _estimate_game_cost(self, white_id: str, black_id: str) -> float:
        """
        Estimate cost for a game between two players before it starts.

        Uses historical data in priority order:
        1. Average cost of previous games between these specific players
        2. Sum of each player's average cost (within rating threshold)
        3. Falls back to pricing-based estimates

        Args:
            white_id: White player ID
            black_id: Black player ID

        Returns:
            Estimated cost in dollars for the game
        """
        # Build pairwise cost cache if needed
        if self._pairwise_cost_cache is None:
            self._pairwise_cost_cache = self._build_pairwise_cost_cache()

        # Try pairwise cost first (order-independent key)
        pair_key = tuple(sorted([white_id, black_id]))
        if pair_key in self._pairwise_cost_cache:
            return self._pairwise_cost_cache[pair_key]

        # Fall back to sum of individual player costs
        return self._get_player_cost(white_id) + self._get_player_cost(black_id)

    def _build_pairwise_cost_cache(self) -> Dict[Tuple[str, str], float]:
        """
        Build cache of average costs for games between specific player pairs.

        Returns:
            Dict mapping (player_a, player_b) tuple (sorted) to average game cost
        """
        # Track costs per pairing
        pair_costs: Dict[Tuple[str, str], List[float]] = {}

        for result in self.stats_collector.results:
            # Calculate actual cost for this game
            game_cost = self._calculate_game_cost(result)
            if game_cost <= 0:
                continue

            # Use sorted tuple as key (order-independent)
            pair_key = tuple(sorted([result.white_id, result.black_id]))

            if pair_key not in pair_costs:
                pair_costs[pair_key] = []
            pair_costs[pair_key].append(game_cost)

        # Calculate averages
        return {
            pair: sum(costs) / len(costs)
            for pair, costs in pair_costs.items()
        }

    def _calculate_priority(self, player_id: str) -> float:
        """
        Calculate scheduling priority for a player.

        Priority = RD / (1 + COST_SENSITIVITY * cost_per_game)

        Higher priority = scheduled sooner. High RD models are prioritized,
        but cost acts as a penalty (expensive models get lower priority).

        Args:
            player_id: The player to calculate priority for

        Returns:
            Priority score (higher = more urgent to schedule)
        """
        rd = self.rating_store.get(player_id).games_rd
        cost = self._get_player_cost(player_id)
        return rd / (1 + self.COST_SENSITIVITY * cost)

    def _is_frozen(self, player_id: str, current_rd: float) -> bool:
        """
        Check if a model is frozen (shouldn't initiate games).

        A model is frozen if:
        - RD < 60 (always frozen)
        - RD < 70 and released > 6 months ago
        - RD < 80 and released > 1 year ago
        - Recent (< 6 months) reasoning model with RD < 100 and rating < 1000
        - Recent (< 6 months) non-reasoning model with RD < 100 and rating < 500
        - Within year (< 12 months) any model with RD < 100 and rating < 0

        Args:
            player_id: The player to check
            current_rd: Current rating deviation

        Returns:
            True if frozen, False otherwise
        """
        # Always frozen below base threshold
        if current_rd < self.FROZEN_RD_THRESHOLD:
            return True

        # Check age-based freezing if we have publish date
        if player_id in self._publish_dates:
            publish_timestamp = self._publish_dates[player_id]
            now = datetime.now(timezone.utc).timestamp()
            age_months = (now - publish_timestamp) / (30.44 * 24 * 60 * 60)  # Average month length (365.25/12)

            # RD < 70 and > 6 months old
            if current_rd < self.FROZEN_AGE_RD_THRESHOLD_6M and age_months > self.FROZEN_AGE_MONTHS_6M:
                return True

            # RD < 80 and > 1 year old
            if current_rd < self.FROZEN_AGE_RD_THRESHOLD_1Y and age_months > self.FROZEN_AGE_MONTHS_1Y:
                return True

            # Within-year weak model freezing (< 12 months, any model type)
            # Freeze at RD < 100 if rating < 0
            if age_months <= self.FROZEN_AGE_MONTHS_1Y:
                current_rating = self.rating_store.get(player_id).rating
                if (current_rd < self.WITHIN_YEAR_WEAK_RD_THRESHOLD and
                        current_rating < self.WITHIN_YEAR_WEAK_RATING_THRESHOLD):
                    return True

                # More specific rules for very recent models (< 6 months)
                if age_months <= self.FROZEN_AGE_MONTHS_6M:
                    is_reasoning = player_id in self.reasoning_ids

                    if is_reasoning:
                        # Reasoning models: freeze at RD < 100 if rating < 1000
                        if (current_rd < self.RECENT_WEAK_REASONING_RD_THRESHOLD and
                                current_rating < self.RECENT_WEAK_REASONING_RATING_THRESHOLD):
                            return True
                    else:
                        # Non-reasoning models: freeze at RD < 100 if rating < 500
                        if (current_rd < self.RECENT_WEAK_NONREASONING_RD_THRESHOLD and
                                current_rating < self.RECENT_WEAK_NONREASONING_RATING_THRESHOLD):
                            return True

        return False

    def _get_game_cap(self, player_id: str) -> Optional[int]:
        """
        Get the game cap for a player.

        Returns:
            Game cap for reasoning models, None for non-reasoning models (no cap)
        """
        if player_id not in self.reasoning_ids:
            return None  # No cap for non-reasoning models

        # Check current rating to determine cap
        rating = self.rating_store.get(player_id).rating
        if rating > self.REASONING_HIGH_RATING_THRESHOLD:
            return self.REASONING_HIGH_RATING_CAP
        return self.REASONING_BASE_CAP

    def generate_pairings(
        self,
        llm_ids: List[str],
        anchor_ids: List[str],
        games_vs_anchor_per_color: int = 10,
        games_vs_llm_per_color: int = 5,
        rating_threshold: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        """
        Generate game pairings.

        Args:
            llm_ids: List of LLM player IDs
            anchor_ids: List of anchor engine IDs
            games_vs_anchor_per_color: Games per LLM vs each anchor (each color)
            games_vs_llm_per_color: Games per LLM pair (each color)
            rating_threshold: If set, only pair LLMs with anchors within this rating difference

        Returns:
            List of (white_id, black_id) tuples
        """
        pairings = []

        # LLM vs anchors
        for llm_id in llm_ids:
            llm_rating = self.rating_store.get(llm_id).rating if rating_threshold else None

            for anchor_id in anchor_ids:
                # Check rating threshold if enabled
                if rating_threshold is not None:
                    anchor_rating = self.rating_store.get(anchor_id).rating
                    if abs(llm_rating - anchor_rating) > rating_threshold:
                        continue  # Skip this anchor - too far from LLM's rating

                # LLM as white
                for _ in range(games_vs_anchor_per_color):
                    pairings.append((llm_id, anchor_id))
                # LLM as black
                for _ in range(games_vs_anchor_per_color):
                    pairings.append((anchor_id, llm_id))

        # LLM vs LLM (round robin)
        for i, llm_a in enumerate(llm_ids):
            for llm_b in llm_ids[i + 1:]:
                # A as white
                for _ in range(games_vs_llm_per_color):
                    pairings.append((llm_a, llm_b))
                # B as white
                for _ in range(games_vs_llm_per_color):
                    pairings.append((llm_b, llm_a))

        return pairings

    async def run_single_game(self, task: GameTask) -> Optional[GameResult]:
        """
        Run a single game with concurrency control.

        Note: Reasoning model cap checking is handled by _game_worker() in run_benchmark().
        This method just runs the game - caller is responsible for any cap management.

        Args:
            task: The game task

        Returns:
            GameResult if game completes normally, None if API error
        """
        async with self._semaphore:
            if self.verbose:
                print(f"[{task.game_num}/{task.total_games}] {task.white.player_id} vs {task.black.player_id}")

            runner = GameRunner(
                white=task.white,
                black=task.black,
                max_moves=self.max_moves,
                verbose=self.verbose,
            )

            result, pgn_str = await runner.play_game()

            # Don't save games that ended due to API errors
            if result.termination == "api_error":
                if self.verbose:
                    print(f"  API error - game not saved")
                return None

            # Save PGN and result
            result = self.pgn_logger.save_game(result, pgn_str)

            if self.verbose:
                print(f"  Result: {result.winner} ({result.termination})")

            # Update ratings
            await self._update_ratings(result)

            # Add to stats
            self.stats_collector.add_result(result)

            return result

    async def _update_ratings(self, result: GameResult) -> None:
        """Update ratings after a game."""
        async with self._rating_lock:
            white_id = result.white_id
            black_id = result.black_id

            # Get current ratings
            white_rating = self.rating_store.get(white_id)
            black_rating = self.rating_store.get(black_id)

            # Determine scores
            if result.winner == "white":
                white_score, black_score = 1.0, 0.0
            elif result.winner == "black":
                white_score, black_score = 0.0, 1.0
            else:
                white_score, black_score = 0.5, 0.5

            # Check if either player is a ghost (opponents don't get rating updates)
            white_is_ghost = self.rating_store.is_ghost(white_id)
            black_is_ghost = self.rating_store.is_ghost(black_id)

            # Update white player
            if self.rating_store.is_anchor(white_id):
                # Anchors: track game stats without changing rating
                self._update_player_stats_only(white_rating, white_score)
            elif black_is_ghost:
                # Ghost opponent: track game stats without changing rating/RD
                self._update_player_stats_only(white_rating, white_score)
            else:
                # Normal: full rating update
                new_white = self.glicko.update_rating(
                    white_rating,
                    opponents=[black_rating],
                    scores=[white_score],
                )
                self.rating_store.set(new_white)

            # Update black player
            if self.rating_store.is_anchor(black_id):
                # Anchors: track game stats without changing rating
                self._update_player_stats_only(black_rating, black_score)
            elif white_is_ghost:
                # Ghost opponent: track game stats without changing rating/RD
                self._update_player_stats_only(black_rating, black_score)
            else:
                # Normal: full rating update
                new_black = self.glicko.update_rating(
                    black_rating,
                    opponents=[white_rating],
                    scores=[black_score],
                )
                self.rating_store.set(new_black)

            # Invalidate cost cache since ratings changed (affects filtering)
            self.invalidate_cost_cache()

    def _update_player_stats_only(self, rating: PlayerRating, score: float) -> None:
        """Update game statistics without changing rating/RD (used for anchors and ghost opponents)."""
        rating.games_played += 1
        if score > 0.75:  # Win (1.0)
            rating.wins += 1
        elif score < 0.25:  # Loss (0.0)
            rating.losses += 1
        else:  # Draw (0.5)
            rating.draws += 1
        self.rating_store.set(rating)

    def _needs_random_bot(
        self,
        llm_id: str,
        player_stats: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if an LLM needs to play against random-bot.

        Returns True if:
        - LLM has no stats (new model)
        - LLM has legal_move_rate below 98% AND rating > -200 AND
          (played < 5 games vs random-bot OR lost at least one to random-bot)

        Models are exempt if they've played at least 5 games against random-bot
        without losing any (wins and draws both count as not losing).

        Args:
            llm_id: The LLM to check
            player_stats: Optional cached player stats dict (to avoid repeated computation)
        """
        if player_stats is None:
            player_stats = self.stats_collector.get_player_stats()
        if llm_id not in player_stats:
            return True  # New model with no games
        stats = player_stats[llm_id]
        legal_rate = stats.get("legal_move_rate", 1.0)
        # Only enforce low accuracy requirement for models rated above -200
        if legal_rate < self.LEGAL_MOVE_RATE_THRESHOLD:
            rating = self.rating_store.get(llm_id).rating
            if rating <= self.LOW_ACCURACY_RATING_THRESHOLD:
                return False  # Very low rated models exempt
            # Check head-to-head record vs random-bot
            h2h = self.stats_collector.get_head_to_head(llm_id, self.RANDOM_BOT_ID)
            games_vs_random = h2h["games"]
            losses_to_random = h2h["player_b_wins"]  # random-bot is player_b
            # Exempt if played >= 5 games and won all of them
            if games_vs_random >= self.RANDOM_BOT_MIN_GAMES and losses_to_random == 0:
                return False
            return True
        return False

    def _get_valid_opponents(
        self,
        llm_id: str,
        anchor_ids: List[str],
        llm_ids: List[str],
        rating_threshold: Optional[int],
        player_stats: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Get valid opponents for an LLM based on current ratings.

        Args:
            llm_id: The LLM to find opponents for
            anchor_ids: List of anchor IDs
            llm_ids: List of all LLM IDs
            rating_threshold: Max rating difference (None = no limit)
            player_stats: Optional cached player stats dict (to avoid repeated computation)

        Returns:
            List of valid opponent IDs
        """
        if rating_threshold is None:
            # No threshold - all anchors and other LLMs are valid
            return anchor_ids + [lid for lid in llm_ids if lid != llm_id]

        llm_data = self.rating_store.get(llm_id)
        llm_rating = llm_data.rating
        llm_rd = llm_data.rating_deviation

        # Use tighter threshold for stable models (low RD)
        # Only narrows threshold; has no effect if rating_threshold <= STABLE_RATING_THRESHOLD
        effective_threshold = rating_threshold
        if llm_rd < self.STABLE_RD_THRESHOLD:
            effective_threshold = min(rating_threshold, self.STABLE_RATING_THRESHOLD)

        valid = []

        # Check anchors
        for anchor_id in anchor_ids:
            anchor_rating = self.rating_store.get(anchor_id).rating
            if abs(llm_rating - anchor_rating) <= effective_threshold:
                valid.append(anchor_id)

        # Check other LLMs
        for other_id in llm_ids:
            if other_id == llm_id:
                continue
            other_rating = self.rating_store.get(other_id).rating
            if abs(llm_rating - other_rating) <= effective_threshold:
                valid.append(other_id)

        # Add random-bot for models with low accuracy or no stats, regardless of rating
        if self._needs_random_bot(llm_id, player_stats):
            if self.RANDOM_BOT_ID in anchor_ids and self.RANDOM_BOT_ID not in valid:
                valid.append(self.RANDOM_BOT_ID)

        return valid

    def _pick_next_game(
        self,
        llm_ids: List[str],
        anchor_ids: List[str],
        games_per_pairing: Dict[Tuple[str, str], int],
        games_vs_anchor_per_color: int,
        games_vs_llm_per_color: int,
        rating_threshold: Optional[int],
    ) -> Optional[Tuple[str, str]]:
        """
        Pick the next game to play based on current ratings and costs.

        Prioritizes:
        1. Random-bot games first (for low-accuracy models to prove competence)
        2. Then all other games (anchors and LLMs mixed together)

        Cost-aware scheduling:
        - LLM priority: RD / (1 + 0.5 * cost) - prefers high RD and low cost
        - Opponent selection: rating_diff + min(50 * cost, 300) - prefers close rating and low cost

        Args:
            llm_ids: List of LLM IDs
            anchor_ids: List of anchor IDs
            games_per_pairing: Dict of (white_id, black_id) -> games played
            games_vs_anchor_per_color: Target games per LLM vs each anchor (per color)
            games_vs_llm_per_color: Target games per LLM pair (per color)
            rating_threshold: Max rating difference for valid opponents

        Returns:
            (white_id, black_id) tuple or None if no valid games remain
        """
        anchor_set = set(anchor_ids)

        # Cache player stats to avoid repeated computation in _needs_random_bot
        player_stats = self.stats_collector.get_player_stats()

        # Sort LLMs by cost-aware priority (highest first)
        # Priority = RD / (1 + COST_SENSITIVITY * cost) - prefers high RD and low cost
        llms_by_priority = sorted(
            llm_ids,
            key=lambda lid: self._calculate_priority(lid),
            reverse=True,
        )

        # Check if any random-bot games remain globally
        random_bot_games_remaining = False
        for llm_id in llm_ids:
            # Get valid opponents for this LLM (respects rating threshold)
            valid_opponents = self._get_valid_opponents(
                llm_id, anchor_ids, llm_ids, rating_threshold, player_stats
            )
            if self.RANDOM_BOT_ID in valid_opponents:
                for white_id, black_id in [(llm_id, self.RANDOM_BOT_ID), (self.RANDOM_BOT_ID, llm_id)]:
                    played = games_per_pairing.get((white_id, black_id), 0)
                    if played < games_vs_anchor_per_color:
                        random_bot_games_remaining = True
                        break
            if random_bot_games_remaining:
                break

        # Build phase list: random-bot first, then all other games (anchors and LLMs mixed)
        phases = []
        if random_bot_games_remaining:
            phases.append("random-bot")
        phases.append("other")

        for phase in phases:
            for llm_id in llms_by_priority:
                current_games = self._games_played.get(llm_id, 0)
                current_rd = self.rating_store.get(llm_id).games_rd

                # Check if this LLM is frozen (RD too low or old model with stable rating)
                if self._is_frozen(llm_id, current_rd):
                    continue  # Frozen models don't initiate games (but can be challenged)

                # Check if this LLM has hit its low RD cap (rating is stable enough)
                if current_rd < self.LOW_RD_THRESHOLD and current_games >= self.LOW_RD_CAP:
                    continue

                # Check if this LLM has hit its reasoning cap
                cap = self._get_game_cap(llm_id)
                if cap is not None and current_games >= cap:
                    continue

                # Get valid opponents based on current ratings
                valid_opponents = self._get_valid_opponents(
                    llm_id, anchor_ids, llm_ids, rating_threshold, player_stats
                )

                if not valid_opponents:
                    continue

                # Find opponents with games remaining
                candidates = []
                llm_rating = self.rating_store.get(llm_id).rating
                for opp_id in valid_opponents:
                    is_anchor = opp_id in anchor_set
                    is_random_bot = opp_id == self.RANDOM_BOT_ID

                    # Filter by phase:
                    # - random-bot phase: only random-bot
                    # - other phase: all opponents except random-bot
                    if phase == "random-bot" and not is_random_bot:
                        continue
                    if phase == "other" and is_random_bot:
                        continue

                    # Check if LLM opponent has hit their caps
                    if not is_anchor:
                        opp_current = self._games_played.get(opp_id, 0)
                        opp_rd = self.rating_store.get(opp_id).games_rd

                        # Frozen models can always be challenged - no cap
                        if not self._is_frozen(opp_id, opp_rd):
                            # Check low RD cap (60 <= RD < 70)
                            if opp_rd < self.LOW_RD_THRESHOLD and opp_current >= self.LOW_RD_CAP:
                                continue  # Skip - rating is stable enough

                            # Check reasoning cap
                            if opp_id in self.reasoning_ids:
                                opp_cap = self._get_game_cap(opp_id)
                                if opp_cap is not None and opp_current >= opp_cap:
                                    continue  # Skip capped reasoning model
                    target = games_vs_anchor_per_color if is_anchor else games_vs_llm_per_color

                    # Check both color combinations
                    opp_rating = self.rating_store.get(opp_id).rating
                    rating_diff = abs(llm_rating - opp_rating)
                    # Calculate cost-aware score for opponent selection
                    opp_cost = self._get_player_cost(opp_id)
                    cost_penalty = min(self.OPPONENT_COST_WEIGHT * opp_cost, self.OPPONENT_COST_CAP)
                    score = rating_diff + cost_penalty
                    for white_id, black_id in [(llm_id, opp_id), (opp_id, llm_id)]:
                        played = games_per_pairing.get((white_id, black_id), 0)
                        if played < target:
                            candidates.append((white_id, black_id, score))

                if candidates:
                    # Sort by cost-aware score (lower = better: close rating + cheap opponent)
                    candidates.sort(key=lambda c: c[2])
                    chosen = candidates[0]
                    return (chosen[0], chosen[1])

        return None

    async def _game_worker(
        self,
        worker_id: int,
        llm_ids: List[str],
        anchor_ids: List[str],
        games_per_pairing: Dict[Tuple[str, str], int],
        scheduler_lock: asyncio.Lock,
        games_vs_anchor_per_color: int,
        games_vs_llm_per_color: int,
        rating_threshold: Optional[int],
        results: List,
        counters: Dict[str, Any],
        max_cost: float,
    ) -> None:
        """
        Worker that continuously picks and plays games until none remain or budget exceeded.
        """
        llm_set = set(llm_ids)  # Constant for tracking which players are LLMs

        while True:
            # Pick next game under lock (covers pairing selection, game counts, and counters)
            async with scheduler_lock:
                # Check if budget exceeded
                if counters["budget_exceeded"]:
                    return  # Budget exceeded, stop

                pairing = self._pick_next_game(
                    llm_ids=llm_ids,
                    anchor_ids=anchor_ids,
                    games_per_pairing=games_per_pairing,
                    games_vs_anchor_per_color=games_vs_anchor_per_color,
                    games_vs_llm_per_color=games_vs_llm_per_color,
                    rating_threshold=rating_threshold,
                )

                if pairing is None:
                    return  # No more games

                white_id, black_id = pairing

                # Estimate cost BEFORE reserving to check budget
                estimated_cost = self._estimate_game_cost(white_id, black_id)
                effective_cost = counters["total_cost"] + counters["pending_cost"] + estimated_cost

                # Pre-flight budget check: don't start game if it would exceed budget
                if effective_cost >= max_cost:
                    counters["budget_exceeded"] = True
                    if self.verbose:
                        print(f"  Budget would be exceeded: ${counters['total_cost']:.2f} + ${counters['pending_cost']:.2f} pending + ${estimated_cost:.2f} new >= ${max_cost:.2f}")
                    return  # Don't start this game

                # Reserve everything atomically: pairing slot, game counts, counter, cost estimate
                games_per_pairing[(white_id, black_id)] = games_per_pairing.get((white_id, black_id), 0) + 1
                for player_id in [white_id, black_id]:
                    if player_id in llm_set:  # Track games for all LLMs
                        self._games_played[player_id] = self._games_played.get(player_id, 0) + 1
                counters["game_num"] += 1
                game_num = counters["game_num"]

                # Add estimated cost to pending
                counters["pending_cost"] += estimated_cost
                counters["pending_estimates"][game_num] = estimated_cost

            # Get player objects (outside lock - just dict lookup)
            try:
                white = self.players[white_id]
                black = self.players[black_id]
            except KeyError as e:
                if self.verbose:
                    print(f"Warning: Player {e} not found, skipping")
                async with scheduler_lock:
                    counters["errors"] += 1
                    games_per_pairing[(white_id, black_id)] -= 1
                    for player_id in [white_id, black_id]:
                        if player_id in llm_set:  # Track games for all LLMs
                            self._games_played[player_id] = max(0, self._games_played.get(player_id, 0) - 1)
                    # Remove pending cost estimate (use pop with default for thread safety)
                    estimate = counters["pending_estimates"].pop(game_num, 0)
                    counters["pending_cost"] -= estimate
                continue

            # Show current ratings
            if self.verbose:
                white_rating = self.rating_store.get(white_id)
                black_rating = self.rating_store.get(black_id)
                print(f"[{game_num}] {white_id} ({white_rating.rating:.0f}) vs "
                      f"{black_id} ({black_rating.rating:.0f})")

            # Create and run game task
            task = GameTask(white=white, black=black, game_num=game_num, total_games=0)

            try:
                result = await self.run_single_game(task)

                if result is None:
                    # API error - release the slots and pending cost
                    async with scheduler_lock:
                        games_per_pairing[(white_id, black_id)] -= 1
                        for player_id in [white_id, black_id]:
                            if player_id in llm_set:  # Track games for all LLMs
                                self._games_played[player_id] = max(0, self._games_played.get(player_id, 0) - 1)
                        counters["api_errors"] += 1
                        # Remove pending cost estimate (use pop with default for thread safety)
                        estimate = counters["pending_estimates"].pop(game_num, 0)
                        counters["pending_cost"] -= estimate
                else:
                    # Calculate actual game cost and update totals
                    game_cost = self._calculate_game_cost(result)
                    async with scheduler_lock:
                        # Remove pending estimate, add actual cost (use pop with default for thread safety)
                        estimate = counters["pending_estimates"].pop(game_num, 0)
                        counters["pending_cost"] -= estimate
                        counters["total_cost"] += game_cost
                        # Check budget (actual + estimated pending)
                        effective_cost = counters["total_cost"] + counters["pending_cost"]
                        if effective_cost >= max_cost:
                            counters["budget_exceeded"] = True
                            if self.verbose:
                                print(f"  Budget exceeded: ${counters['total_cost']:.2f} + ${counters['pending_cost']:.2f} pending >= ${max_cost:.2f}")
                    results.append(result)

            except Exception as e:
                if self.verbose:
                    print(f"  Game error: {e}")
                async with scheduler_lock:
                    games_per_pairing[(white_id, black_id)] -= 1
                    for player_id in [white_id, black_id]:
                        if player_id in llm_set:  # Track games for all LLMs
                            self._games_played[player_id] = max(0, self._games_played.get(player_id, 0) - 1)
                    counters["errors"] += 1
                    # Remove pending cost estimate (use pop with default for thread safety)
                    estimate = counters["pending_estimates"].pop(game_num, 0)
                    counters["pending_cost"] -= estimate

    # Default cost budget for benchmark runs
    DEFAULT_MAX_COST = 10.0  # $10 default budget

    async def run_benchmark(
        self,
        llm_ids: List[str],
        anchor_ids: List[str],
        games_vs_anchor_per_color: int = 10,
        games_vs_llm_per_color: int = 5,
        rating_threshold: Optional[int] = None,
        max_cost: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run the full benchmark with dynamic matchmaking.

        Games are scheduled dynamically based on current ratings.
        Runs up to max_concurrent games in parallel.
        Prioritizes LLMs with highest rating deviation.
        Stops when cost budget is exceeded.

        Args:
            llm_ids: List of LLM player IDs to benchmark
            anchor_ids: List of anchor engine IDs
            games_vs_anchor_per_color: Games per LLM vs each anchor (per color)
            games_vs_llm_per_color: Games per LLM pair (per color)
            rating_threshold: Only pair players within this rating difference
            max_cost: Maximum cost budget in dollars (default: $15)

        Returns:
            Benchmark results summary
        """
        if max_cost is None:
            max_cost = self.DEFAULT_MAX_COST
        # Reset trackers
        self._games_played = {}
        self._cost_cache = {}  # Clear cost cache for fresh calculations
        self._cost_data_cache = None  # Clear cost data cache for fresh calculations
        games_per_pairing: Dict[Tuple[str, str], int] = {}
        scheduler_lock = asyncio.Lock()  # Single lock for all scheduling state

        print(f"Starting benchmark (dynamic matchmaking)")
        print(f"LLMs: {llm_ids}")
        print(f"Anchors: {anchor_ids}")
        print(f"Max concurrent: {self.max_concurrent}")
        if rating_threshold is not None:
            print(f"Rating threshold: ±{rating_threshold}")
        print(f"Target games per anchor (per color): {games_vs_anchor_per_color}")
        print(f"Target games per LLM pair (per color): {games_vs_llm_per_color}")
        if self.reasoning_ids:
            reasoning_in_benchmark = [lid for lid in llm_ids if lid in self.reasoning_ids]
            print(f"Reasoning models ({len(reasoning_in_benchmark)}): {reasoning_in_benchmark}")
            print(f"Reasoning game caps: {self.REASONING_BASE_CAP} (base), {self.REASONING_HIGH_RATING_CAP} (if rating > {self.REASONING_HIGH_RATING_THRESHOLD})")
        print(f"Low RD cap: {self.LOW_RD_CAP} games if RD < {self.LOW_RD_THRESHOLD}")
        if rating_threshold is not None:
            print(f"Stable model pairing: RD < {self.STABLE_RD_THRESHOLD} uses ±{self.STABLE_RATING_THRESHOLD} threshold (vs ±{rating_threshold})")
        print(f"Frozen: RD < {self.FROZEN_RD_THRESHOLD}, or RD < {self.FROZEN_AGE_RD_THRESHOLD_6M} + >{self.FROZEN_AGE_MONTHS_6M}mo old, or RD < {self.FROZEN_AGE_RD_THRESHOLD_1Y} + >{self.FROZEN_AGE_MONTHS_1Y}mo old")
        print(f"Within-year weak freeze: RD < {self.WITHIN_YEAR_WEAK_RD_THRESHOLD} + rating < {self.WITHIN_YEAR_WEAK_RATING_THRESHOLD} (any model <{self.FROZEN_AGE_MONTHS_1Y}mo)")
        print(f"Recent weak freeze (<{self.FROZEN_AGE_MONTHS_6M}mo): reasoning RD < {self.RECENT_WEAK_REASONING_RD_THRESHOLD} + rating < {self.RECENT_WEAK_REASONING_RATING_THRESHOLD}, non-reasoning RD < {self.RECENT_WEAK_NONREASONING_RD_THRESHOLD} + rating < {self.RECENT_WEAK_NONREASONING_RATING_THRESHOLD}")
        print(f"Cost budget: ${max_cost:.2f}")
        print()

        # Shared state for workers
        results: List[GameResult] = []
        counters: Dict[str, Any] = {
            "game_num": 0,
            "errors": 0,
            "api_errors": 0,
            "total_cost": 0.0,
            "pending_cost": 0.0,  # Estimated cost of running games
            "pending_estimates": {},  # game_num -> estimated cost
            "budget_exceeded": False,
        }

        # Launch worker tasks
        workers = [
            self._game_worker(
                worker_id=i,
                llm_ids=llm_ids,
                anchor_ids=anchor_ids,
                games_per_pairing=games_per_pairing,
                scheduler_lock=scheduler_lock,
                games_vs_anchor_per_color=games_vs_anchor_per_color,
                games_vs_llm_per_color=games_vs_llm_per_color,
                rating_threshold=rating_threshold,
                results=results,
                counters=counters,
                max_cost=max_cost,
            )
            for i in range(self.max_concurrent)
        ]

        await asyncio.gather(*workers)

        # Verify all pending estimates were cleaned up (defensive check)
        if counters["pending_estimates"]:
            print(f"Warning: {len(counters['pending_estimates'])} pending estimates not cleaned up")
        if abs(counters["pending_cost"]) > 0.01:
            print(f"Warning: pending_cost not zero: ${counters['pending_cost']:.4f}")

        # Show completion message
        if counters["budget_exceeded"]:
            print(f"\nBenchmark stopped: cost budget exceeded (${counters['total_cost']:.2f} / ${max_cost:.2f})")
        else:
            print(f"\nBenchmark complete: {len(results)} games, ${counters['total_cost']:.2f} spent")
        print(f"Errors: {counters['errors']}, API errors: {counters['api_errors']}")

        # Show final ratings for all LLMs
        print("\nFinal ratings:")
        for llm_id in sorted(llm_ids, key=lambda x: self.rating_store.get(x).rating, reverse=True):
            r = self.rating_store.get(llm_id)
            games = self._games_played.get(llm_id, 0)
            print(f"  {llm_id}: {r.rating:.0f} ±{r.rating_deviation:.0f} (games_rd={r.games_rd:.0f}, {games} games)")

        return {
            "total_games": counters["game_num"],
            "completed_games": len(results),
            "errors": counters["errors"],
            "api_errors": counters["api_errors"],
            "total_cost": counters["total_cost"],
            "budget_exceeded": counters["budget_exceeded"],
            "results": results,
        }

    async def run_single_matchup(
        self,
        white_id: str,
        black_id: str,
        num_games: int = 1,
    ) -> List[GameResult]:
        """
        Run games between two specific players.

        Args:
            white_id: White player ID
            black_id: Black player ID
            num_games: Number of games to play

        Returns:
            List of GameResult
        """
        try:
            white = self.players[white_id]
            black = self.players[black_id]
        except KeyError as e:
            raise ValueError(f"Player {e} not found in players dict") from e

        tasks = [
            GameTask(white=white, black=black, game_num=i, total_games=num_games)
            for i in range(1, num_games + 1)
        ]

        results = await asyncio.gather(
            *[self.run_single_game(task) for task in tasks],
            return_exceptions=True,
        )

        return [r for r in results if r is not None and not isinstance(r, Exception)]
