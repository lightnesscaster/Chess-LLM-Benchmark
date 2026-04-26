"""
Match scheduler for running benchmark games.

Handles:
- Pairing generation (LLM vs anchors, LLM vs LLM)
- Parallel game execution
- Rating updates after games
"""

import asyncio
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Union, Tuple, Optional
from dataclasses import dataclass

from engines.base_engine import BaseEngine
from llm.base_llm import BaseLLMPlayer
from .game_runner import GameRunner
from .freeze_checker import FreezeChecker
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

    # Recent weak model freezing (released after Nov 2025)
    # Reasoning models: freeze at RD < 100 if rating < 1000
    RECENT_WEAK_REASONING_RD_THRESHOLD = 100
    RECENT_WEAK_REASONING_RATING_THRESHOLD = 1000
    # Non-reasoning models: freeze at RD < 100 if rating < 500
    RECENT_WEAK_NONREASONING_RD_THRESHOLD = 100
    RECENT_WEAK_NONREASONING_RATING_THRESHOLD = 500
    RECENT_WEAK_CUTOFF = datetime(2025, 11, 1, tzinfo=timezone.utc).timestamp()

    # Within-year weak model freezing (released after Apr 2025)
    # Any model: freeze at RD < 100 if rating < 0
    WITHIN_YEAR_WEAK_RD_THRESHOLD = 100
    WITHIN_YEAR_WEAK_RATING_THRESHOLD = 0
    WITHIN_YEAR_WEAK_CUTOFF = datetime(2025, 4, 1, tzinfo=timezone.utc).timestamp()

    # Legal move rate threshold - models below this must play random-bot (if rated above -200)
    LEGAL_MOVE_RATE_THRESHOLD = 0.98  # 98% accuracy
    LOW_ACCURACY_RATING_THRESHOLD = -200  # Only enforce accuracy requirement above this rating
    RANDOM_BOT_MIN_GAMES = 5  # Min games vs random-bot to prove competence (must not lose any)
    RANDOM_BOT_ID = "random-bot"

    # Proven worse freezing
    # Freeze if: RD < 80, 3+ losses, and another model (any provider) with RD < 80 and 3+ losses
    # has a rating advantage exceeding the combined RD of both models
    PROVEN_WORSE_RD_THRESHOLD = 80
    PROVEN_WORSE_MIN_LOSSES = 3
    PROVEN_WORSE_TIME_WINDOW = 2  # months after this model's release to consider

    # Inferior effort variant freezing
    # Freeze if: higher effort variant is rated lower than a lower effort sibling (same model_id),
    # both have RD < 150 and both have lost >= 3 games
    EFFORT_VARIANT_RD_THRESHOLD = 150
    EFFORT_VARIANT_MIN_LOSSES = 3
    EFFORT_LEVELS = {
        "no thinking": 0,
        "minimal": 1,
        "low": 2,
        "medium": 3,
        "thinking": 4,
        "high": 4,
        "max": 5,
        "xhigh": 5,
    }

    # Inferior same-provider model freezing
    # Freeze if: RD < 100, lost >= 3 games, rated lower than a same-provider model
    # that was released before or within 3 months after, unless 2x cheaper
    PROVIDER_INFERIOR_RD_THRESHOLD = 100
    PROVIDER_INFERIOR_MIN_LOSSES = 3
    PROVIDER_INFERIOR_TIME_WINDOW = 3  # months after this model's release to consider
    PROVIDER_INFERIOR_COST_RATIO = 2.0  # must be this much cheaper to avoid freeze

    # Expensive inferior model freezing (cross-provider)
    # Freeze if: RD < 150, lost >= 3 games, and a stronger model exists
    # (released before or within 3 months after) that is at least 2x cheaper
    EXPENSIVE_INFERIOR_RD_THRESHOLD = 150
    EXPENSIVE_INFERIOR_MIN_LOSSES = 3
    EXPENSIVE_INFERIOR_TIME_WINDOW = 3  # months after this model's release to consider
    EXPENSIVE_INFERIOR_COST_RATIO = 2.0  # stronger model must be this much cheaper

    # Lost to much weaker model freezing
    # Freeze if: lost to a model rated 600+ points lower
    # (released before or within 3 months after), unless this model is 5x cheaper
    LOST_TO_WEAKER_RATING_GAP = 600
    LOST_TO_WEAKER_TIME_WINDOW = 3  # months after this model's release to consider
    LOST_TO_WEAKER_COST_RATIO = 5.0  # must be this much cheaper to avoid freeze

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
        llm_configs: Optional[Dict[str, dict]] = None,
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
            llm_configs: Dict mapping player_id to LLM config dict (for position benchmarks)
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
        self._llm_configs = llm_configs or {}

        # Track which models already have position benchmark results
        self._benchmark_completed = self._load_existing_benchmark_results()

        # Semaphore for concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Lock for rating updates
        self._rating_lock = asyncio.Lock()

        # Track games played per LLM during benchmark (for reasoning caps and low RD caps)
        # Note: Only used in run_benchmark() flow, protected by scheduler_lock
        self._games_played: Dict[str, int] = {}

        # Pre-estimated RD: tracks projected RD accounting for in-flight games.
        # RD reduction in Glicko-2 is result-independent, so we can predict it
        # before games complete. This prevents scheduling wasted games for models
        # whose RD will drop below freeze/cap thresholds once in-flight games finish.
        self._estimated_rd: Dict[str, float] = {}
        # Track in-flight opponents per player so we can recompute estimated RD
        # correctly when any single game completes or fails (without losing info
        # about other in-flight games). Maps player_id -> {game_num: opponent_id}.
        self._inflight_opponents: Dict[str, Dict[int, str]] = {}

        # Load model publish dates for age-based freezing
        self._publish_dates: Dict[str, int] = {}  # player_id -> timestamp
        self._player_providers: Dict[str, str] = {}  # player_id -> provider name
        self._models_by_provider: Dict[str, List[str]] = {}  # provider -> [player_ids]
        self._player_model_ids: Dict[str, str] = {}  # player_id -> model_id
        self._models_by_model_id: Dict[str, List[str]] = {}  # model_id -> [player_ids]
        self._load_publish_dates()

        # Freeze checker (owns freeze logic and cost estimation)
        engine_ids = {pid for pid, p in players.items() if isinstance(p, BaseEngine)}
        self._freeze_checker = FreezeChecker(
            rating_store, stats_collector, reasoning_ids, engine_ids
        )

        # Cost calculator for cost-aware scheduling (delegates to freeze checker for per-player costs)
        self._cost_calculator = CostCalculator()
        self._pairwise_cost_cache: Optional[Dict[Tuple[str, str], float]] = None  # Pairwise game costs

    def _load_publish_dates(self) -> None:
        """Load model publish dates and provider info from data file."""
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
            pass  # No publish dates available

    def _load_existing_benchmark_results(self) -> set:
        """Load player IDs that already have position benchmark results."""
        results_path = Path(__file__).parent.parent / "position_benchmark" / "results.json"
        try:
            if results_path.exists():
                with open(results_path) as f:
                    data = json.load(f)
                return set(data.keys())
        except (json.JSONDecodeError, OSError):
            pass
        return set()

    def _needs_position_benchmark(self, player_id: str) -> bool:
        """Check if a player needs a position benchmark run."""
        # Engines don't need benchmarks
        if isinstance(self.players.get(player_id), BaseEngine):
            return False
        # Already has results
        if player_id in self._benchmark_completed:
            return False
        # No config available (can't run benchmark without it)
        if player_id not in self._llm_configs:
            return False
        return True

    @staticmethod
    def _format_player_list(player_ids: List[str], limit: int = 12) -> str:
        """Format a bounded player list for scheduler logs."""
        if len(player_ids) <= limit:
            return ", ".join(player_ids)
        shown = ", ".join(player_ids[:limit])
        return f"{shown}, ... ({len(player_ids) - limit} more)"

    @staticmethod
    def _format_rating_snapshot(rating: PlayerRating) -> str:
        """Format rating state for skip logs."""
        return (
            f"rating={rating.rating:.0f}, RD={rating.rating_deviation:.0f}, "
            f"games={rating.games_played}, W-L-D={rating.wins}-{rating.losses}-{rating.draws}"
        )

    def _estimate_benchmark_cost(self, player_id: str) -> float:
        """
        Estimate cost of running position benchmark for a model.

        50 equal positions, ~1500 prompt tokens each, ~10-200 completion tokens.
        """
        override = self._cost_calculator.get_budget_cost_override(player_id)
        if override is not None:
            return override

        model = self._cost_calculator.get_model_for_player(player_id)
        if not model:
            return 0.0
        pricing = self._cost_calculator.get_pricing(model)
        if not pricing:
            return 0.0

        comp_est = 200 if player_id in self.reasoning_ids else 10

        return 50 * 1500 * pricing.get("prompt", 0) + 50 * comp_est * pricing.get("completion", 0)

    async def _run_position_benchmarks(
        self,
        llm_ids: List[str],
        counters: Dict[str, Any],
        max_cost: float,
    ) -> set:
        """
        Phase 1: Run position benchmarks for models missing results.

        Runs sequentially with a shared Stockfish instance.
        Each benchmark counts as 1 game for the model.

        Returns:
            Set of player IDs that were successfully benchmarked.
        """
        import chess.engine as ce
        from position_benchmark.run_benchmark import run_benchmark_for_scheduler

        benchmarked = set()

        already_benchmarked = []
        missing_config = []
        frozen = []
        eligible = []

        for llm_id in llm_ids:
            if isinstance(self.players.get(llm_id), BaseEngine):
                continue
            if llm_id in self._benchmark_completed:
                already_benchmarked.append(llm_id)
                continue
            if llm_id not in self._llm_configs:
                missing_config.append(llm_id)
                continue

            rating = self.rating_store.get(llm_id)
            if self._freeze_checker.is_frozen(llm_id, rating.rating_deviation):
                frozen.append((llm_id, self._format_rating_snapshot(rating)))
                continue

            eligible.append(llm_id)

        print(
            "Phase 1: Position benchmark precheck: "
            f"{len(eligible)} queued, "
            f"{len(already_benchmarked)} already done, "
            f"{len(frozen)} frozen, "
            f"{len(missing_config)} missing config"
        )

        if self.verbose and already_benchmarked:
            print(f"  Already benchmarked: {self._format_player_list(already_benchmarked)}")
        for llm_id, snapshot in frozen:
            print(f"  Skipping position benchmark for {llm_id}: frozen ({snapshot})")
        if missing_config:
            print(
                "  Skipping position benchmark for missing config: "
                f"{self._format_player_list(missing_config)}"
            )

        if not eligible:
            print("  No position benchmarks to run.")
            print()
            return benchmarked

        print(f"Phase 1: Position benchmarks for {len(eligible)} model(s)")

        # Sort by priority (highest RD / lowest cost first)
        eligible.sort(key=lambda lid: self._calculate_priority(lid), reverse=True)

        # Load positions and open Stockfish once
        positions_path = Path(__file__).parent.parent / "position_benchmark" / "positions.json"
        if not positions_path.exists():
            print(f"  Warning: {positions_path} not found, skipping position benchmarks")
            return benchmarked

        with open(positions_path) as f:
            positions_data = json.load(f)
        all_positions = positions_data["positions"]
        # Only run equal positions — blunder positions don't contribute to rating prediction.
        # Track original indices so results.json entries have correct position_idx values
        # (needed by _load_benchmark_predictions which looks up pos_type by index).
        equal_indices = [i for i, p in enumerate(all_positions) if p.get("type") == "equal"]
        positions = [all_positions[i] for i in equal_indices]

        stockfish = None
        try:
            stockfish = ce.SimpleEngine.popen_uci("stockfish")

            for llm_id in eligible:
                # Check budget
                estimated_cost = self._estimate_benchmark_cost(llm_id)
                if counters["total_cost"] + estimated_cost >= max_cost:
                    print(f"  Skipping {llm_id}: would exceed budget "
                          f"(${counters['total_cost']:.2f} + ${estimated_cost:.2f} >= ${max_cost:.2f})")
                    continue

                config = self._llm_configs[llm_id]
                print(f"  Running position benchmark for {llm_id}...")

                result = await run_benchmark_for_scheduler(
                    player_id=llm_id,
                    player_config=config,
                    stockfish=stockfish,
                    positions=positions,
                    depth=30,
                    original_indices=equal_indices,
                )

                if result["success"]:
                    summary = result["summary"]
                    token_usage = result.get("token_usage", {"prompt": 0, "completion": 0})

                    # Calculate actual cost from token usage
                    model_name = self._cost_calculator.get_model_for_player(llm_id) or ""
                    tokens = {
                        "prompt_tokens": token_usage.get("prompt", 0),
                        "completion_tokens": token_usage.get("completion", 0),
                    }
                    override = self._cost_calculator.get_budget_cost_override(llm_id)
                    if override is not None:
                        actual_cost = override
                    else:
                        actual_cost = self._cost_calculator.calculate_game_cost(tokens, model_name) or 0.0

                    counters["total_cost"] += actual_cost
                    self._benchmark_completed.add(llm_id)
                    benchmarked.add(llm_id)

                    # Count as 1 game played (reduces remaining game cap by 1 for reasoning models)
                    self._games_played[llm_id] = self._games_played.get(llm_id, 0) + 1

                    print(f"  Position benchmark for {llm_id}: "
                          f"CPL={summary['avg_cpl']:.0f}, "
                          f"Legal={summary['legal_pct']:.1f}%, "
                          f"Cost=${actual_cost:.4f}")
                else:
                    print(f"  Warning: Position benchmark failed for {llm_id}: {result.get('error', 'unknown')}")

        finally:
            if stockfish is not None:
                stockfish.quit()

        # Refresh rating store once with all new benchmark predictions
        if benchmarked:
            self.rating_store.refresh_benchmark_predictions()

        print()
        return benchmarked

    # Default cost for models with unknown pricing (conservative estimate)
    UNKNOWN_MODEL_DEFAULT_COST = 1.0  # $1.00 per game

    def invalidate_cost_cache(self) -> None:
        """Invalidate all cost caches. Call this when ratings are updated."""
        self._freeze_checker.invalidate_cost_cache()
        self._pairwise_cost_cache = None


    def _calculate_player_game_cost(self, player_id: str, tokens: Optional[dict]) -> float:
        """Calculate benchmark budget cost for one player's side of a game."""
        if not tokens:
            return 0.0

        override = self._cost_calculator.get_budget_cost_override(player_id)
        if override is not None:
            return override

        cost = self._cost_calculator.calculate_game_cost(
            tokens,
            self._cost_calculator.get_model_for_player(player_id) or "",
        )
        return cost or 0.0

    def _calculate_game_cost(self, result: GameResult) -> float:
        """
        Calculate the cost of a completed game from token usage.

        Args:
            result: The game result with token data

        Returns:
            Total cost in dollars for both players
        """
        return (
            self._calculate_player_game_cost(result.white_id, result.tokens_white)
            + self._calculate_player_game_cost(result.black_id, result.tokens_black)
        )

    def _estimate_player_cost_detail(self, player_id: str) -> Tuple[float, str]:
        """Estimate one player's budget cost and describe the source."""
        override = self._cost_calculator.get_budget_cost_override(player_id)
        if override is not None:
            return override, f"{player_id}: config budget_cost_per_game"

        return self._freeze_checker.get_player_cost(player_id), f"{player_id}: historical/pricing estimate"

    def _estimate_game_cost_detail(self, white_id: str, black_id: str) -> Tuple[float, str]:
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
            Estimated cost in dollars for the game, plus source description
        """
        white_override = self._cost_calculator.get_budget_cost_override(white_id)
        black_override = self._cost_calculator.get_budget_cost_override(black_id)
        if white_override is not None or black_override is not None:
            white_cost, white_source = self._estimate_player_cost_detail(white_id)
            black_cost, black_source = self._estimate_player_cost_detail(black_id)
            return white_cost + black_cost, f"{white_source}; {black_source}"

        # Build pairwise cost cache if needed
        if self._pairwise_cost_cache is None:
            self._pairwise_cost_cache = self._build_pairwise_cost_cache()

        # Try pairwise cost first (order-independent key)
        pair_key = tuple(sorted([white_id, black_id]))
        if pair_key in self._pairwise_cost_cache:
            return self._pairwise_cost_cache[pair_key], "historical pair average"

        # Fall back to sum of individual player costs
        white_cost, white_source = self._estimate_player_cost_detail(white_id)
        black_cost, black_source = self._estimate_player_cost_detail(black_id)
        return white_cost + black_cost, f"{white_source}; {black_source}"

    def _estimate_game_cost(self, white_id: str, black_id: str) -> float:
        """Estimate cost for a game between two players before it starts."""
        estimated_cost, _ = self._estimate_game_cost_detail(white_id, black_id)
        return estimated_cost

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
        rd = self._get_estimated_rd(player_id)
        cost = self._freeze_checker.get_player_cost(player_id)
        return rd / (1 + self.COST_SENSITIVITY * cost)

    def _estimate_rd_after_game(self, player_id: str, opponent_id: str, current_rd: float) -> float:
        """
        Estimate RD after playing a game against opponent, using Glicko-2 variance formula.

        RD reduction in Glicko-2 is result-independent — it depends only on opponent
        strength, not the game outcome. This allows accurate pre-estimation.

        Args:
            player_id: The player whose RD to estimate
            opponent_id: The opponent
            current_rd: The player's current (possibly already estimated) RD

        Returns:
            Projected RD after the game (with floor of 45)
        """
        player_data = self.rating_store.get(player_id)
        opp_data = self.rating_store.get(opponent_id)

        mu, _ = self.glicko.glicko_to_glicko2(player_data.rating, current_rd)
        phi = current_rd / self.glicko.SCALE_FACTOR
        opp_mu, opp_phi = self.glicko.glicko_to_glicko2(opp_data.rating, opp_data.rating_deviation)

        v = self.glicko.calculate_variance(mu, [(opp_mu, opp_phi)])
        new_phi = 1.0 / math.sqrt(1.0 / (phi * phi) + 1.0 / v)
        new_rd = new_phi * self.glicko.SCALE_FACTOR

        return max(45.0, new_rd)

    def _get_estimated_rd(self, player_id: str) -> float:
        """Get estimated RD for a player, falling back to actual RD from rating store."""
        if player_id in self._estimated_rd:
            return self._estimated_rd[player_id]
        return self.rating_store.get(player_id).rating_deviation

    def _recompute_estimated_rd(self, player_id: str) -> None:
        """Recompute estimated RD from actual RD plus all remaining in-flight games."""
        rd = self.rating_store.get(player_id).rating_deviation
        for opp_id in self._inflight_opponents.get(player_id, {}).values():
            rd = self._estimate_rd_after_game(player_id, opp_id, rd)
        self._estimated_rd[player_id] = rd

    def _get_freeze_test_opponent(
        self,
        player_id: str,
        valid_opponents: List[str],
        games_per_pairing: Dict[Tuple[str, str], int],
        anchor_set: set,
        games_vs_anchor_per_color: int,
        games_vs_llm_per_color: int,
    ) -> Optional[str]:
        """
        Find the best opponent to test whether player_id would trigger
        the _lost_to_much_weaker freeze rule.

        Returns the strongest valid opponent rated below the freeze ceiling
        (max peer rating - 600), so that a single loss would trigger the freeze.
        Returns None if no such test is useful (no peers, already frozen, etc).
        """
        my_timestamp = self._publish_dates.get(player_id)
        if my_timestamp is None:
            return None

        my_cost = self._freeze_checker.get_player_cost(player_id)
        # Free models are always exempt from lost-to-weaker freeze
        if my_cost == 0:
            return None

        # If already lost to a freeze-triggering opponent, no need to front-load
        if self._freeze_checker.lost_to_much_weaker(player_id):
            return None

        three_months = self.LOST_TO_WEAKER_TIME_WINDOW * 30.44 * 24 * 60 * 60

        # Find qualifying peers and compute freeze ceiling
        max_peer_rating = None
        for peer_id, peer_timestamp in self._publish_dates.items():
            if peer_id == player_id:
                continue
            if peer_timestamp > my_timestamp + three_months:
                continue
            if not self.rating_store.has_player(peer_id):
                continue

            peer_cost = self._freeze_checker.get_player_cost(peer_id)
            # Cost exception: if player is 5x cheaper than this peer, skip peer
            if peer_cost > 0 and my_cost * self.LOST_TO_WEAKER_COST_RATIO <= peer_cost:
                continue

            peer_rating = self.rating_store.get(peer_id).rating
            if max_peer_rating is None or peer_rating > max_peer_rating:
                max_peer_rating = peer_rating

        if max_peer_rating is None:
            return None

        freeze_ceiling = max_peer_rating - self.LOST_TO_WEAKER_RATING_GAP

        # Find the best freeze-test opponent.
        # Prefer in-band opponents (within 200 points of ceiling) for maximum signal,
        # but use cheaper "filter" opponents first if the in-band option is 5x+ more
        # expensive. Filter opponents must be rated between (player_rating - 200) and
        # ceiling, and must not have already played 3+ games against this player.
        FREEZE_TEST_BAND = 200
        FILTER_COST_RATIO = 5.0
        FILTER_MAX_GAMES = 3

        my_rating = self.rating_store.get(player_id).rating
        filter_floor = my_rating - FREEZE_TEST_BAND

        eligible_in_band = []
        eligible_filter = []

        for opp_id in valid_opponents:
            if not self.rating_store.has_player(opp_id):
                continue
            opp_rating = self.rating_store.get(opp_id).rating
            if opp_rating >= freeze_ceiling:
                continue
            # Check if any color combination has games remaining
            target = games_vs_anchor_per_color if opp_id in anchor_set else games_vs_llm_per_color
            has_games = False
            for w, b in [(player_id, opp_id), (opp_id, player_id)]:
                if games_per_pairing.get((w, b), 0) < target:
                    has_games = True
                    break
            if not has_games:
                continue

            opp_cost = self._freeze_checker.get_player_cost(opp_id)
            total_cost = my_cost + opp_cost

            if opp_rating >= freeze_ceiling - FREEZE_TEST_BAND:
                eligible_in_band.append((opp_id, opp_rating, total_cost))
            elif opp_rating >= filter_floor:
                # Count existing games against this opponent
                games_played = 0
                for result in self.stats_collector.results:
                    if (result.white_id == player_id and result.black_id == opp_id) or \
                       (result.black_id == player_id and result.white_id == opp_id):
                        games_played += 1
                if games_played < FILTER_MAX_GAMES:
                    eligible_filter.append((opp_id, opp_rating, total_cost))

        if not eligible_in_band and not eligible_filter:
            return None

        # Pick cheapest in-band opponent
        if eligible_in_band:
            eligible_in_band.sort(key=lambda e: e[2])
            best_in_band = eligible_in_band[0]
        else:
            best_in_band = None

        # Pick cheapest filter opponent
        if eligible_filter:
            eligible_filter.sort(key=lambda e: e[2])
            best_filter = eligible_filter[0]
        else:
            best_filter = None

        # Use filter opponent if in-band is 5x+ more expensive
        if best_in_band and best_filter:
            if best_in_band[2] >= FILTER_COST_RATIO * best_filter[2]:
                best_opp = best_filter[0]
            else:
                best_opp = best_in_band[0]
        elif best_in_band:
            best_opp = best_in_band[0]
        else:
            best_opp = best_filter[0]

        # If we've already completed a game against this opponent, the test is
        # done: either we lost (caught by _lost_to_much_weaker above) or we won.
        best_opp_rating = self.rating_store.get(best_opp).rating
        for result in self.stats_collector.results:
            if (result.white_id == player_id and result.black_id == best_opp) or \
               (result.black_id == player_id and result.white_id == best_opp):
                return None
            # If we've already beaten an opponent rated 200+ above the freeze
            # test opponent, the test is redundant — we've proven strength.
            won_as_white = (result.white_id == player_id and result.winner == "white")
            won_as_black = (result.black_id == player_id and result.winner == "black")
            if won_as_white or won_as_black:
                beaten_id = result.black_id if won_as_white else result.white_id
                if self.rating_store.has_player(beaten_id):
                    beaten_rating = self.rating_store.get(beaten_id).rating
                    if beaten_rating >= best_opp_rating + 200:
                        return None

        return best_opp

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
        llm_rd = self._get_estimated_rd(llm_id)

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

        # Freeze-test priority: before any other scheduling, check if an unfrozen
        # model should be tested against a weak opponent that would trigger the
        # _lost_to_much_weaker freeze rule. This front-loads the test so budget
        # isn't wasted on models that would get frozen after a single loss.
        # Models with a freeze-test in flight are blocked from all other games
        # until the test resolves.
        freeze_test_pending = set()
        for llm_id in llms_by_priority:
            current_games = self._games_played.get(llm_id, 0)
            current_rd = self._get_estimated_rd(llm_id)
            if self._freeze_checker.is_frozen(llm_id, current_rd, player_stats):
                continue
            if current_rd < self.LOW_RD_THRESHOLD and current_games >= self.LOW_RD_CAP:
                continue
            cap = self._get_game_cap(llm_id)
            if cap is not None and current_games >= cap:
                continue
            valid_opponents = self._get_valid_opponents(
                llm_id, anchor_ids, llm_ids, rating_threshold, player_stats
            )
            if not valid_opponents:
                continue
            freeze_opp = self._get_freeze_test_opponent(
                llm_id, valid_opponents, games_per_pairing,
                anchor_set, games_vs_anchor_per_color, games_vs_llm_per_color,
            )
            if freeze_opp is not None:
                # Check if already scheduled (in flight) vs needs scheduling
                already_in_flight = (
                    games_per_pairing.get((llm_id, freeze_opp), 0) > 0 or
                    games_per_pairing.get((freeze_opp, llm_id), 0) > 0
                )
                if already_in_flight:
                    freeze_test_pending.add(llm_id)
                else:
                    ft = games_vs_anchor_per_color if freeze_opp in anchor_set else games_vs_llm_per_color
                    for w, b in [(llm_id, freeze_opp), (freeze_opp, llm_id)]:
                        if games_per_pairing.get((w, b), 0) < ft:
                            return (w, b)

        # Build phase list: random-bot first, then all other games (anchors and LLMs mixed)
        phases = []
        if random_bot_games_remaining:
            phases.append("random-bot")
        phases.append("other")

        for phase in phases:
            for llm_id in llms_by_priority:
                # Skip models waiting for their freeze-test game to complete
                if llm_id in freeze_test_pending:
                    continue

                current_games = self._games_played.get(llm_id, 0)
                current_rd = self._get_estimated_rd(llm_id)

                # Check if this LLM is frozen (RD too low or old model with stable rating)
                if self._freeze_checker.is_frozen(llm_id, current_rd, player_stats):
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
                        opp_rd = self._get_estimated_rd(opp_id)

                        # Frozen models can always be challenged - no cap
                        if not self._freeze_checker.is_frozen(opp_id, opp_rd, player_stats):
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
                    opp_cost = self._freeze_checker.get_player_cost(opp_id)
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
                estimated_cost, estimate_source = self._estimate_game_cost_detail(white_id, black_id)
                effective_cost = counters["total_cost"] + counters["pending_cost"] + estimated_cost

                # Pre-flight budget check: don't start game if it would exceed budget
                if effective_cost >= max_cost:
                    counters["budget_exceeded"] = True
                    if self.verbose:
                        print(
                            f"  Budget would be exceeded by {white_id} vs {black_id}: "
                            f"${counters['total_cost']:.2f} + ${counters['pending_cost']:.2f} pending "
                            f"+ ${estimated_cost:.2f} new >= ${max_cost:.2f} "
                            f"({estimate_source})"
                        )
                    return  # Don't start this game

                # Reserve everything atomically: pairing slot, game counts, counter, cost estimate
                games_per_pairing[(white_id, black_id)] = games_per_pairing.get((white_id, black_id), 0) + 1
                for player_id in [white_id, black_id]:
                    if player_id in llm_set:  # Track games for all LLMs
                        self._games_played[player_id] = self._games_played.get(player_id, 0) + 1
                counters["game_num"] += 1
                game_num = counters["game_num"]

                # Track in-flight opponents and update estimated RD for LLMs
                # RD reduction is result-independent, so we can predict it before the game completes
                for player_id, opp_id in [(white_id, black_id), (black_id, white_id)]:
                    if player_id in llm_set:
                        self._inflight_opponents.setdefault(player_id, {})[game_num] = opp_id
                        est_rd = self._get_estimated_rd(player_id)
                        self._estimated_rd[player_id] = self._estimate_rd_after_game(
                            player_id, opp_id, est_rd
                        )

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
                            # Remove this game from in-flight tracking and recompute estimated RD
                            self._inflight_opponents.get(player_id, {}).pop(game_num, None)
                            self._recompute_estimated_rd(player_id)
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
                                # Remove this game from in-flight tracking and recompute estimated RD
                                self._inflight_opponents.get(player_id, {}).pop(game_num, None)
                                self._recompute_estimated_rd(player_id)
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
                        # Remove this game from in-flight tracking and recompute estimated RD
                        # (actual RD now reflects this game; recompute layers remaining in-flight games on top)
                        for player_id in [white_id, black_id]:
                            if player_id in llm_set:
                                self._inflight_opponents.get(player_id, {}).pop(game_num, None)
                                self._recompute_estimated_rd(player_id)
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
                            # Remove this game from in-flight tracking and recompute estimated RD
                            self._inflight_opponents.get(player_id, {}).pop(game_num, None)
                            self._recompute_estimated_rd(player_id)
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
        self._estimated_rd = {}
        self._inflight_opponents = {}
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
        print(f"Within-year weak freeze: RD < {self.WITHIN_YEAR_WEAK_RD_THRESHOLD} + rating < {self.WITHIN_YEAR_WEAK_RATING_THRESHOLD} (models after Apr 2025)")
        print(f"Recent weak freeze (after Nov 2025): reasoning RD < {self.RECENT_WEAK_REASONING_RD_THRESHOLD} + rating < {self.RECENT_WEAK_REASONING_RATING_THRESHOLD}, non-reasoning RD < {self.RECENT_WEAK_NONREASONING_RD_THRESHOLD} + rating < {self.RECENT_WEAK_NONREASONING_RATING_THRESHOLD}")
        print(f"Proven worse freeze: RD < {self.PROVEN_WORSE_RD_THRESHOLD} + "
              f">={self.PROVEN_WORSE_MIN_LOSSES} losses + another model (RD < {self.PROVEN_WORSE_RD_THRESHOLD}, "
              f">={self.PROVEN_WORSE_MIN_LOSSES} losses, within {self.PROVEN_WORSE_TIME_WINDOW}mo) "
              f"rated higher by combined RD")
        print(f"Effort variant freeze: RD < {self.EFFORT_VARIANT_RD_THRESHOLD} + "
              f">={self.EFFORT_VARIANT_MIN_LOSSES} losses + lower-effort sibling of same model does better")
        print(f"Provider inferior freeze: RD < {self.PROVIDER_INFERIOR_RD_THRESHOLD} + "
              f">={self.PROVIDER_INFERIOR_MIN_LOSSES} losses + better same-provider model "
              f"(within {self.PROVIDER_INFERIOR_TIME_WINDOW}mo) unless {self.PROVIDER_INFERIOR_COST_RATIO}x cheaper")
        print(f"Expensive inferior freeze: RD < {self.EXPENSIVE_INFERIOR_RD_THRESHOLD} + "
              f">={self.EXPENSIVE_INFERIOR_MIN_LOSSES} losses + stronger model (any provider, "
              f"within {self.EXPENSIVE_INFERIOR_TIME_WINDOW}mo) that is {self.EXPENSIVE_INFERIOR_COST_RATIO}x cheaper")
        print(f"Lost to weaker freeze: lost to model {self.LOST_TO_WEAKER_RATING_GAP}+ pts lower "
              f"(within {self.LOST_TO_WEAKER_TIME_WINDOW}mo) unless {self.LOST_TO_WEAKER_COST_RATIO}x cheaper")
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

        # Phase 1: Position benchmarks for models missing results
        benchmarked_this_run = await self._run_position_benchmarks(llm_ids, counters, max_cost)

        # Exclude models that just ran position benchmarks from game pairing.
        # Their benchmark counts as their contribution for this run.
        game_llm_ids = [lid for lid in llm_ids if lid not in benchmarked_this_run]

        # Phase 2: Game workers
        workers = [
            self._game_worker(
                worker_id=i,
                llm_ids=game_llm_ids,
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
