"""
Match scheduler for running benchmark games.

Handles:
- Pairing generation (LLM vs anchors, LLM vs LLM)
- Parallel game execution
- Rating updates after games
"""

import asyncio
import random
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

    # Legal move rate threshold - models below this must play random-bot (if rated above -200)
    LEGAL_MOVE_RATE_THRESHOLD = 0.98  # 98% accuracy
    LOW_ACCURACY_RATING_THRESHOLD = -200  # Only enforce accuracy requirement above this rating
    RANDOM_BOT_ID = "random-bot"

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

            # Update non-anchor players (full rating update)
            if not self.rating_store.is_anchor(white_id):
                new_white = self.glicko.update_rating(
                    white_rating,
                    opponents=[black_rating],
                    scores=[white_score],
                )
                self.rating_store.set(new_white)
            else:
                # Anchors: track game stats without changing rating
                self._update_anchor_stats(white_rating, white_score)

            if not self.rating_store.is_anchor(black_id):
                new_black = self.glicko.update_rating(
                    black_rating,
                    opponents=[white_rating],
                    scores=[black_score],
                )
                self.rating_store.set(new_black)
            else:
                # Anchors: track game stats without changing rating
                self._update_anchor_stats(black_rating, black_score)

    def _update_anchor_stats(self, rating: PlayerRating, score: float) -> None:
        """Update game statistics for an anchor without changing its rating."""
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
        - LLM has legal_move_rate below threshold AND rating > -200

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
            return rating > self.LOW_ACCURACY_RATING_THRESHOLD
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

        llm_rating = self.rating_store.get(llm_id).rating
        valid = []

        # Check anchors
        for anchor_id in anchor_ids:
            anchor_rating = self.rating_store.get(anchor_id).rating
            if abs(llm_rating - anchor_rating) <= rating_threshold:
                valid.append(anchor_id)

        # Check other LLMs
        for other_id in llm_ids:
            if other_id == llm_id:
                continue
            other_rating = self.rating_store.get(other_id).rating
            if abs(llm_rating - other_rating) <= rating_threshold:
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
        Pick the next game to play based on current ratings.

        Prioritizes:
        1. All anchor games first (globally) for rating calibration
        2. Then LLM-vs-LLM games once all anchor games are complete

        Within each phase, prioritizes LLMs with highest rating deviation.

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

        # Sort LLMs by rating deviation (highest first)
        llms_by_rd = sorted(
            llm_ids,
            key=lambda lid: self.rating_store.get(lid).rating_deviation,
            reverse=True,
        )

        # Check if any random-bot games remain globally
        random_bot_games_remaining = False
        other_anchor_games_remaining = False
        for llm_id in llm_ids:
            # Get valid opponents for this LLM (respects rating threshold)
            valid_opponents = self._get_valid_opponents(
                llm_id, anchor_ids, llm_ids, rating_threshold, player_stats
            )
            # Filter to only anchors
            valid_anchors = [opp_id for opp_id in valid_opponents if opp_id in anchor_set]
            for anchor_id in valid_anchors:
                for white_id, black_id in [(llm_id, anchor_id), (anchor_id, llm_id)]:
                    played = games_per_pairing.get((white_id, black_id), 0)
                    if played < games_vs_anchor_per_color:
                        if anchor_id == self.RANDOM_BOT_ID:
                            random_bot_games_remaining = True
                        else:
                            other_anchor_games_remaining = True

        # Build phase list: random-bot first, then other anchors, then LLM-vs-LLM
        # Always include llm phase to avoid getting stuck when anchor games exist but can't be scheduled
        phases = []
        if random_bot_games_remaining:
            phases.append("random-bot")
        if other_anchor_games_remaining:
            phases.append("anchor")
        phases.append("llm")

        for phase in phases:
            for llm_id in llms_by_rd:
                current_games = self._games_played.get(llm_id, 0)
                current_rd = self.rating_store.get(llm_id).rating_deviation

                # Check if this LLM is frozen (RD too low to initiate games)
                if current_rd < self.FROZEN_RD_THRESHOLD:
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
                for opp_id in valid_opponents:
                    is_anchor = opp_id in anchor_set
                    is_random_bot = opp_id == self.RANDOM_BOT_ID

                    # Filter by phase:
                    # - random-bot phase: only random-bot
                    # - anchor phase: only non-random-bot anchors
                    # - llm phase: only non-anchors
                    if phase == "random-bot" and not is_random_bot:
                        continue
                    if phase == "anchor" and (not is_anchor or is_random_bot):
                        continue
                    if phase == "llm" and is_anchor:
                        continue

                    # Check if LLM opponent has hit their caps
                    if not is_anchor:
                        opp_current = self._games_played.get(opp_id, 0)
                        opp_rd = self.rating_store.get(opp_id).rating_deviation

                        # Frozen models (RD < 60) can always be challenged - no cap
                        if opp_rd >= self.FROZEN_RD_THRESHOLD:
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
                    for white_id, black_id in [(llm_id, opp_id), (opp_id, llm_id)]:
                        played = games_per_pairing.get((white_id, black_id), 0)
                        if played < target:
                            # Weight by how many games remaining (more remaining = higher priority)
                            remaining = target - played
                            candidates.append((white_id, black_id, remaining))

                if candidates:
                    # Pick randomly among candidates, weighted by games remaining
                    weights = [c[2] for c in candidates]
                    chosen = random.choices(candidates, weights=weights, k=1)[0]
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
        counters: Dict[str, int],
    ) -> None:
        """
        Worker that continuously picks and plays games until none remain.
        """
        llm_set = set(llm_ids)  # Constant for tracking which players are LLMs

        while True:
            # Pick next game under lock (covers pairing selection, game counts, and counters)
            async with scheduler_lock:
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

                # Reserve everything atomically: pairing slot, game counts, counter
                games_per_pairing[(white_id, black_id)] = games_per_pairing.get((white_id, black_id), 0) + 1
                for player_id in [white_id, black_id]:
                    if player_id in llm_set:  # Track games for all LLMs
                        self._games_played[player_id] = self._games_played.get(player_id, 0) + 1
                counters["game_num"] += 1
                game_num = counters["game_num"]

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
                    # API error - release the slots
                    async with scheduler_lock:
                        games_per_pairing[(white_id, black_id)] -= 1
                        for player_id in [white_id, black_id]:
                            if player_id in llm_set:  # Track games for all LLMs
                                self._games_played[player_id] = max(0, self._games_played.get(player_id, 0) - 1)
                        counters["api_errors"] += 1
                else:
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

    async def run_benchmark(
        self,
        llm_ids: List[str],
        anchor_ids: List[str],
        games_vs_anchor_per_color: int = 10,
        games_vs_llm_per_color: int = 5,
        rating_threshold: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the full benchmark with dynamic matchmaking.

        Games are scheduled dynamically based on current ratings.
        Runs up to max_concurrent games in parallel.
        Prioritizes LLMs with highest rating deviation.

        Args:
            llm_ids: List of LLM player IDs to benchmark
            anchor_ids: List of anchor engine IDs
            games_vs_anchor_per_color: Games per LLM vs each anchor (per color)
            games_vs_llm_per_color: Games per LLM pair (per color)
            rating_threshold: Only pair players within this rating difference

        Returns:
            Benchmark results summary
        """
        # Reset trackers
        self._games_played = {}
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
        print(f"Frozen threshold: RD < {self.FROZEN_RD_THRESHOLD} (no games unless challenged)")
        print()

        # Shared state for workers
        results: List[GameResult] = []
        counters = {"game_num": 0, "errors": 0, "api_errors": 0}

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
            )
            for i in range(self.max_concurrent)
        ]

        await asyncio.gather(*workers)

        print(f"\nBenchmark complete: {len(results)} games, {counters['errors']} errors, {counters['api_errors']} API errors")

        # Show final ratings for all LLMs
        print("\nFinal ratings:")
        for llm_id in sorted(llm_ids, key=lambda x: self.rating_store.get(x).rating, reverse=True):
            r = self.rating_store.get(llm_id)
            games = self._games_played.get(llm_id, 0)
            print(f"  {llm_id}: {r.rating:.0f} ±{r.rating_deviation:.0f} ({games} games)")

        return {
            "total_games": counters["game_num"],
            "completed_games": len(results),
            "errors": counters["errors"],
            "api_errors": counters["api_errors"],
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
