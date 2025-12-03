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
    REASONING_HIGH_RATING_CAP = 25  # Cap for reasoning models rated > 1000
    REASONING_HIGH_RATING_THRESHOLD = 1000  # Rating threshold for higher cap

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

        # Track games played per player during benchmark (for reasoning caps)
        self._games_played: Dict[str, int] = {}
        self._games_played_lock = asyncio.Lock()

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

    async def _check_and_reserve_game(self, white_id: str, black_id: str) -> bool:
        """
        Check if a game can be played (reasoning models haven't hit caps).
        If allowed, reserve the game by incrementing counters.

        Returns:
            True if game is allowed and reserved, False if should be skipped
        """
        async with self._games_played_lock:
            # Check caps for both players
            for player_id in [white_id, black_id]:
                cap = self._get_game_cap(player_id)
                if cap is not None:
                    current = self._games_played.get(player_id, 0)
                    if current >= cap:
                        return False  # Cap reached, skip game

            # Reserve the game by incrementing counters
            for player_id in [white_id, black_id]:
                if player_id in self.reasoning_ids:
                    self._games_played[player_id] = self._games_played.get(player_id, 0) + 1

            return True

    async def _release_game_reservation(self, white_id: str, black_id: str) -> None:
        """Release a game reservation if the game didn't complete (e.g., API error)."""
        async with self._games_played_lock:
            for player_id in [white_id, black_id]:
                if player_id in self.reasoning_ids:
                    self._games_played[player_id] = max(0, self._games_played.get(player_id, 0) - 1)

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

    async def run_single_game(self, task: GameTask, skip_cap_check: bool = False) -> Optional[GameResult]:
        """
        Run a single game with concurrency control.

        Args:
            task: The game task
            skip_cap_check: If True, skip reasoning cap check (caller already handled it)

        Returns:
            GameResult if game completes normally, None if API error or skipped due to cap
        """
        white_id = task.white.player_id
        black_id = task.black.player_id

        # Check and reserve game slot (for reasoning model caps)
        # Skip if caller already reserved under scheduler_lock (dynamic matchmaking)
        if not skip_cap_check:
            if not await self._check_and_reserve_game(white_id, black_id):
                if self.verbose:
                    print(f"[{task.game_num}/{task.total_games}] {white_id} vs {black_id} - SKIPPED (cap reached)")
                return None

        async with self._semaphore:
            if self.verbose:
                print(f"[{task.game_num}/{task.total_games}] {white_id} vs {black_id}")

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
                # Release the reservation since game didn't count
                # (only if we did the reservation here, not if caller handled it)
                if not skip_cap_check:
                    await self._release_game_reservation(white_id, black_id)
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

    def _get_valid_opponents(
        self,
        llm_id: str,
        anchor_ids: List[str],
        llm_ids: List[str],
        rating_threshold: Optional[int],
    ) -> List[str]:
        """
        Get valid opponents for an LLM based on current ratings.

        Args:
            llm_id: The LLM to find opponents for
            anchor_ids: List of anchor IDs
            llm_ids: List of all LLM IDs
            rating_threshold: Max rating difference (None = no limit)

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

        Prioritizes LLMs with highest rating deviation (most uncertain).

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

        # Sort LLMs by rating deviation (highest first)
        llms_by_rd = sorted(
            llm_ids,
            key=lambda lid: self.rating_store.get(lid).rating_deviation,
            reverse=True,
        )

        for llm_id in llms_by_rd:
            # Check if this LLM has hit its reasoning cap
            cap = self._get_game_cap(llm_id)
            if cap is not None:
                current_games = self._games_played.get(llm_id, 0)
                if current_games >= cap:
                    continue

            # Get valid opponents based on current ratings
            valid_opponents = self._get_valid_opponents(
                llm_id, anchor_ids, llm_ids, rating_threshold
            )

            if not valid_opponents:
                continue

            # Find opponents with games remaining
            candidates = []
            for opp_id in valid_opponents:
                # Check if opponent (if LLM) has hit their reasoning cap
                if opp_id in self.reasoning_ids:
                    opp_cap = self._get_game_cap(opp_id)
                    if opp_cap is not None:
                        opp_current = self._games_played.get(opp_id, 0)
                        if opp_current >= opp_cap:
                            continue  # Skip capped opponent

                is_anchor = opp_id in anchor_set
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

                # Reserve everything atomically: pairing slot, reasoning caps, counter
                games_per_pairing[(white_id, black_id)] = games_per_pairing.get((white_id, black_id), 0) + 1
                for player_id in [white_id, black_id]:
                    if player_id in self.reasoning_ids:
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
                        if player_id in self.reasoning_ids:
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
                result = await self.run_single_game(task, skip_cap_check=True)

                if result is None:
                    # API error - release the slots
                    async with scheduler_lock:
                        games_per_pairing[(white_id, black_id)] -= 1
                        for player_id in [white_id, black_id]:
                            if player_id in self.reasoning_ids:
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
                        if player_id in self.reasoning_ids:
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
