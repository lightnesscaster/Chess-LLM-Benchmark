import asyncio
from types import SimpleNamespace
import unittest

import chess

from engines.random_engine import RandomEngine
from game.freeze_checker import FreezeChecker
from game.match_scheduler import (
    GameCancelledBeforeStart,
    GameTask,
    MatchScheduler,
)
from game.stats_collector import StatsCollector
from llm.base_llm import BaseLLMPlayer


TERRA_MEDIUM = "gpt-5.6-terra (medium)"
SOL_MEDIUM = "gpt-5.6-sol (medium)"


class RatingStoreStub:
    def __init__(self, rating: float) -> None:
        self.rating = rating

    def get(self, _player_id: str) -> SimpleNamespace:
        return SimpleNamespace(rating=self.rating)


class SchedulerRatingStoreStub:
    def __init__(self, ratings: dict[str, SimpleNamespace]) -> None:
        self.ratings = ratings

    def get(self, player_id: str) -> SimpleNamespace:
        return self.ratings[player_id]

    def has_player(self, player_id: str) -> bool:
        return player_id in self.ratings


class SchedulerFreezeCheckerStub:
    def __init__(self, frozen_ids: set[str] | None = None) -> None:
        self.frozen_ids = frozen_ids or set()

    def is_frozen(self, player_id: str, *_args) -> bool:
        return player_id in self.frozen_ids

    def get_player_cost(self, _player_id: str) -> float:
        return 0.0

    def should_limit_inflight_near_freeze(
        self,
        _player_id: str,
        current_rd: float,
        _player_stats,
    ) -> bool:
        return current_rd < 120


class CountingPlayer(BaseLLMPlayer):
    calls = 0

    def __init__(self) -> None:
        super().__init__(player_id=TERRA_MEDIUM, model_name="test-model")

    async def select_move(
        self,
        _board: chess.Board,
        is_retry: bool = False,
        last_move_illegal: str | None = None,
    ) -> str:
        type(self).calls += 1
        return "e2e4"

    async def close(self) -> None:
        return None


class NearFreezeInflightPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.checker = FreezeChecker(
            RatingStoreStub(rating=600),
            StatsCollector(),
            reasoning_ids={TERRA_MEDIUM},
        )

    def test_recent_weak_model_is_limited_within_twenty_rd_points(self) -> None:
        self.assertTrue(
            self.checker.should_limit_inflight_near_freeze(
                TERRA_MEDIUM,
                current_rd=119,
            )
        )

    def test_recent_weak_model_is_not_limited_outside_margin(self) -> None:
        self.assertFalse(
            self.checker.should_limit_inflight_near_freeze(
                TERRA_MEDIUM,
                current_rd=121,
            )
        )

    def test_near_freeze_model_cannot_reserve_second_inflight_game(self) -> None:
        anchor_id = "maia-test"
        stats = StatsCollector()
        stats.add_result(
            SimpleNamespace(
                white_id=TERRA_MEDIUM,
                black_id=anchor_id,
                winner="white",
                termination="checkmate",
                illegal_moves_white=0,
                illegal_moves_black=0,
                total_moves_white=20,
                total_moves_black=20,
                moves=40,
            )
        )
        scheduler = MatchScheduler.__new__(MatchScheduler)
        scheduler.stats_collector = stats
        scheduler.rating_store = SchedulerRatingStoreStub(
            {
                TERRA_MEDIUM: SimpleNamespace(
                    rating=600,
                    rating_deviation=121,
                    games_rd=121,
                    games_played=8,
                ),
                anchor_id: SimpleNamespace(
                    rating=600,
                    rating_deviation=0,
                    games_rd=0,
                    games_played=100,
                ),
            }
        )
        scheduler._freeze_checker = SchedulerFreezeCheckerStub()
        scheduler._games_played = {TERRA_MEDIUM: 1}
        scheduler._estimated_rd = {TERRA_MEDIUM: 121}
        scheduler._inflight_opponents = {TERRA_MEDIUM: {1: anchor_id}}
        scheduler._estimate_rd_after_game = (
            lambda _player_id, _opponent_id, _current_rd: 119
        )
        scheduler.reasoning_ids = {TERRA_MEDIUM}
        scheduler._shadow_priority_ids = set()
        scheduler._shadow_minimum_games = 8
        scheduler._shadow_maximum_games_rd = 100
        scheduler._shadow_priority_multiplier = 1
        scheduler._get_freeze_test_opponent = lambda *_args: None

        pairing = scheduler._pick_next_game(
            llm_ids=[TERRA_MEDIUM],
            anchor_ids=[anchor_id],
            games_per_pairing={(TERRA_MEDIUM, anchor_id): 1},
            games_vs_anchor_per_color=5,
            games_vs_llm_per_color=2,
            rating_threshold=600,
        )

        self.assertIsNone(pairing)

    def test_frozen_opponent_remains_challengeable_while_inflight(self) -> None:
        stats = StatsCollector()
        stats.add_result(
            SimpleNamespace(
                white_id=SOL_MEDIUM,
                black_id=TERRA_MEDIUM,
                winner="white",
                termination="checkmate",
                illegal_moves_white=0,
                illegal_moves_black=0,
                total_moves_white=20,
                total_moves_black=20,
                moves=40,
            )
        )
        scheduler = MatchScheduler.__new__(MatchScheduler)
        scheduler.stats_collector = stats
        scheduler.rating_store = SchedulerRatingStoreStub(
            {
                SOL_MEDIUM: SimpleNamespace(
                    rating=1200,
                    rating_deviation=200,
                    games_rd=200,
                    games_played=8,
                ),
                TERRA_MEDIUM: SimpleNamespace(
                    rating=600,
                    rating_deviation=50,
                    games_rd=50,
                    games_played=20,
                ),
            }
        )
        scheduler._freeze_checker = SchedulerFreezeCheckerStub(
            frozen_ids={TERRA_MEDIUM}
        )
        scheduler._games_played = {SOL_MEDIUM: 0, TERRA_MEDIUM: 1}
        scheduler._estimated_rd = {SOL_MEDIUM: 200, TERRA_MEDIUM: 50}
        scheduler._inflight_opponents = {
            TERRA_MEDIUM: {1: SOL_MEDIUM},
        }
        scheduler._estimate_rd_after_game = (
            lambda _player_id, _opponent_id, current_rd: current_rd - 2
        )
        scheduler.reasoning_ids = {SOL_MEDIUM, TERRA_MEDIUM}
        scheduler._shadow_priority_ids = set()
        scheduler._shadow_minimum_games = 8
        scheduler._shadow_maximum_games_rd = 100
        scheduler._shadow_priority_multiplier = 1
        scheduler._get_freeze_test_opponent = lambda *_args: None

        pairing = scheduler._pick_next_game(
            llm_ids=[SOL_MEDIUM, TERRA_MEDIUM],
            anchor_ids=[],
            games_per_pairing={(SOL_MEDIUM, TERRA_MEDIUM): 1},
            games_vs_anchor_per_color=5,
            games_vs_llm_per_color=2,
            rating_threshold=600,
        )

        self.assertIsNotNone(pairing)
        self.assertEqual(set(pairing), {SOL_MEDIUM, TERRA_MEDIUM})


class PreStartCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_game_makes_no_model_call(self) -> None:
        CountingPlayer.calls = 0
        scheduler = MatchScheduler.__new__(MatchScheduler)
        scheduler._semaphore = asyncio.Semaphore(1)
        task = GameTask(
            white=CountingPlayer(),
            black=RandomEngine("random-bot", 400, seed=1),
            game_num=1,
            total_games=1,
        )

        async def no_longer_eligible() -> bool:
            return False

        with self.assertRaisesRegex(RuntimeError, "cancelled before start"):
            await scheduler.run_single_game(
                task,
                pre_start_check=no_longer_eligible,
            )

        self.assertEqual(CountingPlayer.calls, 0)

    async def test_worker_releases_cancelled_reservation_without_error(self) -> None:
        scheduler = MatchScheduler.__new__(MatchScheduler)
        scheduler.players = {
            TERRA_MEDIUM: CountingPlayer(),
            "random-bot": RandomEngine("random-bot", 400, seed=1),
        }
        scheduler.rating_store = SchedulerRatingStoreStub(
            {
                TERRA_MEDIUM: SimpleNamespace(
                    rating=600,
                    rating_deviation=100,
                ),
                "random-bot": SimpleNamespace(
                    rating=400,
                    rating_deviation=0,
                ),
            }
        )
        scheduler.stats_collector = StatsCollector()
        scheduler.verbose = False
        scheduler._games_played = {}
        scheduler._estimated_rd = {}
        scheduler._inflight_opponents = {}

        choices = [(TERRA_MEDIUM, "random-bot"), None]
        scheduler._pick_next_game = lambda **_kwargs: choices.pop(0)
        scheduler._estimate_game_cost_detail = (
            lambda *_args: (0.0, "test estimate")
        )
        scheduler._get_estimated_rd = lambda _player_id: 100
        scheduler._estimate_rd_after_game = (
            lambda _player_id, _opponent_id, current_rd: current_rd - 1
        )
        scheduler._recompute_estimated_rd = lambda _player_id: None

        freeze_checks = iter([False, True])
        scheduler._freeze_checker = SimpleNamespace(
            is_frozen=lambda *_args: next(freeze_checks),
        )

        async def cancel_before_start(
            _task: GameTask,
            pre_start_check=None,
        ):
            self.assertIsNotNone(pre_start_check)
            if not await pre_start_check():
                raise GameCancelledBeforeStart("game cancelled before start")
            self.fail("cancelled game reached execution")

        scheduler.run_single_game = cancel_before_start
        games_per_pairing = {}
        counters = {
            "game_num": 0,
            "errors": 0,
            "api_errors": 0,
            "total_cost": 0.0,
            "pending_cost": 0.0,
            "pending_estimates": {},
            "budget_exceeded": False,
        }

        await scheduler._game_worker(
            worker_id=0,
            llm_ids=[TERRA_MEDIUM],
            anchor_ids=["random-bot"],
            games_per_pairing=games_per_pairing,
            scheduler_lock=asyncio.Lock(),
            games_vs_anchor_per_color=5,
            games_vs_llm_per_color=2,
            rating_threshold=600,
            results=[],
            counters=counters,
            max_cost=1,
        )

        self.assertEqual(counters["errors"], 0)
        self.assertEqual(counters["api_errors"], 0)
        self.assertEqual(counters["pending_cost"], 0)
        self.assertEqual(
            games_per_pairing[(TERRA_MEDIUM, "random-bot")],
            0,
        )
        self.assertEqual(scheduler._games_played[TERRA_MEDIUM], 0)
        self.assertEqual(scheduler._inflight_opponents[TERRA_MEDIUM], {})


if __name__ == "__main__":
    unittest.main()
