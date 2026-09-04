from types import SimpleNamespace
import unittest

from game.freeze_checker import FreezeChecker
from game.match_scheduler import MatchScheduler
from game.stats_collector import StatsCollector


RANDOM_BOT = "random-bot"
TERRA_MEDIUM = "gpt-5.6-terra (medium)"
TERRA_HIGH = "gpt-5.6-terra (high)"
LUNA_MEDIUM = "gpt-5.6-luna (medium)"


def game_result(
    white_id: str,
    black_id: str,
    winner: str,
    *,
    illegal_moves_white: int = 0,
    illegal_moves_black: int = 0,
    total_moves_white: int = 20,
    total_moves_black: int = 20,
) -> SimpleNamespace:
    return SimpleNamespace(
        white_id=white_id,
        black_id=black_id,
        winner=winner,
        termination="checkmate",
        illegal_moves_white=illegal_moves_white,
        illegal_moves_black=illegal_moves_black,
        total_moves_white=total_moves_white,
        total_moves_black=total_moves_black,
        moves=total_moves_white + total_moves_black,
    )


class RatingStoreStub:
    def get(self, _player_id: str) -> SimpleNamespace:
        return SimpleNamespace(rating=600)


def scheduler_with_results(results: list[SimpleNamespace]) -> MatchScheduler:
    stats = StatsCollector()
    stats.add_results(results)
    scheduler = MatchScheduler.__new__(MatchScheduler)
    scheduler.stats_collector = stats
    scheduler.rating_store = RatingStoreStub()
    scheduler._freeze_checker = FreezeChecker(
        scheduler.rating_store,
        stats,
    )
    return scheduler


class RandomBotClearanceInheritanceTests(unittest.TestCase):
    def test_new_higher_effort_sibling_inherits_clearance(self) -> None:
        results = [
            game_result(TERRA_MEDIUM, RANDOM_BOT, "white")
            for _ in range(5)
        ]
        scheduler = scheduler_with_results(results)

        self.assertFalse(scheduler._needs_random_bot(TERRA_HIGH))

    def test_lower_effort_clearance_exempts_higher_effort_sibling(self) -> None:
        results = [
            game_result(TERRA_MEDIUM, RANDOM_BOT, "white")
            for _ in range(4)
        ]
        results.append(game_result(TERRA_MEDIUM, RANDOM_BOT, "draw"))
        results.append(
            game_result(
                TERRA_HIGH,
                "maia-1100",
                "white",
                illegal_moves_white=1,
            )
        )
        scheduler = scheduler_with_results(results)

        self.assertFalse(scheduler._needs_random_bot(TERRA_HIGH))

    def test_higher_effort_random_loss_blocks_inherited_clearance(self) -> None:
        results = [
            game_result(TERRA_MEDIUM, RANDOM_BOT, "white")
            for _ in range(5)
        ]
        results.append(
            game_result(
                TERRA_HIGH,
                RANDOM_BOT,
                "black",
                illegal_moves_white=1,
            )
        )
        scheduler = scheduler_with_results(results)

        self.assertTrue(scheduler._needs_random_bot(TERRA_HIGH))

    def test_clearance_does_not_transfer_between_model_families(self) -> None:
        results = [
            game_result(LUNA_MEDIUM, RANDOM_BOT, "white")
            for _ in range(5)
        ]
        results.append(
            game_result(
                TERRA_HIGH,
                "maia-1100",
                "white",
                illegal_moves_white=1,
            )
        )
        scheduler = scheduler_with_results(results)

        self.assertTrue(scheduler._needs_random_bot(TERRA_HIGH))

    def test_clearance_does_not_transfer_to_lower_effort_sibling(self) -> None:
        results = [
            game_result(TERRA_HIGH, RANDOM_BOT, "white")
            for _ in range(5)
        ]
        results.append(
            game_result(
                TERRA_MEDIUM,
                "maia-1100",
                "white",
                illegal_moves_white=1,
            )
        )
        scheduler = scheduler_with_results(results)

        self.assertTrue(scheduler._needs_random_bot(TERRA_MEDIUM))


if __name__ == "__main__":
    unittest.main()
