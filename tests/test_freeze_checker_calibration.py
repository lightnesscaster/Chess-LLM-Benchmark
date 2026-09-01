from types import SimpleNamespace
import time
import unittest

from game.freeze_checker import FreezeChecker
from game.stats_collector import StatsCollector


HIGH = "deepseek-v4-flash-0731 (high)"
MAX = "deepseek-v4-flash-0731 (max)"
NO_THINKING = "deepseek-v4-flash-0731 (no thinking)"
CHEAP_PEER = "calibrated-peer (high)"
EXTERNAL_OPPONENT = "external-opponent (high)"
RANDOM_BOT = "random-bot"


class RatingStoreStub:
    def __init__(self, ratings: dict[str, tuple[float, float]]) -> None:
        self.ratings = {
            player_id: SimpleNamespace(
                rating=rating,
                rating_deviation=rd,
            )
            for player_id, (rating, rd) in ratings.items()
        }

    def get(self, player_id: str) -> SimpleNamespace:
        return self.ratings[player_id]

    def has_player(self, player_id: str) -> bool:
        return player_id in self.ratings


def loss(loser: str, winner: str) -> SimpleNamespace:
    return SimpleNamespace(
        white_id=loser,
        black_id=winner,
        winner="black",
        termination="checkmate",
        illegal_moves_white=0,
        illegal_moves_black=0,
        total_moves_white=20,
        total_moves_black=20,
        moves=40,
    )


def build_checker(
    ratings: dict[str, tuple[float, float]],
    results: list[SimpleNamespace],
    costs: dict[str, float],
    model_ids: dict[str, str],
) -> tuple[FreezeChecker, dict[str, dict]]:
    stats = StatsCollector()
    for result in results:
        stats.add_result(result)

    checker = FreezeChecker.__new__(FreezeChecker)
    checker.rating_store = RatingStoreStub(ratings)
    checker.stats_collector = stats
    checker.reasoning_ids = {
        player_id for player_id in ratings if "no thinking" not in player_id
    }
    checker._publish_dates = {
        player_id: time.time() - 24 * 60 * 60 for player_id in ratings
    }
    checker._player_model_ids = model_ids
    checker._models_by_model_id = {}
    for player_id, model_id in model_ids.items():
        checker._models_by_model_id.setdefault(model_id, []).append(player_id)
    checker._player_providers = {
        player_id: model_id.split("/", 1)[0]
        for player_id, model_id in model_ids.items()
    }
    checker._models_by_provider = {}
    for player_id, provider in checker._player_providers.items():
        checker._models_by_provider.setdefault(provider, []).append(player_id)
    checker.get_player_cost = lambda player_id: costs.get(player_id, 0.0)
    return checker, stats.get_player_stats()


class FreezeCheckerCalibrationTests(unittest.TestCase):
    def test_sibling_only_losses_do_not_trigger_cross_model_freezes(self) -> None:
        checker, player_stats = build_checker(
            ratings={
                HIGH: (1137, 116),
                MAX: (1251, 116),
                CHEAP_PEER: (1957, 60),
            },
            results=[loss(HIGH, MAX) for _ in range(4)],
            costs={HIGH: 0.2118, MAX: 0.2264, CHEAP_PEER: 0.03},
            model_ids={
                HIGH: "deepseek/deepseek-v4-flash-0731",
                MAX: "deepseek/deepseek-v4-flash-0731",
                CHEAP_PEER: "other/calibrated-peer",
            },
        )

        self.assertFalse(checker.is_frozen(HIGH, 116, player_stats))

    def test_random_bot_loss_still_triggers_weaker_opponent_freeze(self) -> None:
        checker, player_stats = build_checker(
            ratings={
                NO_THINKING: (-189, 164),
                RANDOM_BOT: (400, 0),
                CHEAP_PEER: (1505, 60),
            },
            results=[loss(NO_THINKING, RANDOM_BOT)],
            costs={NO_THINKING: 0.0002, CHEAP_PEER: 0.0003},
            model_ids={
                NO_THINKING: "deepseek/deepseek-v4-flash-0731",
                CHEAP_PEER: "other/calibrated-peer",
            },
        )

        self.assertTrue(checker.is_frozen(NO_THINKING, 164, player_stats))

    def test_three_external_losses_can_trigger_expensive_inferior_freeze(self) -> None:
        checker, player_stats = build_checker(
            ratings={
                HIGH: (1137, 116),
                EXTERNAL_OPPONENT: (1600, 60),
                CHEAP_PEER: (1957, 60),
            },
            results=[loss(HIGH, EXTERNAL_OPPONENT) for _ in range(3)],
            costs={HIGH: 0.2118, EXTERNAL_OPPONENT: 0.2, CHEAP_PEER: 0.03},
            model_ids={
                HIGH: "deepseek/deepseek-v4-flash-0731",
                EXTERNAL_OPPONENT: "other/external-opponent",
                CHEAP_PEER: "other/calibrated-peer",
            },
        )

        self.assertTrue(checker.is_frozen(HIGH, 116, player_stats))


if __name__ == "__main__":
    unittest.main()
