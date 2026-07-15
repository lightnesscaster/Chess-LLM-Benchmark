import unittest

import chess

from engines.random_engine import RandomEngine
from game.game_runner import GameRunner
from llm.base_llm import BaseLLMPlayer
from scripts.audit_game_retry_metrics import audit_games


class SequencePlayer(BaseLLMPlayer):
    def __init__(self, responses: list[str]) -> None:
        super().__init__(player_id="sequence-player", model_name="test-model")
        self.responses = list(responses)
        self.calls: list[tuple[bool, str | None]] = []

    async def select_move(
        self,
        _board: chess.Board,
        is_retry: bool = False,
        last_move_illegal: str | None = None,
    ) -> str:
        self.calls.append((is_retry, last_move_illegal))
        return self.responses.pop(0)

    async def close(self) -> None:
        return None


class IllegalMovePolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_failed_same_turn_retry_is_recorded(self) -> None:
        white = SequencePlayer(["a1a1", "a1a1"])
        black = RandomEngine(player_id="black-random", rating=400, seed=1)

        result, _ = await GameRunner(white, black, max_moves=4).play_game()

        self.assertEqual(result.termination, "forfeit_illegal_move")
        self.assertEqual(result.illegal_moves_white, 2)
        self.assertEqual(result.retry_attempts_white, 1)
        self.assertEqual(result.retry_recoveries_white, 0)
        self.assertEqual(result.retry_failures_white, 1)
        self.assertEqual(result.retry_unknown_white, 0)
        self.assertEqual([call[0] for call in white.calls], [False, True])

    async def test_later_second_strike_forfeits_without_another_retry(self) -> None:
        white = SequencePlayer(["a1a1", "e2e4", "a1a1"])
        black = RandomEngine(player_id="black-random", rating=400, seed=1)

        result, _ = await GameRunner(white, black, max_moves=4).play_game()

        self.assertEqual(result.termination, "forfeit_illegal_move")
        self.assertEqual(result.illegal_moves_white, 2)
        self.assertEqual(result.retry_attempts_white, 1)
        self.assertEqual(result.retry_recoveries_white, 1)
        self.assertEqual(result.retry_failures_white, 0)
        self.assertEqual(result.retry_unknown_white, 0)
        self.assertEqual([call[0] for call in white.calls], [False, True, False])


class RetryAuditTests(unittest.TestCase):
    def test_distinguishes_failed_retry_from_later_second_strike(self) -> None:
        games = {
            "failed": {
                "white_id": "gpt-test",
                "black_id": "random-bot",
                "moves": 5,
                "total_moves_white": 3,
                "termination": "forfeit_illegal_move",
                "winner": "black",
                "illegal_moves_white": 2,
                "illegal_moves_black": 0,
                "illegal_move_details": [
                    {"side": "white", "move_number": 6},
                    {"side": "white", "move_number": 6},
                ],
            },
            "later": {
                "white_id": "gpt-test",
                "black_id": "random-bot",
                "moves": 9,
                "total_moves_white": 5,
                "termination": "forfeit_illegal_move",
                "winner": "black",
                "illegal_moves_white": 2,
                "illegal_moves_black": 0,
                "illegal_move_details": [
                    {"side": "white", "move_number": 4},
                    {"side": "white", "move_number": 10},
                ],
            },
            "recovered": {
                "white_id": "gpt-test",
                "black_id": "random-bot",
                "moves": 8,
                "total_moves_white": 4,
                "termination": "max_moves",
                "winner": "draw",
                "illegal_moves_white": 1,
                "illegal_moves_black": 0,
                "illegal_move_details": [
                    {"side": "white", "move_number": 4},
                ],
            },
        }

        audit = audit_games(games, player_prefixes=("gpt-",))

        self.assertEqual(audit["game_evidence"][0]["winner"], "black")
        self.assertEqual(
            audit["game_evidence"][0]["termination"],
            "forfeit_illegal_move",
        )
        metrics = audit["players"]["gpt-test"]

        self.assertEqual(metrics["retry_attempts"], 3)
        self.assertEqual(metrics["retry_recoveries"], 2)
        self.assertEqual(metrics["retry_failures"], 1)
        self.assertEqual(metrics["later_second_strikes"], 1)
        self.assertAlmostEqual(metrics["retry_recovery_pct"], 200 / 3)
        self.assertEqual(metrics["first_attempt_illegals"], 4)
        self.assertEqual(metrics["first_attempt_turns"], 14)


if __name__ == "__main__":
    unittest.main()
