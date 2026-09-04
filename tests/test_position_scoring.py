import unittest

from position_benchmark.scoring import illegal_move_cpl


class PositionScoringTests(unittest.TestCase):
    def test_illegal_move_penalty_for_equal_position(self) -> None:
        self.assertEqual(illegal_move_cpl(0), 5000)

    def test_illegal_move_penalty_cannot_be_negative(self) -> None:
        self.assertEqual(illegal_move_cpl(-9840), 0)

    def test_illegal_move_penalty_preserves_positive_advantage(self) -> None:
        self.assertEqual(illegal_move_cpl(250), 5250)


if __name__ == "__main__":
    unittest.main()
