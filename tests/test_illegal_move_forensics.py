import unittest

import chess

from scripts.analyze_game_illegal_moves import analyze_game, classify_illegal_attempt


class IllegalMoveClassificationTests(unittest.TestCase):
    def classify(self, fen: str, move: str) -> dict:
        return classify_illegal_attempt(
            chess.Board(fen),
            move,
            previous_own_move=None,
            previous_turn_board=None,
        )

    def test_classifies_structural_illegal_move_reasons(self) -> None:
        starting = chess.STARTING_FEN

        self.assertEqual(self.classify(starting, "not-a-move")["primary_class"], "format_invalid")
        self.assertEqual(self.classify(starting, "e3e4")["primary_class"], "source_empty")
        self.assertEqual(self.classify(starting, "e7e5")["primary_class"], "wrong_side_piece")
        self.assertEqual(self.classify(starting, "e1e2")["primary_class"], "destination_own_piece")
        self.assertEqual(
            self.classify(starting, "c1h6")["primary_class"],
            "movement_rule_or_blocked",
        )

    def test_detects_move_that_ignores_last_opponent_reply(self) -> None:
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("d7d5")
        board.push_uci("e4d5")
        board.push_uci("d8d5")

        result = classify_illegal_attempt(
            board,
            "d5d6",
            previous_own_move="e4d5",
            previous_turn_board=None,
        )

        self.assertTrue(result["legal_if_ignoring_last_opponent_move"])

    def test_reconstructs_first_attempt_controls_and_matches_illegal_ply(self) -> None:
        result = {
            "white_id": "gpt-5.6-test",
            "black_id": "random-bot",
            "illegal_move_details": [
                {
                    "side": "white",
                    "move_number": 3,
                    "parsed_move": "g1g3",
                    "raw_response": "MOVE: g1g3",
                }
            ],
        }
        pgn = """[Result \"*\"]

1. e4 e5 2. Nf3 Nc6 *
"""

        turns, events = analyze_game(
            "game-1",
            result,
            pgn,
            ("gpt-5.6-",),
        )

        self.assertEqual([turn["ply_number"] for turn in turns], [1, 3])
        self.assertEqual(
            [turn["first_attempt_illegal"] for turn in turns],
            [False, True],
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["attempt_kind"], "first_attempt")
        self.assertEqual(events[0]["response_style"], "move_prefix")
        self.assertEqual(events[0]["detailed_class"], "knight_geometry")


if __name__ == "__main__":
    unittest.main()
