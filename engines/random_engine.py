"""
Random move engine - plays a random legal move each turn.
"""

import random
import chess
from .base_engine import BaseEngine


class RandomEngine(BaseEngine):
    """
    Engine that plays random legal moves.

    Useful as a floor anchor for rating calibration.
    """

    def __init__(self, player_id: str, rating: int, seed: int = None):
        """
        Initialize random engine.

        Args:
            player_id: Unique identifier
            rating: Fixed anchor rating
            seed: Optional random seed for reproducibility
        """
        super().__init__(player_id, rating)
        self._rng = random.Random(seed)

    def select_move(self, board: chess.Board) -> chess.Move:
        """Select a random legal move."""
        legal_moves = list(board.legal_moves)
        return self._rng.choice(legal_moves)

    def close(self) -> None:
        """Nothing to clean up."""
        pass
