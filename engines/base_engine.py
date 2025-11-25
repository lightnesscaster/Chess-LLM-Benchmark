"""
Base class for chess engine wrappers.
"""

import abc
import chess


class BaseEngine(abc.ABC):
    """Abstract base class for chess engines."""

    def __init__(self, player_id: str, rating: int):
        """
        Initialize the engine.

        Args:
            player_id: Unique identifier for this engine
            rating: Fixed anchor rating for this engine
        """
        self.player_id = player_id
        self.rating = rating

    @abc.abstractmethod
    def select_move(self, board: chess.Board) -> chess.Move:
        """
        Select a move given the current board position.

        Args:
            board: python-chess Board object

        Returns:
            A legal chess.Move
        """
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Clean up engine resources."""
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
