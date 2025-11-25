"""
Stockfish engine wrapper with configurable strength.
"""

import chess
import chess.engine
from typing import Optional
from .base_engine import BaseEngine


class StockfishEngine(BaseEngine):
    """
    Stockfish engine wrapper.

    Can be configured for various strength levels using:
    - Skill Level (0-20)
    - Move time limits
    - Node limits
    """

    def __init__(
        self,
        player_id: str,
        rating: int,
        engine_path: str = "stockfish",
        skill_level: int = 20,
        move_time: Optional[float] = None,  # Seconds per move
        nodes: Optional[int] = None,        # Max nodes to search
        depth: Optional[int] = None,        # Max depth to search
    ):
        """
        Initialize Stockfish engine.

        Args:
            player_id: Unique identifier
            rating: Fixed anchor rating
            engine_path: Path to stockfish binary
            skill_level: Stockfish skill level 0-20 (20 = strongest)
            move_time: Time limit per move in seconds
            nodes: Node limit per move
            depth: Depth limit per move
        """
        super().__init__(player_id, rating)
        self.engine_path = engine_path
        self.skill_level = skill_level
        self.move_time = move_time
        self.nodes = nodes
        self.depth = depth
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        """Lazily initialize the engine."""
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            # Set skill level
            self._engine.configure({"Skill Level": self.skill_level})
        return self._engine

    def select_move(self, board: chess.Board) -> chess.Move:
        """Select a move using Stockfish."""
        engine = self._ensure_engine()

        # Build limit based on configuration
        limit_kwargs = {}
        if self.move_time is not None:
            limit_kwargs["time"] = self.move_time
        if self.nodes is not None:
            limit_kwargs["nodes"] = self.nodes
        if self.depth is not None:
            limit_kwargs["depth"] = self.depth

        # Default to 0.1 seconds if no limit specified
        if not limit_kwargs:
            limit_kwargs["time"] = 0.1

        limit = chess.engine.Limit(**limit_kwargs)
        result = engine.play(board, limit)
        return result.move

    def close(self) -> None:
        """Close the engine process."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None
