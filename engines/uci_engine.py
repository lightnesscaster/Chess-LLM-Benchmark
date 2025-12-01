"""
Generic UCI engine wrapper for any UCI-compatible chess engine.
"""

import chess
import chess.engine
from typing import Optional
from .base_engine import BaseEngine


class UCIEngine(BaseEngine):
    """
    Generic UCI engine wrapper.

    Works with any UCI-compatible chess engine (Stockfish, Eubos, etc.)
    """

    def __init__(
        self,
        player_id: str,
        rating: int,
        engine_path: str,
        move_time: Optional[float] = None,  # Seconds per move
        nodes: Optional[int] = None,        # Max nodes to search
        depth: Optional[int] = None,        # Max depth to search
    ):
        """
        Initialize UCI engine.

        Args:
            player_id: Unique identifier
            rating: Fixed anchor rating
            engine_path: Path to engine binary or launcher script
            move_time: Time limit per move in seconds
            nodes: Node limit per move
            depth: Depth limit per move
        """
        super().__init__(player_id, rating)
        self.engine_path = engine_path
        self.move_time = move_time
        self.nodes = nodes
        self.depth = depth
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        """Lazily initialize the engine."""
        if self._engine is None:
            try:
                self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"UCI engine not found at path: {self.engine_path}")
            except PermissionError:
                raise PermissionError(f"UCI engine not executable: {self.engine_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize UCI engine at {self.engine_path}: {e}")
        return self._engine

    def select_move(self, board: chess.Board) -> chess.Move:
        """Select a move using the UCI engine."""
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
        if result is None or result.move is None:
            raise RuntimeError(f"UCI engine {self.player_id} failed to return a move")
        return result.move

    def close(self) -> None:
        """Close the engine process."""
        if self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None
