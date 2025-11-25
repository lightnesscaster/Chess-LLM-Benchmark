"""
Maia chess engine wrapper.

Maia is a neural network trained to play human-like chess at various rating levels.
Uses Lc0 (Leela Chess Zero) as the backend with Maia weights.
"""

import chess
import chess.engine
from typing import Optional
from .base_engine import BaseEngine


class MaiaEngine(BaseEngine):
    """
    Maia engine wrapper.

    Maia models are trained to mimic human play at specific rating bands:
    - maia-1100: ~1100 Elo play
    - maia-1200: ~1200 Elo play
    - ...
    - maia-1900: ~1900 Elo play

    Requires Lc0 binary and Maia weights file.
    """

    def __init__(
        self,
        player_id: str,
        rating: int,
        lc0_path: str = "lc0",
        weights_path: str = None,
        nodes: int = 1,  # Maia typically uses 1 node for human-like play
    ):
        """
        Initialize Maia engine.

        Args:
            player_id: Unique identifier (e.g., "maia-1100")
            rating: Fixed anchor rating matching the Maia model
            lc0_path: Path to lc0 binary
            weights_path: Path to Maia weights file (.pb.gz)
            nodes: Number of nodes to search (1 for pure Maia policy)
        """
        super().__init__(player_id, rating)
        self.lc0_path = lc0_path
        self.weights_path = weights_path
        self.nodes = nodes
        self._engine: Optional[chess.engine.SimpleEngine] = None

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        """Lazily initialize the engine."""
        if self._engine is None:
            # Lc0 requires weights to be specified
            if self.weights_path is None:
                raise ValueError("Maia requires weights_path to be specified")

            self._engine = chess.engine.SimpleEngine.popen_uci(
                [self.lc0_path, f"--weights={self.weights_path}"]
            )
        return self._engine

    def select_move(self, board: chess.Board) -> chess.Move:
        """Select a move using Maia."""
        engine = self._ensure_engine()

        # Use node limit for Maia (1 node = pure policy network)
        limit = chess.engine.Limit(nodes=self.nodes)
        result = engine.play(board, limit)
        return result.move

    def close(self) -> None:
        """Close the engine process."""
        if self._engine is not None:
            self._engine.quit()
            self._engine = None
