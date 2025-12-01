"""
Generic UCI engine wrapper for any UCI-compatible chess engine.
"""

import time
import chess
import chess.engine
from typing import Optional
from .base_engine import BaseEngine


class UCIEngine(BaseEngine):
    """
    Generic UCI engine wrapper.

    Works with any UCI-compatible chess engine (Stockfish, Eubos, etc.)
    Supports both fixed limits (move_time, depth, nodes) and clock-based time control.
    """

    def __init__(
        self,
        player_id: str,
        rating: int,
        engine_path: str,
        move_time: Optional[float] = None,  # Seconds per move (fixed)
        nodes: Optional[int] = None,        # Max nodes to search
        depth: Optional[int] = None,        # Max depth to search
        initial_time: Optional[float] = None,  # Clock: initial time in seconds
        increment: Optional[float] = None,     # Clock: increment in seconds
    ):
        """
        Initialize UCI engine.

        Args:
            player_id: Unique identifier
            rating: Fixed anchor rating
            engine_path: Path to engine binary or launcher script
            move_time: Time limit per move in seconds (fixed per move)
            nodes: Node limit per move
            depth: Depth limit per move
            initial_time: Initial clock time in seconds (e.g., 900 for 15 min)
            increment: Time increment per move in seconds (e.g., 10)
        """
        super().__init__(player_id, rating)
        self.engine_path = engine_path
        self.move_time = move_time
        self.nodes = nodes
        self.depth = depth
        self.initial_time = initial_time
        self.increment = increment or 0
        self._engine: Optional[chess.engine.SimpleEngine] = None

        # Clock state (in seconds)
        self._white_clock: Optional[float] = None
        self._black_clock: Optional[float] = None
        self._use_clock = initial_time is not None

        if self._use_clock:
            self.reset_clock()

    def reset_clock(self) -> None:
        """Reset clocks to initial time. Call before each new game."""
        if self.initial_time is not None:
            self._white_clock = self.initial_time
            self._black_clock = self.initial_time

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

        if self._use_clock and self._white_clock is not None and self._black_clock is not None:
            # Clock-based time control
            limit_kwargs["white_clock"] = self._white_clock
            limit_kwargs["black_clock"] = self._black_clock
            limit_kwargs["white_inc"] = self.increment
            limit_kwargs["black_inc"] = self.increment
        else:
            # Fixed limits
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

        # Measure thinking time for clock update
        start_time = time.perf_counter()
        result = engine.play(board, limit)
        elapsed = time.perf_counter() - start_time

        if result is None or result.move is None:
            raise RuntimeError(f"UCI engine {self.player_id} failed to return a move")

        # Update clock if using time control
        if self._use_clock:
            if board.turn == chess.WHITE:
                self._white_clock = max(0, self._white_clock - elapsed + self.increment)
            else:
                self._black_clock = max(0, self._black_clock - elapsed + self.increment)

        return result.move

    def close(self) -> None:
        """Close the engine process."""
        if self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None
