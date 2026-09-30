"""
Generic UCI engine wrapper for any UCI-compatible chess engine.
"""

import random
import statistics
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
        options: Optional[dict] = None,        # UCI options, e.g. {"Hash": 64, "Threads": 1}
        time_scale: float = 1.0,               # Multiplies the clock the engine sees
        time_controls: Optional[list] = None,  # [[initial_s, increment_s, weight], ...] drawn per game
        reference_nps: Optional[float] = None, # Reference hardware speed; sets time_scale at startup
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
            options: UCI options applied when the engine starts
            time_scale: Scales initial time and increment so a faster machine
                searches roughly as many nodes as the reference hardware
                (reference_nps / local_nps). The configured clock stays nominal.
            time_controls: Weighted nominal time controls; one is drawn per game
                (e.g. a lichess bot's classical mix). Overrides initial_time/increment.
            reference_nps: Nodes/second of the reference hardware. When set, the
                engine's local speed is probed once at startup and time_scale
                becomes reference_nps / measured_nps.
        """
        self._init_kwargs = dict(
            player_id=player_id, rating=rating, engine_path=engine_path, move_time=move_time,
            nodes=nodes, depth=depth, initial_time=initial_time, increment=increment,
            options=options, time_scale=time_scale, time_controls=time_controls,
            reference_nps=reference_nps,
        )
        super().__init__(player_id, rating)
        self.engine_path = engine_path
        self.move_time = move_time
        self.nodes = nodes
        self.depth = depth
        self.initial_time = initial_time
        self._nominal_increment = increment or 0
        self.time_scale = time_scale
        self.increment = self._nominal_increment * time_scale
        self.options = options or {}
        self.time_controls = time_controls
        self.reference_nps = reference_nps
        self.measured_nps: Optional[float] = None
        self.game_time_control: Optional[tuple] = None  # nominal (initial, increment) this game
        self._engine: Optional[chess.engine.SimpleEngine] = None

        # Clock state (in seconds)
        self._white_clock: Optional[float] = None
        self._black_clock: Optional[float] = None
        self._use_clock = initial_time is not None or bool(time_controls)

        # Probing speed needs the engine process, so defer it to the first move.
        if self._use_clock and reference_nps is None:
            self.reset_clock()

    def reset_clock(self) -> None:
        """Start a new game's clock (drawing its time control when configured)."""
        if not self._use_clock:
            return
        if self.reference_nps is not None and self.measured_nps is None:
            self._measure_nps()
        if self.time_controls:
            weights = [tc[2] if len(tc) > 2 else 1 for tc in self.time_controls]
            initial, increment = random.choices(self.time_controls, weights=weights)[0][:2]
        else:
            initial, increment = self.initial_time, self._nominal_increment
        self.game_time_control = (initial, increment)
        self.increment = increment * self.time_scale
        self._white_clock = initial * self.time_scale
        self._black_clock = initial * self.time_scale

    def _measure_nps(self) -> None:
        """Probe local search speed and derive time_scale from reference_nps."""
        engine = self._ensure_engine()
        probes = [
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "r2q1rk1/pp2bppp/2n1pn2/3p4/3P4/2NBPN2/PP3PPP/R2Q1RK1 w - - 0 10",
            "2r2rk1/1b2qppp/p3pn2/1p6/3N4/1BP1P3/PP3PPP/R2QR1K1 w - - 0 18",
            "8/5pk1/6p1/3R4/1r5P/6P1/5PK1/8 w - - 0 45",
        ]
        engine.analyse(chess.Board(probes[0]), chess.engine.Limit(time=1.0))  # JIT/cache warm-up
        samples = []
        for fen in probes:
            info = engine.analyse(chess.Board(fen), chess.engine.Limit(time=1.0))
            if info.get("nps"):
                samples.append(float(info["nps"]))
        if not samples:
            raise RuntimeError(f"{self.player_id}: engine reported no nps; cannot scale to reference speed")
        self.measured_nps = statistics.median(samples)
        self.time_scale = self.reference_nps / self.measured_nps
        print(f"  [{self.player_id}: local {self.measured_nps / 1e3:.0f}k nps vs reference "
              f"{self.reference_nps / 1e3:.0f}k -> time_scale {self.time_scale:.3f}]")

    def clone_for_game(self) -> "UCIEngine":
        """Fresh engine process and clock for one game; reuses the measured speed."""
        if self._use_clock and self.reference_nps is not None and self.measured_nps is None:
            self._measure_nps()
        clone = UCIEngine(**self._init_kwargs)
        if self.measured_nps is not None:
            clone.measured_nps = self.measured_nps
            clone.time_scale = self.time_scale
        return clone

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        """Lazily initialize the engine."""
        if self._engine is None:
            try:
                self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
                if self.options:
                    self._engine.configure(self.options)
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

        if self._use_clock and self._white_clock is None:
            self.reset_clock()
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
                print(f"  [Eubos clock: {self._white_clock:.1f}s remaining, thought {elapsed:.1f}s]")
            else:
                self._black_clock = max(0, self._black_clock - elapsed + self.increment)
                print(f"  [Eubos clock: {self._black_clock:.1f}s remaining, thought {elapsed:.1f}s]")

        return result.move

    def close(self) -> None:
        """Close the engine process."""
        if self._engine is not None:
            try:
                self._engine.quit()
            finally:
                self._engine = None
