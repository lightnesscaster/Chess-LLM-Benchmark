"""
Survival engine designed to expose LLM calculation weaknesses.

Strategy:
1. Opening: Play drawish moves from opening book (threshold-based filtering)
2. Middlegame: Maintain evaluation stability relative to previous position
3. Degradation: Gradually widen acceptable eval loss window by move number
4. Blunder punishment: If opponent blunders (+3), take minimal winning advantage
"""

import random
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import chess.polyglot

from .base_engine import BaseEngine


class SurvivalEngine(BaseEngine):
    """
    A survival-focused chess engine that plays solid, drawish chess
    and degrades gradually to expose opponent calculation weaknesses.
    """

    # Phase-based evaluation windows (centipawns)
    # Format: (min_move, max_move, window_min_cp, window_max_cp)
    PHASE_WINDOWS = [
        (1, 20, -50, 50),      # Opening: maintain equality
        (21, 30, -50, 50),     # Early middle: maintain equality
        (31, 40, -100, 0),     # Middle: slight concession allowed
        (41, 50, -200, 0),     # Late middle: give ground
        (51, 999, -300, 0),    # Endgame: collapse
    ]

    def __init__(
        self,
        player_id: str,
        rating: int,
        stockfish_path: str = "stockfish",
        opening_book_path: Optional[str] = None,
        book_draw_threshold: float = 0.10,
        base_depth: int = 12,
        blunder_threshold: float = 3.0,
        seed: Optional[int] = None,
    ):
        """
        Initialize survival engine.

        Args:
            player_id: Unique identifier
            rating: Fixed anchor rating
            stockfish_path: Path to stockfish binary
            opening_book_path: Path to polyglot .bin opening book
            book_draw_threshold: Accept book moves within this % of best weight
            base_depth: Stockfish search depth for multi-PV analysis
            blunder_threshold: Pawns improvement to consider opponent blundered
            seed: Random seed for reproducibility
        """
        super().__init__(player_id, rating)
        self.stockfish_path = stockfish_path
        self.opening_book_path = opening_book_path
        self.book_draw_threshold = book_draw_threshold
        self.base_depth = base_depth
        self.blunder_threshold_cp = int(blunder_threshold * 100)  # Convert to centipawns

        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._book: Optional[chess.polyglot.MemoryMappedReader] = None
        self._rng = random.Random(seed)

        # Game state tracking - use board.ply() for consistent phase detection
        self._last_ply: int = -1  # Track last seen ply to detect new games
        self._last_eval_cp: Optional[int] = None  # Eval after our last move (baseline for next)

        # Load opening book if provided
        self._load_opening_book()

    def _load_opening_book(self) -> None:
        """Load polyglot opening book if path is provided and exists."""
        if self.opening_book_path:
            path = Path(self.opening_book_path)
            if path.exists():
                try:
                    self._book = chess.polyglot.open_reader(str(path))
                except Exception as e:
                    print(f"Warning: Could not load opening book: {e}")
                    self._book = None

    def _ensure_engine(self) -> chess.engine.SimpleEngine:
        """Lazily initialize the Stockfish engine."""
        if self._engine is None:
            try:
                self._engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"Stockfish not found at '{self.stockfish_path}'. "
                    "Please install Stockfish or specify the path via stockfish_path parameter."
                )
        return self._engine

    def _is_new_game(self, board: chess.Board) -> bool:
        """Detect if this is a new game (to reset state)."""
        current_ply = board.ply()

        # New game if ply regressed (went backwards) or at start
        if current_ply < self._last_ply or current_ply == 0:
            return True

        return False

    def _update_ply_tracking(self, board: chess.Board) -> None:
        """Update ply tracking after move selection."""
        self._last_ply = board.ply()

    def _get_game_ply(self, board: chess.Board) -> int:
        """Get current game ply (half-move count). Used for phase detection."""
        return board.ply()

    def _get_eval_cp(self, board: chess.Board) -> int:
        """Get position evaluation in centipawns from side-to-move perspective."""
        engine = self._ensure_engine()
        info = engine.analyse(board, chess.engine.Limit(depth=self.base_depth))
        score = info.get("score")
        if score is None:
            return 0

        pov_score = score.pov(board.turn)
        if pov_score.is_mate():
            # Treat mate as large value
            mate_in = pov_score.mate()
            return 10000 if mate_in > 0 else -10000

        cp_score = pov_score.score()
        return cp_score if cp_score is not None else 0

    def _get_phase_window(self, game_ply: int) -> tuple[int, int]:
        """Get the current phase's eval window (min_cp, max_cp) based on game ply."""
        for min_ply, max_ply, window_min, window_max in self.PHASE_WINDOWS:
            if min_ply <= game_ply <= max_ply:
                return (window_min, window_max)
        # Default to final phase
        return self.PHASE_WINDOWS[-1][2:4]

    def _select_opening_move(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Select a move from the opening book using threshold-based filtering.

        Returns None if no book move available.
        """
        if self._book is None:
            return None

        try:
            entries = list(self._book.find_all(board))
            if not entries:
                return None

            # Sort by weight (descending) - higher weight = more common/solid
            entries.sort(key=lambda e: e.weight, reverse=True)

            # Find best weight and calculate threshold
            best_weight = entries[0].weight
            if best_weight == 0:
                # All weights are 0, just pick randomly
                return self._rng.choice(entries).move

            # Calculate minimum acceptable weight
            # Interpret weights as proxy for "solidness"
            # threshold of 0.10 means accept moves with >= 90% of best weight
            min_weight = best_weight * (1.0 - self.book_draw_threshold)

            # Filter moves above threshold
            acceptable = [e for e in entries if e.weight >= min_weight]
            if not acceptable:
                acceptable = entries[:3]  # Fallback to top 3

            # Random selection from acceptable moves
            selected = self._rng.choice(acceptable)
            return selected.move

        except Exception:
            return None

    def _analyze_moves(self, board: chess.Board, multipv: int = 10) -> list[dict]:
        """
        Analyze position with multi-PV.

        Returns list of dicts with 'move' and 'eval_cp' keys.
        """
        engine = self._ensure_engine()

        # Get fallback move in case analysis fails
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            raise ValueError("No legal moves available - game should have ended")
        fallback_move = legal_moves[0]

        try:
            analysis = engine.analyse(
                board,
                chess.engine.Limit(depth=self.base_depth),
                multipv=multipv
            )
        except Exception:
            # Fallback: just get best move
            try:
                result = engine.play(board, chess.engine.Limit(depth=self.base_depth))
                if result.move is not None:
                    return [{"move": result.move, "eval_cp": 0}]
            except Exception:
                pass
            return [{"move": fallback_move, "eval_cp": 0}]

        moves = []
        for info in analysis:
            if "pv" not in info or not info["pv"]:
                continue

            move = info["pv"][0]
            score = info.get("score")

            if score is None:
                eval_cp = 0
            else:
                pov_score = score.pov(board.turn)
                if pov_score.is_mate():
                    mate_in = pov_score.mate()
                    eval_cp = 10000 if mate_in > 0 else -10000
                else:
                    cp_score = pov_score.score()
                    eval_cp = cp_score if cp_score is not None else 0

            moves.append({"move": move, "eval_cp": eval_cp})

        return moves if moves else [{"move": fallback_move, "eval_cp": 0}]

    def _select_middlegame_move(self, board: chess.Board, game_ply: int) -> chess.Move:
        """
        Select a middlegame move using the survival algorithm.

        Algorithm:
        1. Get baseline eval (from after our LAST move, or current position if first move)
        2. Analyze top moves with multi-PV
        3. For each candidate, calculate delta from baseline
        4. If opponent blundered (current position >= +3 vs baseline), take minimal winning move
        5. Otherwise, filter by phase window and select randomly
        6. Store the resulting eval as baseline for next turn

        This means if opponent makes a small mistake, we "give it back" by staying
        within our phase window relative to our previous position.
        """
        # Get baseline evaluation: use eval after our last move if available,
        # otherwise use current position (first middlegame move of the game)
        if self._last_eval_cp is not None:
            baseline_cp = self._last_eval_cp
        else:
            baseline_cp = self._get_eval_cp(board)

        # Check current position vs baseline to detect opponent blunder
        current_eval = self._get_eval_cp(board)
        opponent_gift = current_eval - baseline_cp

        # Analyze candidate moves (start with 10, expand to 20 if needed)
        candidates = self._analyze_moves(board, multipv=10)

        # Calculate delta for each candidate (vs baseline, not vs current position)
        # Delta = resulting eval - baseline (our last position)
        for c in candidates:
            c["delta_cp"] = c["eval_cp"] - baseline_cp

        # Check for blunder punishment based on opponent's gift
        # If opponent gave us >= blunder_threshold, they blundered
        if opponent_gift >= self.blunder_threshold_cp:
            # Opponent blundered! Take the WORST move that still captures some advantage
            # Find moves that are still better than baseline (positive delta)
            winning_moves = [c for c in candidates if c["delta_cp"] > 0]
            if winning_moves:
                winning_moves.sort(key=lambda c: c["delta_cp"])  # Sort ascending
                selected = winning_moves[0]  # Return move with minimum positive delta
                self._last_eval_cp = selected["eval_cp"]
                return selected["move"]

        # No blunder (or no winning moves) - filter by phase window based on game ply
        window_min, window_max = self._get_phase_window(game_ply)

        # Filter moves within the acceptable window
        acceptable = [c for c in candidates if window_min <= c["delta_cp"] <= window_max]

        # If no moves in window, expand search to 20 PV
        if not acceptable:
            candidates = self._analyze_moves(board, multipv=20)
            for c in candidates:
                c["delta_cp"] = c["eval_cp"] - baseline_cp
            acceptable = [c for c in candidates if window_min <= c["delta_cp"] <= window_max]

        # If still no moves in window, pick the move closest to the window
        if not acceptable:
            # Find move that minimizes distance to window
            def distance_to_window(c):
                delta = c["delta_cp"]
                if delta < window_min:
                    return window_min - delta
                elif delta > window_max:
                    return delta - window_max
                return 0

            candidates.sort(key=distance_to_window)
            selected = candidates[0]
            self._last_eval_cp = selected["eval_cp"]
            return selected["move"]

        # Random selection from acceptable moves
        selected = self._rng.choice(acceptable)
        self._last_eval_cp = selected["eval_cp"]
        return selected["move"]

    def select_move(self, board: chess.Board) -> chess.Move:
        """
        Select a move using the survival strategy.

        Routes to opening book or middlegame algorithm based on game ply.
        """
        # Get current game ply (half-move count)
        game_ply = self._get_game_ply(board)

        # Detect new game and reset state if needed
        if self._is_new_game(board):
            self._last_ply = -1  # Reset ply tracking
            self._last_eval_cp = None  # Reset eval baseline

        # Update ply tracking
        self._update_ply_tracking(board)

        # Opening phase: try book moves first (first 20 half-moves = ~10 full moves)
        if game_ply <= 20:
            book_move = self._select_opening_move(board)
            if book_move is not None and book_move in board.legal_moves:
                # Store eval after book move for baseline tracking
                board.push(book_move)
                self._last_eval_cp = self._get_eval_cp(board)
                board.pop()
                return book_move

        # Middlegame/endgame: use survival algorithm
        return self._select_middlegame_move(board, game_ply)

    def close(self) -> None:
        """Clean up engine and book resources."""
        if self._engine is not None:
            try:
                self._engine.quit()
            except Exception:
                pass
            self._engine = None

        if self._book is not None:
            try:
                self._book.close()
            except Exception:
                pass
            self._book = None
