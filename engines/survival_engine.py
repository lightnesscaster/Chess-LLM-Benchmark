"""
Survival engine designed to expose LLM calculation weaknesses.

Strategy:
1. Opening: Play drawish moves from opening book (threshold-based filtering)
2. Middlegame: Maintain evaluation stability relative to previous position
3. Degradation: Gradually widen acceptable eval loss window by move number
4. Blunder punishment: If opponent blunders (+3), take minimal winning advantage
"""

import logging
import random
import time
from pathlib import Path
from typing import Optional

import chess
import chess.engine
import chess.polyglot
import requests

from .base_engine import BaseEngine

logger = logging.getLogger(__name__)


class SurvivalEngine(BaseEngine):
    """
    A survival-focused chess engine that plays solid, drawish chess
    and degrades gradually to expose opponent calculation weaknesses.
    """

    # Phase-based evaluation windows (centipawns)
    # Format: (min_ply, max_ply, window_min_cp, window_max_cp)
    PHASE_WINDOWS = [
        (0, 20, -50, 50),      # Opening: maintain equality
        (21, 30, -50, 50),     # Early middle: maintain equality
        (31, 999, -150, 50),   # Middle onwards: slight concession allowed
    ]

    # Advantage cap: if winning by more than this, give back to target range
    ADVANTAGE_CAP_THRESHOLD = 200  # If eval >= +200cp, activate cap
    ADVANTAGE_CAP_MIN = 0          # Target range minimum
    ADVANTAGE_CAP_MAX = 200        # Target range maximum

    def __init__(
        self,
        player_id: str,
        rating: int,
        stockfish_path: str = "stockfish",
        opening_book_path: Optional[str] = None,
        book_draw_threshold: float = 0.10,
        base_depth: int = 15,
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
        Select a move from Lichess opening explorer based on draw percentage.

        Queries the Lichess API for W/D/L statistics and picks the move
        with the highest draw percentage to play solid, drawish chess.
        Falls back to polyglot book if API fails.

        Returns None if no book move available.
        """
        # Try Lichess API first
        lichess_move = self._select_opening_move_lichess(board)
        if lichess_move is not None:
            return lichess_move

        # Fall back to polyglot book
        return self._select_opening_move_polyglot(board)

    def _select_opening_move_lichess(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Query Lichess opening explorer API and select move with highest draw %.
        """
        try:
            fen = board.fen()
            # Use masters database for high-quality games
            url = "https://explorer.lichess.ovh/masters"
            params = {
                "fen": fen,
                "topGames": 0,
                "recentGames": 0,
            }

            response = requests.get(url, params=params, timeout=(3, 7))
            if response.status_code == 429:
                # Rate limited, wait and retry once
                logger.debug("  Rate limited by Lichess API, retrying after 2s")
                time.sleep(2)
                response = requests.get(url, params=params, timeout=(3, 7))
                if response.status_code != 200:
                    logger.debug(f"  Lichess API retry failed with status {response.status_code}")
                    return None

            if response.status_code != 200:
                logger.debug(f"  Lichess API returned status {response.status_code}")
                return None

            try:
                data = response.json()
            except requests.exceptions.JSONDecodeError:
                logger.debug("  Lichess API returned invalid JSON")
                return None

            moves = data.get("moves", [])
            if not moves:
                logger.debug("  No moves from Lichess API")
                return None

            # Calculate draw percentage for each move
            move_stats = []
            for m in moves:
                try:
                    white_wins = int(m.get("white", 0))
                    draws = int(m.get("draws", 0))
                    black_wins = int(m.get("black", 0))
                    total = white_wins + draws + black_wins
                    if total < 10:
                        continue  # Skip moves with too few games

                    draw_pct = draws / total if total > 0 else 0  # Range [0, 1]
                    uci_move = m.get("uci")
                    move = chess.Move.from_uci(uci_move)
                    if move in board.legal_moves:
                        move_stats.append({
                            "move": move,
                            "uci": uci_move,
                            "draw_pct": draw_pct,
                            "total_games": total,
                        })
                        logger.debug(f"    {uci_move}: draw%={draw_pct:.1%} ({total} games)")
                except (ValueError, TypeError, AttributeError):
                    continue

            if not move_stats:
                logger.debug("  No valid moves from Lichess API")
                return None

            # Sort by draw percentage (highest first)
            move_stats.sort(key=lambda x: x["draw_pct"], reverse=True)

            # Select from moves within relative threshold of best draw %
            # (consistent with polyglot threshold logic)
            best_draw_pct = move_stats[0]["draw_pct"]
            threshold = best_draw_pct * (1.0 - self.book_draw_threshold)
            acceptable = [m for m in move_stats if m["draw_pct"] >= threshold]

            if not acceptable:
                acceptable = move_stats[:3] if len(move_stats) >= 3 else move_stats

            if not acceptable:  # Safety check
                return None

            selected = self._rng.choice(acceptable)
            logger.debug(f"  LICHESS OPENING: selected {selected['uci']} (draw%={selected['draw_pct']:.1%})")
            return selected["move"]

        except requests.RequestException as e:
            logger.debug(f"  Lichess API request failed: {e}")
            return None
        except Exception as e:
            logger.debug(f"  Lichess API error: {e}")
            return None

    def _select_opening_move_polyglot(self, board: chess.Board) -> Optional[chess.Move]:
        """
        Select a move from the polyglot opening book (fallback).
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
            min_weight = best_weight * (1.0 - self.book_draw_threshold)

            # Filter moves above threshold
            acceptable = [e for e in entries if e.weight >= min_weight]
            if not acceptable:
                acceptable = entries[:3]  # Fallback to top 3

            # Random selection from acceptable moves
            selected = self._rng.choice(acceptable)
            logger.debug(f"  POLYGLOT OPENING: selected {selected.move.uci()}")
            return selected.move

        except Exception:
            return None

    def _count_opponent_good_responses(self, board: chess.Board, move: chess.Move, threshold_cp: int = 200) -> int:
        """
        Count how many good responses the opponent has after we play a move.

        A "good" response is within threshold_cp of the best move.
        Uses lower depth (8) for speed since this is called per candidate.

        Returns count of good responses (moves within threshold of best).
        """
        engine = self._ensure_engine()

        # Push our move to analyze opponent's position
        board.push(move)
        try:
            analysis = engine.analyse(
                board,
                chess.engine.Limit(depth=8),
                multipv=10
            )

            if not analysis:
                return 10  # Assume many good responses if analysis fails

            # Get best eval from opponent's perspective
            best_eval = None
            evals = []
            for info in analysis:
                score = info.get("score")
                if score is None:
                    continue
                pov_score = score.pov(board.turn)
                if pov_score.is_mate():
                    mate_in = pov_score.mate()
                    eval_cp = 10000 if mate_in > 0 else -10000
                else:
                    cp_score = pov_score.score()
                    eval_cp = cp_score if cp_score is not None else 0
                evals.append(eval_cp)
                if best_eval is None:
                    best_eval = eval_cp

            if best_eval is None:
                return 10  # Assume many good responses if no evals

            # Count moves within threshold of best
            good_count = sum(1 for e in evals if best_eval - e <= threshold_cp)
            return good_count

        except Exception:
            return 10  # Assume many good responses on error
        finally:
            board.pop()

    def _creates_mate_threat(self, board: chess.Board, move: chess.Move) -> bool:
        """
        Check if playing this move creates a forced mate threat.

        Uses Stockfish to detect any forced mate (not just mate-in-1).
        Returns True if the position after our move shows a forced mate for us.
        """
        board.push(move)
        try:
            engine = self._ensure_engine()
            # Quick depth 10 analysis to detect forced mates
            info = engine.analyse(board, chess.engine.Limit(depth=10))
            score = info.get("score")
            if score is None:
                return False

            # Get score from the perspective of the side that just moved (us)
            # After push, it's opponent's turn, so we want pov(not board.turn)
            pov_score = score.pov(not board.turn)
            if pov_score.is_mate():
                mate_in = pov_score.mate()
                # Positive mate_in means we have a forced mate
                if mate_in is not None and mate_in > 0:
                    logger.debug(f"    {move.uci()} creates mate in {mate_in}")
                    return True
            return False
        finally:
            board.pop()

    def _filter_by_mate_threats(self, board: chess.Board, candidates: list[dict]) -> list[dict]:
        """
        Filter out candidates that create forced mate threats.

        This makes survival-bot less aggressive by avoiding positions
        where opponent must find the only defense or get mated.
        """
        filtered = []
        for c in candidates:
            creates_threat = self._creates_mate_threat(board, c["move"])
            c["creates_mate_threat"] = creates_threat
            if creates_threat:
                logger.debug(f"    {c['move'].uci()}: creates mate threat - REJECTED")
            else:
                logger.debug(f"    {c['move'].uci()}: no mate threat")
                filtered.append(c)
        return filtered

    def _filter_by_response_diversity(self, board: chess.Board, candidates: list[dict], min_responses: int) -> list[dict]:
        """
        Filter candidates to only include moves that give opponent at least min_responses good options.

        This makes survival-bot more forgiving by avoiding forcing moves.
        """
        filtered = []
        for c in candidates:
            good_responses = self._count_opponent_good_responses(board, c["move"])
            c["opponent_good_responses"] = good_responses
            logger.debug(f"    {c['move'].uci()}: opponent has {good_responses} good responses")
            if good_responses >= min_responses:
                filtered.append(c)
        return filtered

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
        baseline_from_last = self._last_eval_cp is not None
        if baseline_from_last:
            baseline_cp = self._last_eval_cp
        else:
            baseline_cp = self._get_eval_cp(board)

        # Check current position vs baseline to detect opponent blunder
        current_eval = self._get_eval_cp(board)
        opponent_gift = current_eval - baseline_cp

        logger.debug(
            f"[Ply {game_ply}] FEN: {board.fen()}\n"
            f"  baseline_cp={baseline_cp} (from_last={baseline_from_last}), "
            f"current_eval={current_eval}, opponent_gift={opponent_gift}"
        )

        # Analyze candidate moves (start with 10, expand to 20 if needed)
        candidates = self._analyze_moves(board, multipv=10)

        # Calculate delta for each candidate (vs baseline, not vs current position)
        # Delta = resulting eval - baseline (our last position)
        for c in candidates:
            c["delta_cp"] = c["eval_cp"] - baseline_cp

        # Log all candidates
        logger.debug(f"  Candidates (top 10):")
        for c in candidates:
            logger.debug(f"    {c['move'].uci()}: eval={c['eval_cp']}, delta={c['delta_cp']}")

        # Check for blunder punishment based on opponent's gift
        # If opponent gave us >= blunder_threshold, they blundered
        # Only punish blunders 50% of the time to be more forgiving
        if opponent_gift >= self.blunder_threshold_cp:
            if self._rng.random() >= 0.5:
                logger.debug(f"  BLUNDER DETECTED but FORGIVEN (50% chance): opponent_gift={opponent_gift}")
                # Fall through to normal move selection
            else:
                logger.debug(f"  BLUNDER DETECTED: opponent_gift={opponent_gift} >= threshold={self.blunder_threshold_cp}")
                # Opponent blundered! Take the WORST move that still captures some advantage
                # Find moves that are still better than baseline (positive delta)
                winning_moves = [c for c in candidates if c["delta_cp"] > 0]
                if winning_moves:
                    # Filter out moves that create mate threats
                    logger.debug(f"  Checking for mate threats in {len(winning_moves)} winning moves...")
                    non_threatening_winning = self._filter_by_mate_threats(board, winning_moves)
                    if non_threatening_winning:
                        winning_moves = non_threatening_winning
                    else:
                        logger.debug(f"  All winning moves create mate threats, keeping original list")

                    winning_moves.sort(key=lambda c: c["delta_cp"])  # Sort ascending
                    selected = winning_moves[0]  # Return move with minimum positive delta
                    logger.debug(f"  SELECTED (blunder punishment): {selected['move'].uci()} eval={selected['eval_cp']} delta={selected['delta_cp']}")
                    self._last_eval_cp = selected["eval_cp"]
                    return selected["move"]
                logger.debug(f"  No winning moves despite blunder, falling through")
                # No winning moves despite blunder - fall through to normal move selection

        # Advantage cap: if winning by too much, give back to target range
        # This prevents crushing weaker opponents while maintaining a slight edge
        # Don't cap mate positions (eval >= 10000) - always take the mate
        if self.ADVANTAGE_CAP_THRESHOLD <= current_eval < 10000:
            logger.debug(f"  ADVANTAGE CAP: current_eval={current_eval} >= threshold={self.ADVANTAGE_CAP_THRESHOLD}")
            # Find moves that result in eval within target range (0 to +200cp)
            cap_moves = [c for c in candidates
                        if self.ADVANTAGE_CAP_MIN <= c["eval_cp"] <= self.ADVANTAGE_CAP_MAX]

            # Filter out moves that create mate threats
            if cap_moves:
                logger.debug(f"  Checking for mate threats in {len(cap_moves)} cap moves...")
                non_threatening_cap = self._filter_by_mate_threats(board, cap_moves)
                if non_threatening_cap:
                    cap_moves = non_threatening_cap
                else:
                    logger.debug(f"  All cap moves create mate threats, keeping original list")

            if cap_moves:
                selected = self._rng.choice(cap_moves)
                logger.debug(f"  SELECTED (advantage cap): {selected['move'].uci()} eval={selected['eval_cp']} delta={selected['delta_cp']}")
                self._last_eval_cp = selected["eval_cp"]
                return selected["move"]
            else:
                # No moves in target range - pick move closest to cap max while still above it
                # (minimize advantage while staying winning)
                above_cap = [c for c in candidates if c["eval_cp"] > self.ADVANTAGE_CAP_MAX]

                # Filter out moves that create mate threats
                if above_cap:
                    logger.debug(f"  Checking for mate threats in {len(above_cap)} above-cap moves...")
                    non_threatening_above = self._filter_by_mate_threats(board, above_cap)
                    if non_threatening_above:
                        above_cap = non_threatening_above
                    else:
                        logger.debug(f"  All above-cap moves create mate threats, keeping original list")

                if above_cap:
                    above_cap.sort(key=lambda c: c["eval_cp"])  # Sort ascending (closest to cap)
                    selected = above_cap[0]
                    logger.debug(f"  SELECTED (above cap, closest): {selected['move'].uci()} eval={selected['eval_cp']} delta={selected['delta_cp']}")
                    self._last_eval_cp = selected["eval_cp"]
                    return selected["move"]
                else:
                    # All moves result in eval below cap minimum (losing) - pick best available
                    candidates.sort(key=lambda c: c["eval_cp"], reverse=True)
                    selected = candidates[0]
                    logger.debug(f"  SELECTED (cap but losing, best): {selected['move'].uci()} eval={selected['eval_cp']} delta={selected['delta_cp']}")
                    self._last_eval_cp = selected["eval_cp"]
                    return selected["move"]

        # Survival mode: when losing badly, play optimally to survive longer
        # Select moves within 200cp of best, then filter
        LOSING_THRESHOLD = -300
        LOSING_MOVE_WINDOW = 200
        if current_eval < LOSING_THRESHOLD:
            logger.debug(f"  SURVIVAL MODE: current_eval={current_eval} < {LOSING_THRESHOLD}")
            # Get best eval from candidates
            best_eval = max(c["eval_cp"] for c in candidates)
            # Select moves within 200cp of best
            survival_candidates = [c for c in candidates if best_eval - c["eval_cp"] <= LOSING_MOVE_WINDOW]
            logger.debug(f"  Survival candidates (within {LOSING_MOVE_WINDOW}cp of best {best_eval}): {[c['move'].uci() for c in survival_candidates]}")

            # Skip extra filtering in extreme positions (near mate)
            if abs(current_eval) >= 5000:
                logger.debug(f"  Skipping extra filtering (extreme eval={current_eval})")
                selected = survival_candidates[0]  # Best move
                logger.debug(f"  SELECTED (survival, extreme): {selected['move'].uci()} eval={selected['eval_cp']}")
                self._last_eval_cp = selected["eval_cp"]
                return selected["move"]

            # Apply response diversity filter
            logger.debug(f"  Checking response diversity for {len(survival_candidates)} survival candidates...")
            diverse_moves = self._filter_by_response_diversity(board, survival_candidates, min_responses=3)
            if diverse_moves:
                logger.debug(f"  {len(diverse_moves)} moves give opponent 3+ good responses")
                survival_candidates = diverse_moves
            else:
                diverse_moves = [c for c in survival_candidates if c.get("opponent_good_responses", 0) >= 2]
                if diverse_moves:
                    logger.debug(f"  {len(diverse_moves)} moves give opponent 2+ good responses")
                    survival_candidates = diverse_moves
                else:
                    logger.debug(f"  No diverse moves, keeping original survival candidates")

            # Apply mate threat filter
            logger.debug(f"  Checking for mate threats in {len(survival_candidates)} survival candidates...")
            non_threatening = self._filter_by_mate_threats(board, survival_candidates)
            if non_threatening:
                logger.debug(f"  {len(non_threatening)} moves don't create mate threats")
                survival_candidates = non_threatening
            else:
                logger.debug(f"  All moves create mate threats, keeping original list")

            # Pick the best move from filtered candidates
            survival_candidates.sort(key=lambda c: c["eval_cp"], reverse=True)
            selected = survival_candidates[0]
            logger.debug(f"  SELECTED (survival mode): {selected['move'].uci()} eval={selected['eval_cp']}")
            self._last_eval_cp = selected["eval_cp"]
            return selected["move"]

        # No blunder (or no winning moves) - filter by phase window based on game ply
        window_min, window_max = self._get_phase_window(game_ply)
        logger.debug(f"  Phase window: [{window_min}, {window_max}]")

        # Filter moves within the acceptable window
        acceptable = [c for c in candidates if window_min <= c["delta_cp"] <= window_max]
        logger.debug(f"  Acceptable moves (in window): {[c['move'].uci() for c in acceptable]}")

        # If no moves in window, expand search to 20 PV
        if not acceptable:
            logger.debug(f"  No moves in window, expanding to 20 PV...")
            candidates = self._analyze_moves(board, multipv=20)
            for c in candidates:
                c["delta_cp"] = c["eval_cp"] - baseline_cp
            acceptable = [c for c in candidates if window_min <= c["delta_cp"] <= window_max]
            logger.debug(f"  Acceptable moves after expansion: {[c['move'].uci() for c in acceptable]}")

        # If still no moves in window, pick the move closest to the window
        if not acceptable:
            logger.debug(f"  Still no moves in window, picking closest...")
            # Find move that minimizes distance to window
            def distance_to_window(c):
                delta = c["delta_cp"]
                if delta < window_min:
                    return window_min - delta
                elif delta > window_max:
                    return delta - window_max
                return 0

            candidates.sort(key=distance_to_window)
            for c in candidates[:5]:
                logger.debug(f"    {c['move'].uci()}: delta={c['delta_cp']}, distance={distance_to_window(c)}")
            selected = candidates[0]
            logger.debug(f"  SELECTED (closest to window): {selected['move'].uci()} eval={selected['eval_cp']} delta={selected['delta_cp']}")
            self._last_eval_cp = selected["eval_cp"]
            return selected["move"]

        # Skip extra filtering in extreme positions (near mate)
        if abs(current_eval) >= 5000:
            logger.debug(f"  Skipping extra filtering (extreme eval={current_eval})")
            selected = self._rng.choice(acceptable)
            logger.debug(f"  SELECTED (from acceptable, no extra filtering): {selected['move'].uci()} eval={selected['eval_cp']} delta={selected['delta_cp']}")
            self._last_eval_cp = selected["eval_cp"]
            return selected["move"]

        # Filter by response diversity - prefer moves that give opponent many good options
        # This makes survival-bot more forgiving by avoiding forcing moves
        logger.debug(f"  Checking response diversity for {len(acceptable)} acceptable moves...")
        diverse_moves = self._filter_by_response_diversity(board, acceptable, min_responses=3)
        if diverse_moves:
            logger.debug(f"  {len(diverse_moves)} moves give opponent 3+ good responses")
            acceptable = diverse_moves
        else:
            # Fall back to requiring only 2 good responses
            logger.debug(f"  No moves with 3+ responses, trying 2+...")
            diverse_moves = [c for c in acceptable if c.get("opponent_good_responses", 0) >= 2]
            if diverse_moves:
                logger.debug(f"  {len(diverse_moves)} moves give opponent 2+ good responses")
                acceptable = diverse_moves
            else:
                logger.debug(f"  No moves with 2+ responses, using original acceptable list")

        # Filter out moves that create mate-in-1 threats
        # This makes survival-bot less aggressive
        logger.debug(f"  Checking for mate threats in {len(acceptable)} acceptable moves...")
        non_threatening = self._filter_by_mate_threats(board, acceptable)
        if non_threatening:
            logger.debug(f"  {len(non_threatening)} moves don't create mate threats")
            acceptable = non_threatening
        else:
            logger.debug(f"  All moves create mate threats, keeping original list")

        # Random selection from acceptable moves
        selected = self._rng.choice(acceptable)
        logger.debug(f"  SELECTED (from acceptable): {selected['move'].uci()} eval={selected['eval_cp']} delta={selected['delta_cp']} opponent_responses={selected.get('opponent_good_responses', 'N/A')}")
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
            logger.debug(f"[Ply {game_ply}] NEW GAME DETECTED - resetting state (last_ply was {self._last_ply})")
            self._last_ply = -1  # Reset ply tracking
            self._last_eval_cp = None  # Reset eval baseline

        # Update ply tracking
        self._update_ply_tracking(board)

        # Always play g3 as first move when white (ply 0)
        if game_ply == 0:
            g3_move = chess.Move.from_uci("g2g3")
            if g3_move in board.legal_moves:
                logger.debug(f"[Ply {game_ply}] FORCED OPENING: g3")
                self._last_eval_cp = self._get_eval_cp(board)
                return g3_move

        # Check if we're winning by too much - if so, skip book and use middlegame
        # algorithm which handles advantage cap (gives back to 0-200cp range)
        # Don't cap mate positions (eval >= 10000) - always take the mate
        current_eval = self._get_eval_cp(board)
        if self.ADVANTAGE_CAP_THRESHOLD <= current_eval < 10000:
            logger.debug(f"[Ply {game_ply}] Skipping book due to advantage cap (eval={current_eval})")
            return self._select_middlegame_move(board, game_ply)

        # Opening phase: try book moves first (first 20 half-moves = ~10 full moves)
        if game_ply <= 20:
            book_move = self._select_opening_move(board)
            if book_move is not None and book_move in board.legal_moves:
                # Store eval after book move for baseline tracking
                # After push, board.turn is opponent, so we negate to get our perspective
                try:
                    board.push(book_move)
                    self._last_eval_cp = -self._get_eval_cp(board)
                finally:
                    board.pop()
                logger.debug(f"[Ply {game_ply}] BOOK MOVE: {book_move.uci()}, stored baseline={self._last_eval_cp}")
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
