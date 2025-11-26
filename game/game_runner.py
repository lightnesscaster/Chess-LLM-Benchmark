"""
Game runner that plays a chess game between two players.

Handles:
- Turn-based play
- Illegal move detection with retry policy (2 strikes = forfeit)
- Game termination conditions
- PGN generation
"""

import uuid
import asyncio
import tempfile
import os
from datetime import datetime, timezone
from typing import Union, Tuple, Optional

import chess
import chess.pgn

from engines.base_engine import BaseEngine
from llm.base_llm import BaseLLMPlayer
from .models import GameResult

# Live game file for following along
LIVE_GAME_FILE = "/tmp/chess_llm_live_game.txt"


# Type alias for players (either engine or LLM)
Player = Union[BaseEngine, BaseLLMPlayer]


class GameRunner:
    """
    Runs a single chess game between two players.

    Enforces the illegal move policy:
    - First illegal move: Warn and retry
    - Second illegal move in the same game: Forfeit
    """

    def __init__(
        self,
        white: Player,
        black: Player,
        max_moves: int = 200,
        verbose: bool = False,
    ):
        """
        Initialize the game runner.

        Args:
            white: Player with white pieces
            black: Player with black pieces
            max_moves: Maximum number of half-moves (plies) before draw
            verbose: Print moves as they happen
        """
        self.white = white
        self.black = black
        self.max_moves = max_moves
        self.verbose = verbose

    def _write_live_game(self, white_id: str, black_id: str, board: chess.Board,
                          moves_played: int, last_move: str = None, status: str = "in_progress"):
        """Write current game state to live file for following along."""
        try:
            with open(LIVE_GAME_FILE, 'w') as f:
                f.write(f"=== LIVE GAME ===\n")
                f.write(f"White: {white_id}\n")
                f.write(f"Black: {black_id}\n")
                f.write(f"Status: {status}\n")
                f.write(f"Moves played: {moves_played}\n")
                if last_move:
                    f.write(f"Last move: {last_move}\n")
                f.write(f"\nBoard:\n{board}\n")
                f.write(f"\nFEN: {board.fen()}\n")
        except Exception:
            pass  # Don't fail the game if we can't write the live file

    async def play_game(self) -> Tuple[GameResult, str]:
        """
        Play a complete game.

        Returns:
            Tuple of (GameResult, PGN string)
        """
        game_id = str(uuid.uuid4())
        board = chess.Board()

        # Reset token counters for LLM players
        if isinstance(self.white, BaseLLMPlayer):
            self.white.reset_token_usage()
        if isinstance(self.black, BaseLLMPlayer):
            self.black.reset_token_usage()

        # Set up PGN
        pgn_game = chess.pgn.Game()
        pgn_game.headers["Event"] = "LLM Chess Benchmark"
        pgn_game.headers["Site"] = "Local"
        pgn_game.headers["Date"] = datetime.now(timezone.utc).strftime("%Y.%m.%d")
        pgn_game.headers["Round"] = "1"
        pgn_game.headers["White"] = self._get_player_id(self.white)
        pgn_game.headers["Black"] = self._get_player_id(self.black)

        node = pgn_game

        # Track illegal moves per player for this game
        illegal_count = {chess.WHITE: 0, chess.BLACK: 0}
        total_moves = {chess.WHITE: 0, chess.BLACK: 0}

        winner = "draw"
        termination = "normal"
        moves_played = 0

        # Write initial game state
        self._write_live_game(
            self._get_player_id(self.white),
            self._get_player_id(self.black),
            board, 0, status="starting"
        )

        while not board.is_game_over() and moves_played < self.max_moves:
            side = board.turn
            player = self.white if side == chess.WHITE else self.black

            # Get move with illegal retry policy
            move_result = await self._get_move_with_retry(
                player, board, side, illegal_count
            )

            if move_result is None:
                # Player forfeited due to second illegal move
                if side == chess.WHITE:
                    winner = "black"
                    termination = "forfeit_illegal_move"
                else:
                    winner = "white"
                    termination = "forfeit_illegal_move"
                break

            # Valid move received
            move_uci, _ = move_result
            total_moves[side] += 1

            try:
                chess_move = chess.Move.from_uci(move_uci)
                board.push(chess_move)
                node = node.add_variation(chess_move)
                moves_played += 1

                if self.verbose:
                    side_name = "White" if side == chess.WHITE else "Black"
                    print(f"  {moves_played}. {side_name}: {move_uci}")

                # Write live game state
                self._write_live_game(
                    self._get_player_id(self.white),
                    self._get_player_id(self.black),
                    board, moves_played, move_uci
                )

            except Exception as e:
                # Should not happen if validation is correct, but safety check
                print(f"Error applying move {move_uci}: {e}")
                if side == chess.WHITE:
                    winner = "black"
                else:
                    winner = "white"
                termination = "error"
                break

        # Determine final result if game ended naturally
        if board.is_game_over() and termination == "normal":
            outcome = board.outcome()
            if outcome is not None:
                if outcome.winner == chess.WHITE:
                    winner = "white"
                elif outcome.winner == chess.BLACK:
                    winner = "black"
                else:
                    winner = "draw"
                termination = outcome.termination.name.lower()
            else:
                winner = "draw"
                termination = "unknown"
        elif moves_played >= self.max_moves and termination == "normal":
            winner = "draw"
            termination = "max_moves"

        # Set PGN result
        pgn_result_map = {"white": "1-0", "black": "0-1", "draw": "1/2-1/2"}
        pgn_game.headers["Result"] = pgn_result_map.get(winner, "*")
        pgn_game.headers["Termination"] = termination
        pgn_str = str(pgn_game)

        # Collect token usage from LLM players
        tokens_white = None
        tokens_black = None
        if isinstance(self.white, BaseLLMPlayer):
            tokens_white = self.white.get_token_usage()
        if isinstance(self.black, BaseLLMPlayer):
            tokens_black = self.black.get_token_usage()

        # Build result object
        game_result = GameResult(
            game_id=game_id,
            white_id=self._get_player_id(self.white),
            black_id=self._get_player_id(self.black),
            winner=winner,
            termination=termination,
            moves=moves_played,
            illegal_moves_white=illegal_count[chess.WHITE],
            illegal_moves_black=illegal_count[chess.BLACK],
            total_moves_white=total_moves[chess.WHITE],
            total_moves_black=total_moves[chess.BLACK],
            pgn_path="",  # Will be set by logger
            created_at=datetime.now(timezone.utc).isoformat(),
            tokens_white=tokens_white,
            tokens_black=tokens_black,
        )

        return game_result, pgn_str

    async def _get_move_with_retry(
        self,
        player: Player,
        board: chess.Board,
        side: chess.Color,
        illegal_count: dict,
    ) -> Optional[Tuple[str, bool]]:
        """
        Get a move from the player with illegal move retry policy.

        Args:
            player: The player to get move from
            board: Current board state
            side: Which side is to move
            illegal_count: Dict tracking illegal moves per side

        Returns:
            Tuple of (uci_move, was_retry) or None if player forfeits
        """
        max_attempts = 2
        last_illegal_move = None

        for attempt in range(max_attempts):
            is_retry = attempt > 0

            # Get move from player
            move_uci = await self._ask_player_for_move(
                player, board, is_retry, last_illegal_move
            )

            # Validate the move
            if move_uci is not None:
                is_legal, validated_move = self._validate_move(board, move_uci)

                if is_legal:
                    return (validated_move, is_retry)

            # Move was illegal
            illegal_count[side] += 1
            last_illegal_move = move_uci or "invalid"

            if self.verbose:
                side_name = "White" if side == chess.WHITE else "Black"
                print(f"  {side_name} illegal move #{illegal_count[side]}: {last_illegal_move}")

            # Check if this is the second illegal move
            if illegal_count[side] >= 2:
                if self.verbose:
                    print(f"  {side_name} forfeits due to second illegal move")
                return None

        # Should not reach here, but safety
        return None

    async def _ask_player_for_move(
        self,
        player: Player,
        board: chess.Board,
        is_retry: bool,
        last_illegal_move: Optional[str],
    ) -> Optional[str]:
        """
        Ask a player for their move.

        Args:
            player: The player (engine or LLM)
            board: Current board state
            is_retry: Whether this is a retry after illegal move
            last_illegal_move: The previous illegal move attempt

        Returns:
            UCI move string or None
        """
        try:
            if isinstance(player, BaseEngine):
                # Engines return chess.Move directly
                move = player.select_move(board)
                return move.uci()
            else:
                # LLMs return UCI string
                move_uci = await player.select_move(
                    board,
                    is_retry=is_retry,
                    last_move_illegal=last_illegal_move,
                )
                return move_uci.strip() if move_uci else None

        except Exception as e:
            if self.verbose:
                print(f"  Error getting move from {self._get_player_id(player)}: {e}")
            return None

    def _validate_move(self, board: chess.Board, move_uci: str) -> Tuple[bool, str]:
        """
        Validate that a move is legal.

        Args:
            board: Current board state
            move_uci: Move in UCI format

        Returns:
            Tuple of (is_legal, normalized_move_uci)
        """
        try:
            # Clean up the move string
            move_uci = move_uci.strip().lower()

            # Try to parse as UCI
            move = chess.Move.from_uci(move_uci)

            # Check if legal
            if move in board.legal_moves:
                return (True, move.uci())
            else:
                return (False, move_uci)

        except (ValueError, chess.InvalidMoveError):
            return (False, move_uci)

    def _get_player_id(self, player: Player) -> str:
        """Get the player ID string."""
        return player.player_id
