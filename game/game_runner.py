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
import time
import logging
import copy
import json
import math
import io
from pathlib import Path
from contextlib import nullcontext
from datetime import datetime, timezone
from typing import Union, Tuple, Optional

import chess
import chess.pgn

from engines.base_engine import BaseEngine
from llm import BaseLLMPlayer, TransientAPIError, request_llm_move
from llm.protocol import split_draw_offer
from .models import GameResult
from . import live_games
from .draw_offers import annotate_draw_events
from .checkpoints import CheckpointStore, CheckpointError

logger = logging.getLogger(__name__)


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
        pre_moves: list = None,
        rating_snapshots: dict | None = None,
        publish_live: bool = False,
        checkpoint_dir: str | Path | None = None,
        resume_checkpoint: str | Path | None = None,
    ):
        """
        Initialize the game runner.

        Args:
            white: Player with white pieces
            black: Player with black pieces
            max_moves: Maximum number of half-moves (plies) before draw
            verbose: Print moves as they happen
            pre_moves: List of UCI move strings to pre-play before starting
            publish_live: Publish public snapshots for a saved rated run
            checkpoint_dir: Private directory for durable saved-game recovery
            resume_checkpoint: Restore a saved game instead of starting a new one
        """
        self.white = white
        self.black = black
        self.max_moves = max_moves
        self.verbose = verbose
        self.pre_moves = pre_moves
        self.rating_snapshots = copy.deepcopy(rating_snapshots)
        self.publish_live = publish_live
        self._draw_offer = None
        self._draw_events = []
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.resume_checkpoint = Path(resume_checkpoint) if resume_checkpoint is not None else None
        if self.resume_checkpoint is not None and pre_moves:
            raise ValueError("A checkpoint cannot be combined with pre-moves")
        self._checkpoint_store = None
        self._pending_request = None
        self._pending_illegal_move = None
        self._partial_accounting = set()
        self._checkpoint_interval = 1.0

    def _publish_live(self, game_id: str, board: chess.Board, illegal_count: dict,
                      **state) -> None:
        """Keep all snapshot work outside move validation's error handling."""
        if self._live_publisher is None:
            return
        try:
            self._live_publisher.submit(live_games.build_snapshot(
                game_id, self._get_player_id(self.white), self._get_player_id(self.black),
                board, started_at=self._live_started_at,
                illegal_moves_white=illegal_count[chess.WHITE],
                illegal_moves_black=illegal_count[chess.BLACK],
                draw_offer=self._draw_offer, **state,
            ))
        except Exception:
            logger.warning("Live snapshot could not be queued")

    def _finish_live(self, status: str, termination: str) -> None:
        if self._live_publisher is None:
            return
        try:
            self._live_publisher.finish(status, termination)
        except Exception:
            logger.warning("Live terminal snapshot could not be queued")

    async def play_game(self) -> Tuple[GameResult, str]:
        """Play a game with optional, bounded best-effort live publication."""
        self._live_publisher = None
        self._live_started_at = time.time()
        if self.publish_live:
            try:
                publisher = live_games.LiveGamePublisher(live_games.get_live_game_store())
                publisher.start()
                self._live_publisher = publisher
            except Exception:
                logger.warning("Live publication could not start")
        try:
            self._game_id = str(uuid.uuid4())
            path = self.resume_checkpoint
            if path is None and self.checkpoint_dir is not None:
                path = self.checkpoint_dir / f"{self._game_id}.json"
            with CheckpointStore(path) if path is not None else nullcontext(None) as store:
                self._checkpoint_store = store
                return await self._play_game()
        except asyncio.CancelledError:
            self._finish_live("interrupted", "cancelled")
            raise
        except Exception:
            self._finish_live("error", "error")
            raise
        finally:
            self._checkpoint_store = None
            if self._live_publisher is not None:
                try:
                    await self._live_publisher.close()
                except Exception:
                    logger.warning("Live publication flush unavailable")

    async def _play_game(self) -> Tuple[GameResult, str]:
        """
        Play a complete game.

        Returns:
            Tuple of (GameResult, PGN string)
        """
        game_id = self._game_id
        board = chess.Board()
        self._draw_offer = None
        self._draw_events = []
        self._pending_request = None
        self._pending_illegal_move = None
        self._partial_accounting = set()

        # Reset token counters and timing for LLM players
        if isinstance(self.white, BaseLLMPlayer):
            self.white.reset_token_usage()
            self.white.reset_timing()
        if isinstance(self.black, BaseLLMPlayer):
            self.black.reset_token_usage()
            self.black.reset_timing()

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
        retry_attempts = {chess.WHITE: 0, chess.BLACK: 0}
        retry_recoveries = {chess.WHITE: 0, chess.BLACK: 0}
        retry_failures = {chess.WHITE: 0, chess.BLACK: 0}
        total_moves = {chess.WHITE: 0, chess.BLACK: 0}
        illegal_move_details = []  # List of dicts with debugging info

        winner = "draw"
        termination = "normal"
        moves_played = 0

        self._counter_state = {
            "illegal_count": illegal_count, "retry_attempts": retry_attempts,
            "retry_recoveries": retry_recoveries, "retry_failures": retry_failures,
            "total_moves": total_moves,
        }
        self._game_board = board
        self._pgn_game = pgn_game
        self._illegal_move_details = illegal_move_details
        if self.resume_checkpoint is not None:
            state = self._checkpoint_store.read()
            self._restore_checkpoint(state, board, pgn_game)
            game_id = self._game_id
            moves_played = len(board.move_stack)
            node = pgn_game.end()
            if state.get("completed") is not None:
                return GameResult.from_json(state["completed"]["result"]), state["completed"]["pgn"]

        # Pre-play moves if resuming a game
        if self.pre_moves:
            for uci_str in self.pre_moves:
                move = chess.Move.from_uci(uci_str)
                if move not in board.legal_moves:
                    raise ValueError(f"Illegal pre-move: {uci_str} at position {board.fen()}")
                board.push(move)
                node = node.add_variation(move)
                moves_played += 1
            if self.verbose:
                print(f"  Resumed from move {moves_played} ({len(self.pre_moves)} pre-played moves)")
                print(f"  Position: {board.fen()}")
                print()

        self._save_checkpoint()
        if self.verbose and self._checkpoint_store is not None:
            print(f"  Checkpoint: {self._checkpoint_store.path.resolve()}")
        self._publish_live(game_id, board, illegal_count)

        while not board.is_game_over() and moves_played < self.max_moves:
            side = board.turn
            player = self.white if side == chess.WHITE else self.black

            # Get move with illegal retry policy
            try:
                move_result = await self._get_move_with_retry(
                    player,
                    board,
                    side,
                    illegal_count,
                    illegal_move_details,
                    moves_played,
                    retry_attempts,
                    retry_recoveries,
                    retry_failures,
                )
            except TransientAPIError as e:
                # Network error after retries - end as draw, not a forfeit
                if self.verbose:
                    side_name = "White" if side == chess.WHITE else "Black"
                    model_info = ""
                    if isinstance(player, BaseLLMPlayer):
                        model_info = f" [{player.model_name}]"
                        # Include inference provider if available (OpenRouter specific)
                        provider = getattr(player, 'last_provider', None)
                        if provider:
                            model_info += f" via {provider}"
                    print(f"  {side_name} ({player.player_id}{model_info}) API error (not counted as illegal move): {e}")
                winner = "draw"
                termination = "api_error"
                break

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
            if move_uci == "resign":
                self._draw_offer = None
                winner = "black" if side == chess.WHITE else "white"
                termination = "resignation"
                break
            side_name = "white" if side == chess.WHITE else "black"
            if move_uci == "accept_draw":
                self._draw_events.append({"action": "accept", "by": side_name, "ply": moves_played})
                self._draw_offer = None
                winner, termination = "draw", "draw_agreement"
                if self.verbose:
                    print(f"  {side_name.title()} accepted the draw offer")
                break
            move_uci, offer_draw = split_draw_offer(move_uci)
            if self._draw_offer is not None:
                self._draw_events.append({"action": "decline", "by": side_name, "ply": moves_played})
                self._draw_offer = None
            total_moves[side] += 1

            try:
                chess_move = chess.Move.from_uci(move_uci)
                board.push(chess_move)
                node = node.add_variation(chess_move)
                moves_played += 1
                if offer_draw and not board.is_game_over() and moves_played < self.max_moves:
                    self._draw_offer = side_name
                    self._draw_events.append({"action": "offer", "by": side_name, "ply": moves_played})

                # Mark response as successful for LLM players (for context in next prompt)
                if isinstance(player, BaseLLMPlayer):
                    player.mark_move_successful()

                if self.verbose:
                    side_name = "White" if side == chess.WHITE else "Black"
                    token_info = ""
                    if isinstance(player, BaseLLMPlayer) and player.total_tokens > 0:
                        last_prompt = getattr(player, '_last_prompt_tokens', 0)
                        last_completion = getattr(player, '_last_completion_tokens', 0)
                        if last_prompt or last_completion:
                            token_info = f"  (tokens: {last_prompt}p/{last_completion}c)"
                    print(f"  {moves_played}. {side_name}: {move_uci}{' (draw offered)' if self._draw_offer else ''}{token_info}")

            except Exception as e:
                # Should not happen if validation is correct, but safety check
                print(f"Error applying move {move_uci}: {e}")
                if side == chess.WHITE:
                    winner = "black"
                else:
                    winner = "white"
                termination = "error"
                break

            self._publish_live(game_id, board, illegal_count)
            self._pending_request = None
            self._pending_illegal_move = None
            self._save_checkpoint()

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
        annotate_draw_events(pgn_game, self._draw_events)
        pgn_str = str(pgn_game)

        # Collect token usage and timing from LLM players
        tokens_white = None
        tokens_black = None
        timing_white = None
        timing_black = None
        if isinstance(self.white, BaseLLMPlayer):
            tokens_white = self.white.get_token_usage()
            timing_white = self.white.get_timing_usage()
        if isinstance(self.black, BaseLLMPlayer):
            tokens_black = self.black.get_token_usage()
            timing_black = self.black.get_timing_usage()

        # Build result object
        # Keep the initial and resumed result identical while keeping request text
        # out of the durable checkpoint. No-save runs retain their debug details.
        if self._checkpoint_store is not None:
            illegal_move_details[:] = self._checkpoint_details(illegal_move_details)
        game_result = GameResult(
            game_id=game_id,
            white_id=self._get_player_id(self.white),
            black_id=self._get_player_id(self.black),
            winner=winner,
            termination=termination,
            moves=moves_played,
            illegal_moves_white=illegal_count[chess.WHITE],
            illegal_moves_black=illegal_count[chess.BLACK],
            retry_attempts_white=retry_attempts[chess.WHITE],
            retry_attempts_black=retry_attempts[chess.BLACK],
            retry_recoveries_white=retry_recoveries[chess.WHITE],
            retry_recoveries_black=retry_recoveries[chess.BLACK],
            retry_failures_white=retry_failures[chess.WHITE],
            retry_failures_black=retry_failures[chess.BLACK],
            retry_unknown_white=(
                retry_attempts[chess.WHITE]
                - retry_recoveries[chess.WHITE]
                - retry_failures[chess.WHITE]
            ),
            retry_unknown_black=(
                retry_attempts[chess.BLACK]
                - retry_recoveries[chess.BLACK]
                - retry_failures[chess.BLACK]
            ),
            total_moves_white=total_moves[chess.WHITE],
            total_moves_black=total_moves[chess.BLACK],
            pgn_path="",  # Will be set by logger
            created_at=datetime.now(timezone.utc).isoformat(),
            tokens_white=tokens_white,
            tokens_black=tokens_black,
            timing_white=timing_white,
            timing_black=timing_black,
            illegal_move_details=illegal_move_details if illegal_move_details else None,
            accounting_status_white="partial" if "white" in self._partial_accounting else None,
            accounting_status_black="partial" if "black" in self._partial_accounting else None,
        )

        # API failures remain resumable; they are not completed rated games.
        if termination not in ("api_error", "error"):
            self._pending_request = None
            self._pending_illegal_move = None
            self._save_checkpoint(completed={"result": game_result.to_json(), "pgn": pgn_str})
        else:
            self._save_checkpoint()

        self._publish_live(
            game_id, board, illegal_count,
            status="error" if termination in ("api_error", "error") else "finished",
            winner=winner, termination=termination,
        )
        return game_result, pgn_str

    @staticmethod
    def _checkpoint_details(details: list) -> list:
        return [{key: value for key, value in detail.items()
                 if key not in ("prompt", "raw_response")} for detail in details]

    @staticmethod
    def _player_identity(player: Player) -> dict:
        """Whitelist reproducible settings, never credentials or request contents."""
        fields = (
            "model_name", "temperature", "max_tokens", "reasoning", "reasoning_effort",
            "reasoning_max_tokens", "provider_order", "provider_ignore", "include_legal_moves",
            "subscription_only", "extra_args", "engine_path", "options", "rating",
            "move_time", "nodes", "depth", "weights_path", "skill_level",
            "lc0_path", "stockfish_path", "opening_book_path", "base_depth",
            "blunder_threshold_cp", "book_draw_threshold",
        )
        identity = {"class": f"{type(player).__module__}.{type(player).__qualname__}",
                    "player_id": player.player_id}
        for field in fields:
            if hasattr(player, field):
                identity[field] = getattr(player, field)
        return json.loads(json.dumps(identity, default=str, allow_nan=False))

    def _save_checkpoint(self, *, clock_overrides: dict | None = None,
                         completed: dict | None = None) -> None:
        if self._checkpoint_store is None:
            return
        players = {}
        for side, player in (("white", self.white), ("black", self.black)):
            clock = None
            if clock_overrides is not None and side in clock_overrides:
                clock = clock_overrides[side]
            elif hasattr(player, "snapshot_clock"):
                clock = player.snapshot_clock()
            players[side] = {"identity": self._player_identity(player), "clock": clock}
            if isinstance(player, BaseLLMPlayer):
                players[side]["tokens"] = player.get_token_usage()
                players[side]["timing"] = player.get_timing_usage()
        state = {
            "version": 1, "game_id": self._game_id,
            "started_at": self._live_started_at, "max_moves": self.max_moves,
            "moves": [move.uci() for move in self._game_board.move_stack],
            "fen": self._game_board.fen(), "headers": dict(self._pgn_game.headers),
            "players": players, "rating_snapshots": self.rating_snapshots,
            "counters": {key: {"white": value[chess.WHITE], "black": value[chess.BLACK]}
                         for key, value in self._counter_state.items()},
            "draw_offer": self._draw_offer, "draw_events": self._draw_events,
            "pending_request": self._pending_request,
            "pending_illegal_move": self._pending_illegal_move,
            "partial_accounting": sorted(self._partial_accounting),
            "illegal_move_details": self._checkpoint_details(self._illegal_move_details),
            "completed": completed,
        }
        self._checkpoint_store.write(state)

    def _restore_checkpoint(self, state: dict, board: chess.Board,
                            pgn_game: chess.pgn.Game) -> None:
        """Validate the entire position and matchup before restoring mutable state."""
        try:
            if type(state["version"]) is not int or state["version"] != 1:
                raise ValueError("Unsupported checkpoint version")
            uuid.UUID(state["game_id"])
            if type(state["max_moves"]) is not int or state["max_moves"] <= 0:
                raise ValueError("Invalid checkpoint move limit")
            if not isinstance(state["moves"], list):
                raise ValueError("Invalid checkpoint moves")
            replay = chess.Board()
            for uci in state["moves"]:
                move = chess.Move.from_uci(uci)
                if move not in replay.legal_moves:
                    raise ValueError("Illegal checkpoint history")
                replay.push(move)
            if replay.fen() != state["fen"] or len(replay.move_stack) > state["max_moves"]:
                raise ValueError("Checkpoint position does not match its history")
            for side, player in (("white", self.white), ("black", self.black)):
                saved = state["players"][side]
                if saved["identity"] != self._player_identity(player):
                    raise ValueError(f"Checkpoint {side} player configuration does not match")
                if bool(saved["clock"] is not None) != bool(getattr(player, "_use_clock", False)):
                    raise ValueError("Checkpoint is missing the engine clock")
                if isinstance(player, BaseLLMPlayer):
                    tokens, timing = saved["tokens"], saved["timing"]
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        if type(tokens[key]) is not int or tokens[key] < 0:
                            raise ValueError("Invalid checkpoint token accounting")
                    values = [timing["total_time"], *timing["move_times"]]
                    if any(isinstance(value, bool) or not isinstance(value, (int, float))
                           or not math.isfinite(value) or value < 0 for value in values):
                        raise ValueError("Invalid checkpoint timing")
            for key in self._counter_state:
                for side in ("white", "black"):
                    value = state["counters"][key][side]
                    if type(value) is not int or value < 0:
                        raise ValueError("Invalid checkpoint counters")
            events = copy.deepcopy(state["draw_events"])
            proof = chess.pgn.Game.from_board(replay)
            annotate_draw_events(proof, events)
            pending_offer = None
            for event in events:
                pending_offer = event["by"] if event["action"] == "offer" else None
            if state["draw_offer"] != pending_offer:
                # A terminal resignation can clear a pending offer without a decline.
                terminal = state.get("completed")
                if not (terminal and terminal["result"]["termination"] == "resignation"
                        and state["draw_offer"] is None):
                    raise ValueError("Checkpoint draw context is inconsistent")
            pending = state["pending_request"]
            if pending is not None and (
                pending["side"] != ("white" if replay.turn else "black")
                or type(pending["is_retry"]) is not bool
                or pending["last_illegal_move"] != state["pending_illegal_move"]
            ):
                raise ValueError("Invalid checkpoint pending request")
            if not set(state["partial_accounting"]) <= {"white", "black"}:
                raise ValueError("Invalid checkpoint accounting status")
            completed = state.get("completed")
            if completed is not None:
                result = GameResult.from_json(completed["result"])
                parsed = chess.pgn.read_game(io.StringIO(completed["pgn"]))
                if (result.game_id != state["game_id"] or result.moves != len(replay.move_stack)
                        or result.white_id != self.white.player_id or result.black_id != self.black.player_id
                        or parsed is None or parsed.errors or parsed.end().board().fen() != replay.fen()):
                    raise ValueError("Invalid completed checkpoint")
        except (KeyError, TypeError, AttributeError, OverflowError) as exc:
            raise ValueError("Malformed game checkpoint") from exc

        for side, player in (("white", self.white), ("black", self.black)):
            saved = state["players"][side]
            if saved["clock"] is not None:
                player.restore_clock(saved["clock"])
            if isinstance(player, BaseLLMPlayer):
                for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    setattr(player, key, saved["tokens"][key])
                player._input_accounting_requests = copy.deepcopy(saved["tokens"].get("input_accounting_requests", []))
                player.move_times = list(saved["timing"]["move_times"])
                player.total_move_time = saved["timing"]["total_time"]
        for move in replay.move_stack:
            board.push(move)
        pgn_game.headers.clear()
        pgn_game.headers.update(state["headers"])
        node = pgn_game
        for move in replay.move_stack:
            node = node.add_variation(move)
        for key, values in self._counter_state.items():
            values.update({chess.WHITE: state["counters"][key]["white"],
                           chess.BLACK: state["counters"][key]["black"]})
        self._game_id = state["game_id"]
        self.max_moves = state["max_moves"]
        self._live_started_at = state["started_at"]
        self.rating_snapshots = copy.deepcopy(state["rating_snapshots"])
        self._draw_offer = state["draw_offer"]
        self._draw_events = events
        self._pending_request = pending
        self._pending_illegal_move = state["pending_illegal_move"]
        self._partial_accounting = set(state["partial_accounting"])
        if pending is not None and isinstance(self.white if board.turn else self.black, BaseLLMPlayer):
            self._partial_accounting.add(pending["side"])
        self._illegal_move_details.extend(state["illegal_move_details"])

    async def _clocked_engine_move(self, player: BaseEngine, board: chess.Board) -> chess.Move:
        """Checkpoint the pre-move clock minus elapsed thinking, with no increment."""
        side = "white" if board.turn else "black"
        snapshot = player.snapshot_clock()
        started = time.perf_counter()
        search = asyncio.create_task(asyncio.to_thread(player.select_move, board.copy()))
        try:
            while True:
                done, _ = await asyncio.wait({search}, timeout=self._checkpoint_interval)
                if done:
                    return search.result()
                clock = copy.deepcopy(snapshot)
                clock[f"{side}_clock"] = max(0.0, snapshot[f"{side}_clock"] - (time.perf_counter() - started))
                self._save_checkpoint(clock_overrides={side: clock})
        except BaseException:
            # Threaded UCI searches cannot be cancelled by cancelling their awaiter.
            # Stop the process before releasing the checkpoint lease.
            try:
                await asyncio.to_thread(player.close)
            except Exception:
                logger.warning("Interrupted engine could not close cleanly")
            await asyncio.gather(search, return_exceptions=True)
            raise

    async def _get_move_with_retry(
        self,
        player: Player,
        board: chess.Board,
        side: chess.Color,
        illegal_count: dict,
        illegal_move_details: list,
        move_number: int,
        retry_attempts: dict,
        retry_recoveries: dict,
        retry_failures: dict,
    ) -> Optional[Tuple[str, bool]]:
        """
        Get a move from the player with illegal move retry policy.

        Args:
            player: The player to get move from
            board: Current board state
            side: Which side is to move
            illegal_count: Dict tracking illegal moves per side
            illegal_move_details: List to append illegal move debug info to
            move_number: Current move number in the game

        Returns:
            Tuple of (uci_move, was_retry) or None if player forfeits

        Raises:
            TransientAPIError: If the API call fails due to network issues
        """
        max_attempts = 2
        last_illegal_move = self._pending_illegal_move
        if illegal_count[side] >= 2:
            return None

        for attempt in range(max_attempts):
            is_retry = attempt > 0 or last_illegal_move is not None
            if is_retry and self._pending_request is None:
                retry_attempts[side] += 1

            self._pending_request = {
                "side": "white" if side else "black", "is_retry": is_retry,
                "last_illegal_move": last_illegal_move,
            }
            self._save_checkpoint()

            # Get move from player (may raise TransientAPIError)
            move_uci = await self._ask_player_for_move(
                player, board, is_retry, last_illegal_move
            )

            if isinstance(move_uci, str) and move_uci.strip().casefold() == "resign":
                return ("resign", is_retry)

            allow_draw = isinstance(self.white, BaseLLMPlayer) and isinstance(self.black, BaseLLMPlayer)
            side_name = "white" if side == chess.WHITE else "black"
            if (allow_draw and move_uci == "accept_draw"
                    and self._draw_offer is not None and self._draw_offer != side_name):
                if is_retry:
                    retry_recoveries[side] += 1
                return ("accept_draw", is_retry)

            # Validate the move
            if move_uci is not None:
                candidate, offer_draw = split_draw_offer(move_uci)
                is_legal, validated_move = self._validate_move(board, candidate)
                is_legal = is_legal and (not offer_draw or allow_draw)

                if is_legal:
                    if is_retry:
                        retry_recoveries[side] += 1
                    return (validated_move + (" offer_draw" if offer_draw else ""), is_retry)

            # Move was illegal
            illegal_count[side] += 1
            if is_retry:
                retry_failures[side] += 1
            last_illegal_move = move_uci or "invalid"
            self._pending_illegal_move = last_illegal_move
            self._pending_request = None

            if self.verbose:
                side_name = "White" if side == chess.WHITE else "Black"
                print(f"  {side_name} illegal move #{illegal_count[side]}: {last_illegal_move}")

            # Capture debugging details for LLM players
            if isinstance(player, BaseLLMPlayer):
                illegal_move_details.append({
                    "side": "white" if side == chess.WHITE else "black",
                    "move_number": move_number + 1,  # 1-indexed for readability
                    "fen": board.fen(),
                    "prompt": player.last_prompt,
                    "raw_response": player.last_raw_response,
                    "parsed_move": last_illegal_move,
                })

            self._save_checkpoint()

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

        Raises:
            TransientAPIError: If the API call fails due to network issues (not an illegal move)
        """
        try:
            if isinstance(player, BaseEngine):
                # Engines return chess.Move directly
                if self._checkpoint_store is not None and getattr(player, "_use_clock", False):
                    move = await self._clocked_engine_move(player, board)
                else:
                    move = player.select_move(board)
                return move.uci()
            else:
                # LLMs return UCI string
                move_uci = await request_llm_move(
                    player,
                    board,
                    is_retry=is_retry,
                    last_move_illegal=last_illegal_move,
                    allow_resignation=True,
                    allow_draw_offers=isinstance(self.white, BaseLLMPlayer) and isinstance(self.black, BaseLLMPlayer),
                    opponent_draw_offer=(self._draw_offer is not None and self._draw_offer != ("white" if board.turn else "black")),
                    rating_context={
                        "self": self.rating_snapshots.get(player.player_id),
                        "opponent": self.rating_snapshots.get(
                            self.black.player_id if player is self.white else self.white.player_id
                        ),
                    } if self.rating_snapshots is not None else None,
                )
                return move_uci

        except TransientAPIError:
            # Let transient API errors propagate - these aren't illegal moves
            raise

        except CheckpointError:
            raise

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
