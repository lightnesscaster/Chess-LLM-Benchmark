"""Authoritative human-versus-LLM chess game state for the admin UI."""

from __future__ import annotations

import asyncio
import copy
import os
from collections.abc import Callable, Mapping
from pathlib import Path

import chess
import yaml

from llm import (
    ClaudeCodePlayer,
    CodexSubagentPlayer,
    GeminiPlayer,
    OpenRouterPlayer,
    TransientAPIError,
    request_llm_move,
)
from llm.openrouter_completion_client import OpenRouterCompletionPlayer


class ConfigurationError(ValueError):
    """Raised when the web play providers or model selection are invalid."""


class GameStateError(ValueError):
    """Raised when a requested action is invalid for the current game."""


class ProviderError(RuntimeError):
    """Raised when the selected LLM cannot return a move."""


MoveProvider = Callable[[dict, chess.Board, bool, str | None], str | None]


def _load_config(config_path: Path) -> dict:
    try:
        with Path(config_path).open() as config_file:
            config = yaml.safe_load(config_file) or {}
    except (OSError, yaml.YAMLError) as error:
        raise ConfigurationError("The benchmark model configuration could not be loaded.") from error
    if not isinstance(config, dict):
        raise ConfigurationError("The benchmark model configuration is invalid.")
    return config


def _web_backend(model: dict) -> str:
    return str(model.get("web_api") or model.get("api") or "openrouter")


def _backend_is_configured(model: dict, environ: Mapping[str, str]) -> bool:
    backend = _web_backend(model)
    if backend not in {
        "openrouter",
        "completion",
        "gemini",
        "codex",
        "claude_code",
    }:
        return False
    if backend == "codex":
        return bool(environ.get("CODEX_AUTH_JSON_B64"))
    if backend == "claude_code":
        return bool(environ.get("CLAUDE_CODE_OAUTH_TOKEN"))
    if backend == "gemini":
        return bool(environ.get("GEMINI_API_KEY"))
    return bool(environ.get("OPENROUTER_API_KEY"))


def list_playable_models(
    config_path: Path,
    environ: Mapping[str, str] = os.environ,
) -> list[dict]:
    """List configured LLMs the deployed web process can call."""
    models = []
    seen_ids = set()
    for model in _load_config(config_path).get("llms", []) or []:
        if not isinstance(model, dict) or model.get("unavailable") is True:
            continue
        model_id = str(model.get("player_id") or "").strip()
        model_name = str(model.get("model_name") or "").strip()
        if (
            not model_id
            or not model_name
            or model_id in seen_ids
            or not _backend_is_configured(model, environ)
        ):
            continue
        seen_ids.add(model_id)
        models.append({
            "id": model_id,
            "name": model_id,
            "model_name": model_name,
        })
    return models


def _select_model(
    model_id: str,
    config_path: Path,
    environ: Mapping[str, str],
) -> dict:
    if not isinstance(model_id, str) or not model_id or len(model_id) > 200:
        raise ConfigurationError("Select a valid model.")

    playable_ids = {
        model["id"] for model in list_playable_models(config_path, environ)
    }
    if model_id not in playable_ids:
        raise ConfigurationError("That model is not available for web play.")

    for model in _load_config(config_path).get("llms", []) or []:
        if isinstance(model, dict) and model.get("player_id") == model_id:
            return copy.deepcopy(model)
    raise ConfigurationError("That model is not available for web play.")


def _new_state(model: dict, human_color: str) -> dict:
    if human_color not in {"white", "black"}:
        raise GameStateError("Choose either white or black pieces.")
    return {
        "model_id": model["player_id"],
        "model_name": model["model_name"],
        "human_color": human_color,
        "moves": [],
        "status": "active",
        "winner": None,
        "termination": None,
        "llm_illegal_moves": 0,
    }


def _board_from_state(state: dict) -> chess.Board:
    if not isinstance(state, dict):
        raise GameStateError("Start a new game.")
    moves = state.get("moves")
    if not isinstance(moves, list) or len(moves) > 200:
        raise GameStateError("The saved game state is invalid.")
    board = chess.Board()
    for move_uci in moves:
        try:
            move = chess.Move.from_uci(move_uci)
        except (TypeError, ValueError, chess.InvalidMoveError) as error:
            raise GameStateError("The saved game state is invalid.") from error
        if move not in board.legal_moves:
            raise GameStateError("The saved game state is invalid.")
        board.push(move)
    return board


def _validate_state(
    state: dict,
    config_path: Path,
    environ: Mapping[str, str],
) -> tuple[dict, chess.Board]:
    model = _select_model(state.get("model_id"), config_path, environ)
    if state.get("model_name") != model.get("model_name"):
        raise GameStateError("The saved game state is invalid.")
    if state.get("human_color") not in {"white", "black"}:
        raise GameStateError("The saved game state is invalid.")
    if state.get("status") not in {"active", "finished"}:
        raise GameStateError("The saved game state is invalid.")
    illegal_moves = state.get("llm_illegal_moves")
    if not isinstance(illegal_moves, int) or not 0 <= illegal_moves <= 2:
        raise GameStateError("The saved game state is invalid.")
    return model, _board_from_state(state)


def _human_chess_color(state: dict) -> chess.Color:
    return chess.WHITE if state["human_color"] == "white" else chess.BLACK


def _finish_from_board(state: dict, board: chess.Board) -> bool:
    outcome = board.outcome()
    if outcome is None:
        if len(state["moves"]) >= 200:
            state.update(status="finished", winner="draw", termination="max_moves")
            return True
        return False

    if outcome.winner is None:
        winner = "draw"
    elif outcome.winner == _human_chess_color(state):
        winner = "human"
    else:
        winner = "llm"
    state.update(
        status="finished",
        winner=winner,
        termination=outcome.termination.name.lower(),
    )
    return True


async def _request_model_move(
    model: dict,
    board: chess.Board,
    is_retry: bool,
    last_illegal_move: str | None,
    environ: Mapping[str, str],
) -> str | None:
    backend = _web_backend(model)
    common = {
        "player_id": model["player_id"],
        "temperature": model.get("temperature", 0.0),
        "timeout": model.get("timeout", 300),
    }
    if backend == "codex":
        player = CodexSubagentPlayer(
            **common,
            model_name=model["model_name"],
            reasoning_effort=model.get("reasoning_effort", "medium"),
            subscription_only=True,
        )
    elif backend == "claude_code":
        player = ClaudeCodePlayer(
            **common,
            model_name=model.get("web_model_name", model["model_name"]),
            reasoning_effort=model.get(
                "web_reasoning_effort",
                model.get("reasoning_effort", "medium"),
            ),
        )
    elif backend == "gemini":
        player = GeminiPlayer(
            **common,
            model_name=model["model_name"].removeprefix("google/"),
            api_key=environ.get("GEMINI_API_KEY"),
            reasoning=model.get("reasoning"),
            reasoning_effort=model.get("reasoning_effort"),
        )
    elif backend == "completion":
        player = OpenRouterCompletionPlayer(
            **common,
            model_name=model["model_name"],
            api_key=environ.get("OPENROUTER_API_KEY"),
            max_tokens=model.get("max_tokens", 8),
            provider_order=model.get("provider_order"),
            provider_ignore=model.get("provider_ignore"),
        )
    else:
        player = OpenRouterPlayer(
            **common,
            model_name=model["model_name"],
            api_key=environ.get("OPENROUTER_API_KEY"),
            max_tokens=model.get("max_tokens", 0),
            reasoning=model.get("reasoning"),
            reasoning_effort=model.get("reasoning_effort"),
            reasoning_max_tokens=model.get("reasoning_max_tokens"),
            provider_order=model.get("provider_order"),
            provider_ignore=model.get("provider_ignore"),
        )

    try:
        return await request_llm_move(
            player,
            board,
            is_retry=is_retry,
            last_move_illegal=last_illegal_move,
        )
    finally:
        await player.close()


def _default_move_provider(
    model: dict,
    board: chess.Board,
    is_retry: bool,
    last_illegal_move: str | None,
    environ: Mapping[str, str],
) -> str | None:
    try:
        return asyncio.run(
            _request_model_move(
                model,
                board,
                is_retry,
                last_illegal_move,
                environ,
            )
        )
    except TransientAPIError:
        raise
    except Exception as error:
        raise ProviderError("The LLM could not provide a move.") from error


def _apply_llm_turn(
    state: dict,
    board: chess.Board,
    model: dict,
    environ: Mapping[str, str],
    move_provider: MoveProvider | None,
) -> None:
    provider = move_provider
    last_illegal_move = None

    while True:
        is_retry = last_illegal_move is not None
        try:
            if provider is None:
                move_uci = _default_move_provider(
                    model,
                    board.copy(stack=True),
                    is_retry,
                    last_illegal_move,
                    environ,
                )
            else:
                move_uci = provider(
                    copy.deepcopy(model),
                    board.copy(stack=True),
                    is_retry,
                    last_illegal_move,
                )
        except TransientAPIError as error:
            raise ProviderError("The LLM could not provide a move.") from error

        normalized = str(move_uci or "").strip().lower()
        try:
            move = chess.Move.from_uci(normalized)
        except (ValueError, chess.InvalidMoveError):
            move = None

        if move is not None and move in board.legal_moves:
            board.push(move)
            state["moves"].append(move.uci())
            _finish_from_board(state, board)
            return

        state["llm_illegal_moves"] += 1
        if state["llm_illegal_moves"] >= 2:
            state.update(
                status="finished",
                winner="human",
                termination="llm_forfeit_illegal_move",
            )
            return
        last_illegal_move = normalized or "invalid"


def start_game(
    model_id: str,
    human_color: str,
    config_path: Path,
    environ: Mapping[str, str] = os.environ,
    move_provider: MoveProvider | None = None,
) -> dict:
    """Create a game and make the opening LLM move when the human is black."""
    model = _select_model(model_id, config_path, environ)
    state = _new_state(model, human_color)
    if human_color == "black":
        _apply_llm_turn(state, chess.Board(), model, environ, move_provider)
    return state


def play_human_move(
    state: dict,
    move_uci: str,
    config_path: Path,
    environ: Mapping[str, str] = os.environ,
    move_provider: MoveProvider | None = None,
) -> dict:
    """Apply a legal human move and, if needed, the configured LLM reply."""
    working_state = copy.deepcopy(state)
    model, board = _validate_state(working_state, config_path, environ)
    if working_state["status"] != "active":
        raise GameStateError("This game is already finished.")
    if board.turn != _human_chess_color(working_state):
        raise GameStateError("It is not your turn.")

    normalized = str(move_uci or "").strip().lower()
    try:
        move = chess.Move.from_uci(normalized)
    except (ValueError, chess.InvalidMoveError) as error:
        raise GameStateError("Choose a legal move.") from error
    if move not in board.legal_moves:
        raise GameStateError("Choose a legal move.")

    board.push(move)
    working_state["moves"].append(move.uci())
    if not _finish_from_board(working_state, board):
        _apply_llm_turn(
            working_state,
            board,
            model,
            environ,
            move_provider,
        )
    return working_state


def game_view(state: dict) -> dict:
    """Serialize signed game state into browser-facing board information."""
    board = chess.Board()
    san_moves = []
    for move_uci in state.get("moves", []):
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise GameStateError("The saved game state is invalid.")
        san_moves.append(board.san(move))
        board.push(move)

    if state.get("status") == "finished":
        turn = "finished"
    else:
        human_color = _human_chess_color(state)
        turn = "human" if board.turn == human_color else "llm"

    return {
        "fen": board.fen(),
        "moves": list(state.get("moves", [])),
        "san_moves": san_moves,
        "model_id": state.get("model_id"),
        "model_name": state.get("model_name"),
        "human_color": state.get("human_color"),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "turn": turn,
        "status": state.get("status"),
        "winner": state.get("winner"),
        "termination": state.get("termination"),
        "last_move": state.get("moves", [])[-1] if state.get("moves") else None,
        "llm_illegal_moves": state.get("llm_illegal_moves", 0),
    }
