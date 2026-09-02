"""Authoritative human-versus-LLM chess game state for the admin UI."""

from __future__ import annotations

import asyncio
import copy
import json
import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path

import chess
import chess.pgn
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

EFFORT_LABELS = {
    "default": "Auto",
    "none": "None",
    "minimal": "Minimal",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "xhigh": "XHigh",
    "max": "Max",
}
EFFORT_SUFFIX = re.compile(
    r"\s+\((?:no thinking|thinking|default|minimal|low|medium|high|xhigh|max)\)$",
    re.IGNORECASE,
)


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


def _configured_models(config: dict) -> list:
    return list(config.get("web_play_models", []) or []) + list(
        config.get("llms", []) or []
    )


def _verified_claude_models(environ: Mapping[str, str]) -> set[str]:
    catalog_path = Path(
        environ.get(
            "CLAUDE_MODEL_CATALOG_PATH",
            "/tmp/chessbench_claude_models.json",
        )
    )
    try:
        payload = json.loads(catalog_path.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return set()
    return {str(model) for model in models if isinstance(model, str)}


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
        model_name = str(model.get("web_model_name") or model.get("model_name") or "")
        return bool(environ.get("CLAUDE_CODE_OAUTH_TOKEN")) and (
            model_name in _verified_claude_models(environ)
        )
    if backend == "gemini":
        return bool(environ.get("GEMINI_API_KEY"))
    return bool(environ.get("OPENROUTER_API_KEY"))


def _configured_efforts(model: dict) -> tuple[list[str], str]:
    explicit = model.get("web_reasoning_efforts")
    if isinstance(explicit, list):
        efforts = []
        for value in explicit:
            effort = str(value).strip().lower()
            if effort in EFFORT_LABELS and effort not in efforts:
                efforts.append(effort)
        if efforts:
            default = str(
                model.get("web_default_reasoning_effort") or efforts[0]
            ).strip().lower()
            return efforts, default if default in efforts else efforts[0]

    if model.get("reasoning") is False:
        return ["none"], "none"
    player_id = str(model.get("player_id") or "").strip().lower()
    if (
        "reasoning" not in model
        and not model.get("web_reasoning_effort")
        and not model.get("reasoning_effort")
        and player_id.endswith("(no thinking)")
    ):
        return ["none"], "none"
    effort = str(
        model.get("web_reasoning_effort")
        or model.get("reasoning_effort")
        or "default"
    ).strip().lower()
    if effort not in EFFORT_LABELS:
        effort = "default"
    return [effort], effort


def _model_with_effort(model: dict, effort: str) -> dict:
    selected = copy.deepcopy(model)
    if effort == "none":
        selected["reasoning"] = False
        selected.pop("reasoning_effort", None)
        selected.pop("web_reasoning_effort", None)
    elif effort != "default":
        if _web_backend(selected) == "claude_code":
            selected["web_reasoning_effort"] = effort
        else:
            selected["reasoning_effort"] = effort
        selected.pop("reasoning", None)
    selected["_web_effort"] = effort
    return selected


def _playable_model_groups(
    config_path: Path,
    environ: Mapping[str, str],
) -> list[dict]:
    groups_by_model = {}
    used_ids = set()
    for model in _configured_models(_load_config(config_path)):
        if (
            not isinstance(model, dict)
            or model.get("unavailable") is True
            or model.get("web_hidden") is True
            or not _backend_is_configured(model, environ)
        ):
            continue
        player_id = str(model.get("player_id") or "").strip()
        model_name = str(model.get("model_name") or "").strip()
        if not player_id or not model_name:
            continue

        group = groups_by_model.get(model_name)
        if group is None:
            display_name = EFFORT_SUFFIX.sub("", player_id).strip() or player_id
            group_id = display_name
            if group_id in used_ids:
                group_id = f"{display_name} [{model_name}]"
            used_ids.add(group_id)
            group = {
                "id": group_id,
                "name": display_name,
                "model_name": model_name,
                "variants": {},
                "default_effort": None,
            }
            groups_by_model[model_name] = group

        efforts, default_effort = _configured_efforts(model)
        for effort in efforts:
            if effort not in group["variants"]:
                group["variants"][effort] = _model_with_effort(model, effort)
        if group["default_effort"] is None or model.get(
            "web_default_reasoning_effort"
        ):
            group["default_effort"] = default_effort

    return list(groups_by_model.values())


def list_playable_models(
    config_path: Path,
    environ: Mapping[str, str] = os.environ,
) -> list[dict]:
    """List configured LLMs the deployed web process can call."""
    return [
        {
            "id": group["id"],
            "name": group["name"],
            "model_name": group["model_name"],
            "efforts": [
                {"id": effort, "name": EFFORT_LABELS[effort]}
                for effort in group["variants"]
            ],
            "default_effort": group["default_effort"],
        }
        for group in _playable_model_groups(config_path, environ)
    ]


def _select_model(
    model_id: str,
    config_path: Path,
    environ: Mapping[str, str],
    reasoning_effort: str | None = None,
) -> dict:
    if not isinstance(model_id, str) or not model_id or len(model_id) > 200:
        raise ConfigurationError("Select a valid model.")

    for group in _playable_model_groups(config_path, environ):
        if group["id"] != model_id:
            continue
        effort = str(reasoning_effort or group["default_effort"]).strip().lower()
        selected = group["variants"].get(effort)
        if selected is None:
            raise ConfigurationError("That effort is not available for this model.")
        selected = copy.deepcopy(selected)
        selected["player_id"] = group["id"]
        selected["_web_effort"] = effort
        return selected
    raise ConfigurationError("That model is not available for web play.")


def _new_state(model: dict, human_color: str) -> dict:
    if human_color not in {"white", "black"}:
        raise GameStateError("Choose either white or black pieces.")
    return {
        "model_id": model["player_id"],
        "model_name": model["model_name"],
        "reasoning_effort": model.get("_web_effort", "default"),
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
    model = _select_model(
        state.get("model_id"),
        config_path,
        environ,
        state.get("reasoning_effort"),
    )
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
    reasoning_effort: str | None = None,
) -> dict:
    """Create a game and make the opening LLM move when the human is black."""
    model = _select_model(model_id, config_path, environ, reasoning_effort)
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
    pgn_game = chess.pgn.Game()
    pgn_game.headers["Event"] = "Human vs LLM"
    pgn_game.headers["Site"] = "ChessBench"
    model_id = str(state.get("model_id") or "LLM")
    if state.get("human_color") == "black":
        pgn_game.headers["White"] = model_id
        pgn_game.headers["Black"] = "You"
    else:
        pgn_game.headers["White"] = "You"
        pgn_game.headers["Black"] = model_id
    pgn_node = pgn_game
    for move_uci in state.get("moves", []):
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            raise GameStateError("The saved game state is invalid.")
        san_moves.append(board.san(move))
        board.push(move)
        pgn_node = pgn_node.add_variation(move)

    result = "*"
    if state.get("status") == "finished":
        if state.get("winner") == "draw":
            result = "1/2-1/2"
        elif state.get("winner") == "human":
            result = "1-0" if state.get("human_color") == "white" else "0-1"
        elif state.get("winner") == "llm":
            result = "0-1" if state.get("human_color") == "white" else "1-0"
    pgn_game.headers["Result"] = result
    if state.get("termination"):
        pgn_game.headers["Termination"] = str(state["termination"])

    if state.get("status") == "finished":
        turn = "finished"
    else:
        human_color = _human_chess_color(state)
        turn = "human" if board.turn == human_color else "llm"

    return {
        "fen": board.fen(),
        "pgn": str(pgn_game),
        "moves": list(state.get("moves", [])),
        "san_moves": san_moves,
        "model_id": state.get("model_id"),
        "model_name": state.get("model_name"),
        "reasoning_effort": state.get("reasoning_effort", "default"),
        "human_color": state.get("human_color"),
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "turn": turn,
        "status": state.get("status"),
        "winner": state.get("winner"),
        "termination": state.get("termination"),
        "last_move": state.get("moves", [])[-1] if state.get("moves") else None,
        "llm_illegal_moves": state.get("llm_illegal_moves", 0),
    }
