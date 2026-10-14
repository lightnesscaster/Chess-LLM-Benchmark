"""Private, operator-approved corrections to a single active rated game."""

from __future__ import annotations

import copy
import hashlib
import json
import time
from datetime import datetime
from functools import lru_cache
from uuid import UUID

import chess

from firebase_client import RESULTS_COLLECTION, get_firestore_client
from web.approved_players import normalize_email
from web.play_service import GameStateError, _board_from_state


RECOVERIES_COLLECTION = "play_recoveries"


class RecoveryError(ValueError):
    """The signed game does not match an available approved correction."""


def recovery_id(state: dict, email: str) -> str:
    """Address a private approval by its owner and exact affected position."""
    scope = {
        "owner_email": normalize_email(email),
        "model_name": state.get("model_name"),
        "human_color": state.get("human_color"),
        "expected_moves": state.get("moves"),
    }
    return hashlib.sha256(json.dumps(scope, sort_keys=True).encode()).hexdigest()


def _state_digest(state: dict) -> str:
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def _prepare_delivery(state: dict, record: dict, email: str) -> dict:
    """Permit redelivery only to the identical original signed game snapshot."""
    if record.get("used_game_id"):
        if (
            record["used_game_id"] != state.get("game_id")
            or record.get("source_state_sha256") != _state_digest(state)
        ):
            raise RecoveryError("This recovery has already been used for another game state.")
        record = {**record, "used_game_id": None}
    return prepare_recovery(state, record, email)


def prepare_recovery(
    state: dict, record: dict, email: str, *, now: float | None = None,
) -> dict:
    """Validate an approved last-turn correction, preserving all rated metadata."""
    message = "This game has no available approved recovery."
    try:
        instant = time.time() if now is None else now
        started = datetime.fromisoformat(state["started_at"])
        UUID(state["game_id"])
        if (
            started.tzinfo is None
            or not started.timestamp() < record["created_at"] <= instant < record["expires_at"]
            or record.get("used_game_id")
            or state.get("status") != "active"
            or state.get("winner") or state.get("termination") or state.get("rating_result")
            or not state.get("human_profile") or not state.get("rated_model_id")
            or normalize_email(record.get("owner_email")) != normalize_email(email)
            or not normalize_email(email)
            or record.get("model_name") != state.get("model_name")
            or record.get("human_color") != state.get("human_color")
            or record.get("expected_moves") != state.get("moves")
        ):
            raise RecoveryError(message)
        original_board = _board_from_state(state)
        restored_moves = record["restored_moves"]
        original_moves = state["moves"]
        if (
            not isinstance(restored_moves, list)
            or len(restored_moves) < 2
            or len(restored_moves) != len(original_moves)
            or restored_moves[:-2] != original_moves[:-2]
            or restored_moves == original_moves
        ):
            raise RecoveryError(message)
        restored = copy.deepcopy(state)
        restored["moves"] = list(restored_moves)
        board = _board_from_state(restored)
        human = {"white": chess.WHITE, "black": chess.BLACK}[state["human_color"]]
        if board.turn != human or original_board.turn != human or board.is_game_over() or len(restored_moves) >= 200:
            raise RecoveryError(message)
        prefix_ply = len(restored_moves) - 2
        # Boundary responses belong to the overwritten turn, while an offer
        # on the retained final move still belongs to the unchanged prefix.
        events = [
            event for event in restored.get("draw_events", [])
            if event["ply"] < prefix_ply
            or (event["ply"] == prefix_ply and event["action"] == "offer")
        ]
        if events and events[-1]["action"] == "offer":
            events.append({
                "action": "decline",
                "by": "white" if prefix_ply % 2 == 0 else "black",
                "ply": prefix_ply,
            })
        restored["draw_events"] = events
        restored["draw_offer"] = None
    except (KeyError, TypeError, ValueError, GameStateError) as error:
        raise RecoveryError(message) from error
    # Preserve paid usage, but the overwritten response's usage is unrecoverable.
    restored["llm_accounting_status"] = "partial"
    return restored


class PlayRecoveryStore:
    """Consume private approvals once, without creating results or changing ratings."""

    def __init__(self, db=None):
        self._db = db if db is not None else get_firestore_client()
        self._collection = self._db.collection(RECOVERIES_COLLECTION)

    def lookup(self, state: dict, email: str) -> dict | None:
        key = recovery_id(state, email)
        snapshot = self._collection.document(key).get()
        if not snapshot.exists:
            return None
        try:
            _prepare_delivery(state, snapshot.to_dict() or {}, email)
        except RecoveryError:
            return None
        return {"id": key}

    def restore(self, state: dict, email: str, approval_id: str) -> dict:
        """Atomically claim an exact active game before returning its correction."""
        from firebase_admin import firestore

        key = recovery_id(state, email)
        if approval_id != key:
            raise RecoveryError("This recovery does not match the current game.")
        reference = self._collection.document(key)
        try:
            game_id = str(UUID(state["game_id"]))
        except (KeyError, TypeError, ValueError) as error:
            raise RecoveryError("The saved game identity is invalid.") from error
        result_reference = self._db.collection(RESULTS_COLLECTION).document(game_id)

        @firestore.transactional
        def claim(transaction):
            snapshot = reference.get(transaction=transaction)
            result = result_reference.get(transaction=transaction)
            if not snapshot.exists or result.exists:
                raise RecoveryError("This game has no available approved recovery.")
            record = snapshot.to_dict() or {}
            restored = _prepare_delivery(state, record, email)
            if not record.get("used_game_id"):
                transaction.update(reference, {
                    "used_game_id": game_id, "used_at": time.time(),
                    "source_state_sha256": _state_digest(state),
                })
            return restored

        return claim(self._db.transaction())


@lru_cache(maxsize=1)
def get_play_recovery_store() -> PlayRecoveryStore:
    """Return the private approval store lazily."""
    return PlayRecoveryStore()
