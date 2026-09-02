"""Persistence and one-sided scoring for approved human challenges."""

from __future__ import annotations

from datetime import datetime, timezone

import chess
import chess.pgn

from game.models import GameResult
from game.pgn_logger import PGNLogger
from rating.glicko2 import Glicko2System, PlayerRating
from rating.rating_store import RatingStore, invalidate_cache
from utils import is_reasoning_model


class HumanChallengeError(ValueError):
    """Raised when a human challenge cannot be scored."""


def _model_rating(rating_store: RatingStore, model_id: str) -> PlayerRating:
    """Match recalculation's fallback seed for a previously unseen model."""
    existed = rating_store.has_player(model_id)
    rating = rating_store.get(model_id)
    if (
        not existed
        and rating.games_played == 0
        and rating.rating == 1500
        and rating.rating_deviation == 350
    ):
        initial = 1200 if is_reasoning_model(model_id) else 400
        return PlayerRating(
            player_id=model_id,
            rating=initial,
            rating_deviation=350,
            games_rd=350,
            unclamped_rating=initial,
        )
    return rating


def _human_id(result_or_state) -> str:
    if isinstance(result_or_state, GameResult):
        username = result_or_state.human_lichess_username
    else:
        profile = result_or_state.get("human_profile")
        username = profile.get("username") if isinstance(profile, dict) else None
    return f"lichess:{username}"


def build_human_challenge_result(state, email, completed_at=None) -> GameResult:
    """Convert a finished signed-session game into a durable result record."""
    if not isinstance(state, dict) or state.get("status") != "finished":
        raise HumanChallengeError("Only a finished game can be scored.")
    if state.get("winner") not in {"human", "llm", "draw"}:
        raise HumanChallengeError("The finished game has no valid result.")
    profile = state.get("human_profile")
    if not isinstance(profile, dict):
        raise HumanChallengeError("The game has no Lichess rating snapshot.")
    username = str(profile.get("username") or "").strip()
    human_rating = profile.get("rating")
    human_rd = profile.get("rating_deviation")
    if (
        not username
        or isinstance(human_rating, bool)
        or not isinstance(human_rating, (int, float))
        or isinstance(human_rd, bool)
        or not isinstance(human_rd, (int, float))
        or not 0 <= human_rating <= 4000
        or not 0 < human_rd <= 500
    ):
        raise HumanChallengeError("The game has no valid Lichess rating snapshot.")
    model_id = str(state.get("rated_model_id") or state.get("model_id") or "").strip()
    game_id = str(state.get("game_id") or "").strip()
    human_color = state.get("human_color")
    if not game_id or not model_id or human_color not in {"white", "black"}:
        raise HumanChallengeError("The finished game state is invalid.")

    human_id = _human_id(state)
    white_id, black_id = (
        (human_id, model_id) if human_color == "white" else (model_id, human_id)
    )
    if state["winner"] == "draw":
        winner = "draw"
    elif state["winner"] == "human":
        winner = human_color
    else:
        winner = "black" if human_color == "white" else "white"

    move_count = len(state.get("moves") or [])
    human_illegal = 0
    llm_illegal = int(state.get("llm_illegal_moves") or 0)
    return GameResult(
        game_id=game_id,
        white_id=white_id,
        black_id=black_id,
        winner=winner,
        termination=str(state.get("termination") or "unknown"),
        moves=move_count,
        illegal_moves_white=human_illegal if human_color == "white" else llm_illegal,
        illegal_moves_black=llm_illegal if human_color == "white" else human_illegal,
        total_moves_white=(move_count + 1) // 2,
        total_moves_black=move_count // 2,
        pgn_path="",
        created_at=completed_at or datetime.now(timezone.utc).isoformat(),
        game_type="human_challenge",
        human_email=str(email or "").strip().casefold(),
        human_lichess_username=username,
        human_rating=float(human_rating),
        human_rating_deviation=float(human_rd),
        human_rating_provisional=bool(profile.get("provisional", False)),
    )


def score_model_against_human(model_rating: PlayerRating, result: GameResult) -> PlayerRating:
    """Apply one fixed human snapshot to the model and never update the human."""
    human_id = _human_id(result)
    if result.game_type != "human_challenge" or human_id not in {
        result.white_id,
        result.black_id,
    }:
        raise HumanChallengeError("This is not a valid human challenge result.")
    if model_rating.player_id not in {result.white_id, result.black_id}:
        raise HumanChallengeError("The model rating does not match this game.")
    opponent = PlayerRating(
        player_id=human_id,
        rating=float(result.human_rating),
        rating_deviation=float(result.human_rating_deviation),
        games_rd=float(result.human_rating_deviation),
    )
    if result.winner == "draw":
        score = 0.5
    elif result.winner == "white":
        score = 1.0 if result.white_id == model_rating.player_id else 0.0
    else:
        score = 1.0 if result.black_id == model_rating.player_id else 0.0
    return Glicko2System().update_rating(model_rating, [opponent], [score])


def _pgn_for_state(state: dict, result: GameResult) -> str:
    game = chess.pgn.Game()
    game.headers.update({
        "Event": "Rated Human Challenge",
        "Site": "ChessBench",
        "Date": result.created_at[:10].replace("-", "."),
        "White": result.white_id,
        "Black": result.black_id,
        "Result": (
            "1-0" if result.winner == "white" else
            "0-1" if result.winner == "black" else "1/2-1/2"
        ),
        "Termination": result.termination,
        "LichessRapid": str(round(result.human_rating)),
        "LichessRapidRD": str(round(result.human_rating_deviation)),
    })
    board = game.board()
    node = game
    for move_uci in state.get("moves") or []:
        try:
            move = chess.Move.from_uci(move_uci)
        except (TypeError, ValueError, chess.InvalidMoveError) as error:
            raise HumanChallengeError("The finished game moves are invalid.") from error
        if move not in board.legal_moves:
            raise HumanChallengeError("The finished game moves are invalid.")
        node = node.add_variation(move)
        board.push(move)
    return str(game)


def _record_firestore_challenge(
    state: dict,
    email: str,
    rating_store: RatingStore,
    pgn_logger: PGNLogger,
) -> dict:
    """Atomically claim a game ID, save it, and update the model rating."""
    from firebase_admin import firestore

    result = build_human_challenge_result(state, email)
    model_id = result.black_id if result.white_id.startswith("lichess:") else result.white_id
    fallback_rating = _model_rating(rating_store, model_id)
    result.pgn_path = f"firestore://{pgn_logger._games_collection}/{result.game_id}"
    pgn = _pgn_for_state(state, result)

    result_ref = pgn_logger._db.collection(pgn_logger._results_collection).document(result.game_id)
    game_ref = pgn_logger._db.collection(pgn_logger._games_collection).document(result.game_id)
    rating_ref = rating_store._db.collection(rating_store._collection).document(model_id)
    transaction = pgn_logger._db.transaction()

    @firestore.transactional
    def apply(transaction):
        existing_result = result_ref.get(transaction=transaction)
        rating_document = rating_ref.get(transaction=transaction)
        current = (
            PlayerRating.from_dict(rating_document.to_dict())
            if rating_document.exists
            else fallback_rating
        )
        if existing_result.exists:
            return False, current

        updated = score_model_against_human(current, result)
        transaction.set(game_ref, {"game_id": result.game_id, "pgn": pgn})
        transaction.set(result_ref, result.to_json())
        transaction.set(rating_ref, updated.to_dict())
        return True, updated

    recorded, updated_rating = apply(transaction)
    invalidate_cache()
    return {
        "recorded": recorded,
        "model_rating": round(updated_rating.rating),
        "model_rating_deviation": round(updated_rating.rating_deviation),
    }


def record_human_challenge(state, email, rating_store=None, pgn_logger=None) -> dict:
    """Persist and score a completed game once per game ID."""
    selected_logger = pgn_logger or PGNLogger()
    selected_store = rating_store or RatingStore()
    if (
        rating_store is None
        and pgn_logger is None
        and selected_logger._use_firestore
        and selected_store._use_firestore
    ):
        return _record_firestore_challenge(
            state,
            email,
            selected_store,
            selected_logger,
        )

    game_id = str(state.get("game_id") or "") if isinstance(state, dict) else ""
    existing = selected_logger.load_result(game_id) if game_id else None
    if existing is not None:
        model_id = existing.black_id if existing.white_id.startswith("lichess:") else existing.white_id
        rating = selected_store.get(model_id)
        return {
            "recorded": False,
            "model_rating": round(rating.rating),
            "model_rating_deviation": round(rating.rating_deviation),
        }

    result = build_human_challenge_result(state, email)
    model_id = result.black_id if result.white_id.startswith("lichess:") else result.white_id
    current_rating = _model_rating(selected_store, model_id)
    updated_rating = score_model_against_human(current_rating, result)
    selected_logger.save_game(result, _pgn_for_state(state, result))
    selected_store.set(updated_rating)
    invalidate_cache()
    return {
        "recorded": True,
        "model_rating": round(updated_rating.rating),
        "model_rating_deviation": round(updated_rating.rating_deviation),
    }
