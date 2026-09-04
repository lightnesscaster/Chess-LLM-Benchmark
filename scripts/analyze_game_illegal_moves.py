#!/usr/bin/env python3
"""Analyze illegal LLM moves against reconstructed legal-turn controls.

The default mode reads result and PGN documents from Firestore without mutating
remote state. It writes only aggregate statistics and sanitized chess-event data;
full stored prompts are never copied into the output artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from io import StringIO
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable

import chess
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from firebase_client import (  # noqa: E402
    GAMES_COLLECTION,
    RESULTS_COLLECTION,
    get_firestore_client,
)


UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)


def load_json_mapping(path: Path) -> dict[str, Any]:
    """Load a dictionary or a list keyed by game_id from JSON."""
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return {
            str(row.get("game_id") or index): row
            for index, row in enumerate(data)
        }
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object or list in {path}")
    return {str(key): value for key, value in data.items()}


def selected_game(
    game: dict[str, Any],
    player_prefixes: tuple[str, ...],
) -> bool:
    """Return whether either player matches the requested prefixes."""
    return any(
        str(game.get(f"{side}_id") or "").startswith(player_prefixes)
        for side in ("white", "black")
    )


def load_games_and_pgns(
    *,
    results_input: Path | None,
    pgns_input: Path | None,
    player_prefixes: tuple[str, ...],
) -> tuple[dict[str, dict[str, Any]], dict[str, str], int]:
    """Load selected result records and their PGNs from local files or Firestore."""
    if results_input is not None:
        all_results = load_json_mapping(results_input)
        results = {
            game_id: game
            for game_id, game in all_results.items()
            if selected_game(game, player_prefixes)
        }
        if pgns_input is None:
            raise ValueError("--pgns-input is required with --results-input")
        raw_pgns = load_json_mapping(pgns_input)
        pgns = {
            game_id: str(value.get("pgn") if isinstance(value, dict) else value)
            for game_id, value in raw_pgns.items()
            if game_id in results
        }
        return results, pgns, len(all_results)

    if pgns_input is not None:
        raise ValueError("--pgns-input requires --results-input")

    db = get_firestore_client()
    all_result_docs = list(db.collection(RESULTS_COLLECTION).stream())
    results = {
        doc.id: doc.to_dict()
        for doc in all_result_docs
        if selected_game(doc.to_dict(), player_prefixes)
    }
    references = [
        db.collection(GAMES_COLLECTION).document(game_id)
        for game_id in results
    ]
    pgns = {
        doc.id: str((doc.to_dict() or {}).get("pgn") or "")
        for doc in db.get_all(references)
        if doc.exists
    }
    return results, pgns, len(all_result_docs)


def side_name(turn: chess.Color) -> str:
    return "white" if turn == chess.WHITE else "black"


def opponent_type(player_id: str) -> str:
    """Assign a compact, stable opponent category."""
    if player_id == "random-bot":
        return "random"
    if player_id.startswith("maia-"):
        return "maia"
    if player_id.startswith("gpt-"):
        return "gpt"
    if player_id in {"eubos", "survival-bot"}:
        return "other_engine"
    return "other_llm"


def ply_phase(ply_number: int) -> str:
    if ply_number <= 20:
        return "opening_1_20"
    if ply_number <= 60:
        return "middlegame_21_60"
    return "late_61_plus"


def material_phase(piece_count: int) -> str:
    if piece_count >= 26:
        return "opening_like_26_plus"
    if piece_count >= 13:
        return "middlegame_13_25"
    return "endgame_12_or_less"


def history_bucket(history_plies: int) -> str:
    if history_plies < 20:
        return "0_19"
    if history_plies < 40:
        return "20_39"
    if history_plies < 80:
        return "40_79"
    return "80_plus"


def legal_move_bucket(legal_moves: int) -> str:
    if legal_moves <= 10:
        return "1_10"
    if legal_moves <= 20:
        return "11_20"
    if legal_moves <= 30:
        return "21_30"
    return "31_plus"


def last_move_features(board: chess.Board) -> tuple[bool, bool]:
    """Return whether the immediately preceding legal move captured or checked."""
    if not board.move_stack:
        return False, False
    previous = board.copy(stack=True)
    move = previous.pop()
    captured = previous.is_capture(move)
    return captured, board.is_check()


def legal_if_ignoring_last_opponent_move(
    board: chess.Board,
    move: chess.Move,
) -> bool:
    """Heuristically test whether a move fits the board before the last reply."""
    if not board.move_stack:
        return False
    previous = board.copy(stack=True)
    previous.pop()
    previous.turn = board.turn
    return move in previous.legal_moves


def classify_illegal_attempt(
    board: chess.Board,
    parsed_move: Any,
    *,
    previous_own_move: str | None,
    previous_turn_board: chess.Board | None,
) -> dict[str, Any]:
    """Classify the structural reason an attempted move is illegal."""
    text = str(parsed_move or "").strip().lower()
    output: dict[str, Any] = {
        "primary_class": "format_invalid",
        "detailed_class": "format_invalid",
        "source_piece": None,
        "well_formed_uci": False,
        "matches_previous_own_move": bool(previous_own_move and text == previous_own_move),
        "legal_on_previous_target_turn": False,
        "legal_if_ignoring_last_opponent_move": False,
        "stale_board_signal": False,
    }
    if not UCI_RE.fullmatch(text):
        return output

    output["well_formed_uci"] = True
    try:
        move = chess.Move.from_uci(text)
    except ValueError:
        output["detailed_class"] = "invalid_uci_coordinates"
        return output

    output["legal_if_ignoring_last_opponent_move"] = (
        legal_if_ignoring_last_opponent_move(board, move)
    )
    if previous_turn_board is not None:
        output["legal_on_previous_target_turn"] = move in previous_turn_board.legal_moves

    piece = board.piece_at(move.from_square)
    destination = board.piece_at(move.to_square)
    if piece is not None:
        output["source_piece"] = chess.piece_name(piece.piece_type)
    if piece is None:
        output["primary_class"] = "source_empty"
        output["detailed_class"] = "source_empty"
    elif piece.color != board.turn:
        output["primary_class"] = "wrong_side_piece"
        output["detailed_class"] = "wrong_side_piece"
    elif (
        piece.piece_type == chess.PAWN
        and chess.square_rank(move.to_square) in (0, 7)
        and move.promotion is None
    ):
        output["primary_class"] = "missing_promotion"
        output["detailed_class"] = "missing_promotion"
    elif destination is not None and destination.color == board.turn:
        output["primary_class"] = "destination_own_piece"
        output["detailed_class"] = "destination_own_piece"
    elif move in board.legal_moves:
        output["primary_class"] = "unexpectedly_legal"
        output["detailed_class"] = "unexpectedly_legal"
    elif board.is_pseudo_legal(move):
        output["primary_class"] = "king_safety"
        output["detailed_class"] = "king_safety"
    elif piece.piece_type == chess.KING and abs(
        chess.square_file(move.to_square) - chess.square_file(move.from_square)
    ) == 2:
        output["primary_class"] = "illegal_castling"
        output["detailed_class"] = "illegal_castling"
    else:
        output["primary_class"] = "movement_rule_or_blocked"
        file_delta = abs(
            chess.square_file(move.to_square) - chess.square_file(move.from_square)
        )
        rank_delta = abs(
            chess.square_rank(move.to_square) - chess.square_rank(move.from_square)
        )
        if piece.piece_type == chess.PAWN:
            if file_delta == 0 and destination is not None:
                output["detailed_class"] = "pawn_forward_blocked"
            elif file_delta == 1 and destination is None:
                output["detailed_class"] = "pawn_diagonal_without_capture"
            else:
                output["detailed_class"] = "pawn_movement_rule"
        elif piece.piece_type == chess.KNIGHT:
            output["detailed_class"] = "knight_geometry"
        elif piece.piece_type == chess.BISHOP:
            output["detailed_class"] = (
                "slider_path_blocked" if file_delta == rank_delta else "bishop_geometry"
            )
        elif piece.piece_type == chess.ROOK:
            output["detailed_class"] = (
                "slider_path_blocked"
                if file_delta == 0 or rank_delta == 0
                else "rook_geometry"
            )
        elif piece.piece_type == chess.QUEEN:
            output["detailed_class"] = (
                "slider_path_blocked"
                if file_delta == rank_delta or file_delta == 0 or rank_delta == 0
                else "queen_geometry"
            )
        else:
            output["detailed_class"] = "king_geometry_or_attacked"
    output["stale_board_signal"] = bool(
        output["matches_previous_own_move"]
        or output["legal_on_previous_target_turn"]
        or output["legal_if_ignoring_last_opponent_move"]
    )
    return output


def turn_features(
    board: chess.Board,
    *,
    game_id: str,
    player_id: str,
    opponent_id: str,
    ply_number: int,
    prior_strike: bool,
) -> dict[str, Any]:
    """Extract board and context features available on legal and illegal turns."""
    legal_moves = list(board.legal_moves)
    piece_count = len(board.piece_map())
    captured, checked = last_move_features(board)
    return {
        "game_id": game_id,
        "player_id": player_id,
        "opponent_id": opponent_id,
        "opponent_type": opponent_type(opponent_id),
        "side": side_name(board.turn),
        "ply_number": ply_number,
        "ply_phase": ply_phase(ply_number),
        "history_plies": len(board.move_stack),
        "history_bucket": history_bucket(len(board.move_stack)),
        "material_phase": material_phase(piece_count),
        "piece_count": piece_count,
        "legal_move_count": len(legal_moves),
        "legal_move_bucket": legal_move_bucket(len(legal_moves)),
        "in_check": board.is_check(),
        "castling_rights": board.has_castling_rights(board.turn),
        "promotion_available": any(move.promotion for move in legal_moves),
        "last_move_capture": captured,
        "last_move_check": checked,
        "prior_strike": prior_strike,
        "fen": board.fen(),
    }


def response_style(raw_response: Any, parsed_move: Any) -> str:
    """Classify whether the raw response followed the requested surface form."""
    raw = str(raw_response or "").strip()
    parsed = str(parsed_move or "").strip()
    if not raw:
        return "empty"
    if raw.lower() == parsed.lower():
        return "bare_move"
    if re.fullmatch(r"MOVE:\s*\S+", raw, re.IGNORECASE):
        return "move_prefix"
    return "extra_text"


def group_illegal_details(
    result: dict[str, Any],
    side: str,
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for detail in result.get("illegal_move_details") or []:
        if detail.get("side") != side:
            continue
        try:
            move_number = int(detail.get("move_number"))
        except (TypeError, ValueError):
            continue
        grouped[move_number].append(detail)
    return grouped


def analyze_game(
    game_id: str,
    result: dict[str, Any],
    pgn_text: str,
    player_prefixes: tuple[str, ...],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reconstruct first-attempt turns and individual illegal response events."""
    pgn_game = chess.pgn.read_game(StringIO(pgn_text))
    if pgn_game is None:
        raise ValueError(f"Could not parse PGN for {game_id}")
    moves = list(pgn_game.mainline_moves())
    board = pgn_game.board()
    sides = {
        side: str(result.get(f"{side}_id") or "")
        for side in ("white", "black")
    }
    selected_sides = {
        side
        for side, player_id in sides.items()
        if player_id.startswith(player_prefixes)
    }
    details = {
        side: group_illegal_details(result, side)
        for side in selected_sides
    }
    prior_strikes = {side: 0 for side in selected_sides}
    previous_own_move: dict[str, str | None] = {side: None for side in selected_sides}
    previous_turn_board: dict[str, chess.Board | None] = {
        side: None for side in selected_sides
    }
    turns: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    used_detail_moves: dict[str, set[int]] = defaultdict(set)

    def add_turn(side: str, ply_number: int) -> None:
        player_id = sides[side]
        other = "black" if side == "white" else "white"
        row = turn_features(
            board,
            game_id=game_id,
            player_id=player_id,
            opponent_id=sides[other],
            ply_number=ply_number,
            prior_strike=prior_strikes[side] > 0,
        )
        illegal_details = details[side].get(ply_number, [])
        row["first_attempt_illegal"] = bool(illegal_details)
        row["illegal_attempts_on_turn"] = len(illegal_details)
        turns.append(row)
        if not illegal_details:
            return

        used_detail_moves[side].add(ply_number)
        prior_strikes[side] += 1
        for attempt_index, detail in enumerate(illegal_details):
            classification = classify_illegal_attempt(
                board,
                detail.get("parsed_move"),
                previous_own_move=previous_own_move[side],
                previous_turn_board=previous_turn_board[side],
            )
            events.append(
                {
                    **row,
                    "attempt_kind": "first_attempt" if attempt_index == 0 else "retry",
                    "parsed_move": detail.get("parsed_move"),
                    "raw_response": detail.get("raw_response"),
                    "response_style": response_style(
                        detail.get("raw_response"), detail.get("parsed_move")
                    ),
                    **classification,
                }
            )

    for ply_index, move in enumerate(moves, start=1):
        side = side_name(board.turn)
        if side in selected_sides:
            add_turn(side, ply_index)
            previous_turn_board[side] = board.copy(stack=True)
            previous_own_move[side] = move.uci()
        board.push(move)

    final_ply = len(moves) + 1
    final_side = side_name(board.turn)
    if (
        final_side in selected_sides
        and final_ply in details[final_side]
        and final_ply not in used_detail_moves[final_side]
    ):
        add_turn(final_side, final_ply)

    unused = {
        (side, move_number)
        for side in selected_sides
        for move_number in details[side]
        if move_number not in used_detail_moves[side]
    }
    if unused:
        raise ValueError(f"Unmatched illegal details in {game_id}: {sorted(unused)}")
    return turns, events


def count_rows(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    """Return attempts, illegals, and rates for each value of one feature."""
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        value = str(row.get(key))
        counts[value][0] += 1
        counts[value][1] += int(row.get("first_attempt_illegal") is True)
    return [
        {
            key: value,
            "attempts": attempts,
            "illegals": illegals,
            "illegal_pct": 100.0 * illegals / attempts if attempts else None,
        }
        for value, (attempts, illegals) in sorted(counts.items())
    ]


def counter_rows(rows: Iterable[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts = Counter(str(row.get(key)) for row in rows)
    return [{key: value, "count": count} for value, count in counts.most_common()]


def joint_rate_rows(
    rows: Iterable[dict[str, Any]],
    first_key: str,
    second_key: str,
) -> list[dict[str, Any]]:
    """Return illegality rates for the Cartesian cells of two features."""
    counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        key = (str(row.get(first_key)), str(row.get(second_key)))
        counts[key][0] += 1
        counts[key][1] += int(row.get("first_attempt_illegal") is True)
    return [
        {
            first_key: first_value,
            second_key: second_value,
            "attempts": attempts,
            "illegals": illegals,
            "illegal_pct": 100.0 * illegals / attempts if attempts else None,
        }
        for (first_value, second_value), (attempts, illegals) in sorted(counts.items())
    ]


def mantel_haenszel_odds_ratio(
    rows: Iterable[dict[str, Any]],
    key: str,
    exposed_value: Any,
) -> float | None:
    """Estimate a player-stratified common odds ratio for one binary feature."""
    strata: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for row in rows:
        exposed = row.get(key) == exposed_value
        illegal = row.get("first_attempt_illegal") is True
        index = 0 if exposed and illegal else 1 if exposed else 2 if illegal else 3
        strata[str(row.get("player_id"))][index] += 1

    numerator = 0.0
    denominator = 0.0
    for illegal_exposed, legal_exposed, illegal_control, legal_control in strata.values():
        total = illegal_exposed + legal_exposed + illegal_control + legal_control
        if total <= 0:
            continue
        numerator += illegal_exposed * legal_control / total
        denominator += legal_exposed * illegal_control / total
    if denominator == 0:
        return math.inf if numerator > 0 else None
    return numerator / denominator


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(probability * (len(ordered) - 1))))
    return ordered[index]


def stratified_binary_association(
    rows: list[dict[str, Any]],
    key: str,
    exposed_value: Any,
    *,
    bootstrap_samples: int = 1000,
) -> dict[str, Any]:
    """Summarize a binary feature with a player-stratified, game-cluster bootstrap."""
    exposed = [row for row in rows if row.get(key) == exposed_value]
    control = [row for row in rows if row.get(key) != exposed_value]
    exposed_illegal = sum(row["first_attempt_illegal"] is True for row in exposed)
    control_illegal = sum(row["first_attempt_illegal"] is True for row in control)
    estimate = mantel_haenszel_odds_ratio(rows, key, exposed_value)

    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_game[str(row["game_id"])].append(row)
    game_ids = sorted(by_game)
    rng = random.Random(f"2026-07-15:{key}:{exposed_value}")
    bootstrapped: list[float] = []
    for _ in range(bootstrap_samples):
        sampled_rows: list[dict[str, Any]] = []
        for game_id in rng.choices(game_ids, k=len(game_ids)):
            sampled_rows.extend(by_game[game_id])
        value = mantel_haenszel_odds_ratio(sampled_rows, key, exposed_value)
        if value is not None and math.isfinite(value):
            bootstrapped.append(value)

    return {
        "feature": key,
        "exposed_value": exposed_value,
        "exposed_attempts": len(exposed),
        "exposed_illegals": exposed_illegal,
        "exposed_illegal_pct": 100.0 * exposed_illegal / len(exposed) if exposed else None,
        "control_attempts": len(control),
        "control_illegals": control_illegal,
        "control_illegal_pct": 100.0 * control_illegal / len(control) if control else None,
        "player_stratified_odds_ratio": estimate,
        "game_cluster_bootstrap_95pct": [
            percentile(bootstrapped, 0.025),
            percentile(bootstrapped, 0.975),
        ],
        "bootstrap_samples_used": len(bootstrapped),
    }


def build_report(
    results: dict[str, dict[str, Any]],
    pgns: dict[str, str],
    *,
    source_results: int,
    player_prefixes: tuple[str, ...],
) -> dict[str, Any]:
    """Build the complete aggregate and sanitized-event report."""
    turns: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    missing_pgns: list[str] = []
    failures: list[dict[str, str]] = []
    for game_id, result in results.items():
        pgn = pgns.get(game_id)
        if not pgn:
            missing_pgns.append(game_id)
            continue
        try:
            game_turns, game_events = analyze_game(
                game_id,
                result,
                pgn,
                player_prefixes,
            )
        except (ValueError, chess.InvalidMoveError) as exc:
            failures.append({"game_id": game_id, "error": str(exc)})
            continue
        turns.extend(game_turns)
        events.extend(game_events)

    first_events = [event for event in events if event["attempt_kind"] == "first_attempt"]
    retry_events = [event for event in events if event["attempt_kind"] == "retry"]
    feature_keys = [
        "player_id",
        "opponent_type",
        "side",
        "ply_phase",
        "history_bucket",
        "material_phase",
        "legal_move_bucket",
        "in_check",
        "castling_rights",
        "promotion_available",
        "last_move_capture",
        "last_move_check",
        "prior_strike",
    ]
    binary_specs = [
        ("side", "white"),
        ("ply_phase", "middlegame_21_60"),
        ("history_bucket", "20_39"),
        ("material_phase", "middlegame_13_25"),
        ("opponent_type", "gpt"),
        ("in_check", True),
        ("castling_rights", True),
        ("promotion_available", True),
        ("last_move_capture", True),
        ("last_move_check", True),
        ("prior_strike", True),
    ]
    sanitized_events = [
        {
            key: value
            for key, value in event.items()
            if key not in {"first_attempt_illegal", "illegal_attempts_on_turn"}
        }
        for event in events
    ]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "pgn-reconstructed-first-attempt-controls-v1",
        "player_prefixes": list(player_prefixes),
        "source_result_documents": source_results,
        "selected_games": len(results),
        "analyzed_games": len(results) - len(missing_pgns) - len(failures),
        "missing_pgns": missing_pgns,
        "analysis_failures": failures,
        "first_attempt_turns": len(turns),
        "first_attempt_illegals": len(first_events),
        "first_attempt_illegal_pct": (
            100.0 * len(first_events) / len(turns) if turns else None
        ),
        "retry_illegal_events": len(retry_events),
        "rates": {key: count_rows(turns, key) for key in feature_keys},
        "joint_rates": {
            "ply_phase__castling_rights": joint_rate_rows(
                turns, "ply_phase", "castling_rights"
            ),
            "ply_phase__last_move_capture": joint_rate_rows(
                turns, "ply_phase", "last_move_capture"
            ),
            "history_bucket__castling_rights": joint_rate_rows(
                turns, "history_bucket", "castling_rights"
            ),
        },
        "rates_by_player": {
            player_id: {
                key: count_rows(
                    [row for row in turns if row["player_id"] == player_id],
                    key,
                )
                for key in feature_keys
            }
            for player_id in sorted({str(row["player_id"]) for row in turns})
        },
        "stratified_binary_associations": [
            stratified_binary_association(turns, key, exposed_value)
            for key, exposed_value in binary_specs
        ],
        "first_attempt_classifications": {
            "primary_class": counter_rows(first_events, "primary_class"),
            "detailed_class": counter_rows(first_events, "detailed_class"),
            "source_piece": counter_rows(first_events, "source_piece"),
            "response_style": counter_rows(first_events, "response_style"),
            "well_formed_uci": counter_rows(first_events, "well_formed_uci"),
            "matches_previous_own_move": counter_rows(
                first_events, "matches_previous_own_move"
            ),
            "legal_on_previous_target_turn": counter_rows(
                first_events, "legal_on_previous_target_turn"
            ),
            "legal_if_ignoring_last_opponent_move": counter_rows(
                first_events, "legal_if_ignoring_last_opponent_move"
            ),
            "stale_board_signal": counter_rows(first_events, "stale_board_signal"),
        },
        "retry_classifications": {
            "primary_class": counter_rows(retry_events, "primary_class"),
            "response_style": counter_rows(retry_events, "response_style"),
        },
        "illegal_events": sanitized_events,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-input", type=Path)
    parser.add_argument("--pgns-input", type=Path)
    parser.add_argument(
        "--player-prefix",
        action="append",
        default=[],
        help="Select player IDs by prefix; defaults to gpt-5.6-",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prefixes = tuple(args.player_prefix or ["gpt-5.6-"])
    results, pgns, source_results = load_games_and_pgns(
        results_input=args.results_input,
        pgns_input=args.pgns_input,
        player_prefixes=prefixes,
    )
    report = build_report(
        results,
        pgns,
        source_results=source_results,
        player_prefixes=prefixes,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Analyzed {report['analyzed_games']}/{report['selected_games']} games: "
        f"{report['first_attempt_illegals']}/{report['first_attempt_turns']} "
        f"first attempts illegal ({report['first_attempt_illegal_pct']:.2f}%)"
    )
    print(f"Retry-illegal events: {report['retry_illegal_events']}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
