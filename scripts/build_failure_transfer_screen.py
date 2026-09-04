#!/usr/bin/env python3
"""Build a leakage-safe cross-model screen from saved live-game failure states."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any

import chess
import chess.engine
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from position_benchmark.run_benchmark import eval_to_cp, replay_position_board  # noqa: E402
from scripts.analyze_game_illegal_moves import (  # noqa: E402
    last_move_features,
    load_games_and_pgns,
    material_phase,
    ply_phase,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT = (
    ROOT
    / "position_benchmark/validation/2026-07-15-gpt56-illegal-forensics.json"
)
DEFAULT_DIR = ROOT / "position_benchmark/candidates/failure_transfer_screen_v1"
DEFAULT_MATRIX = DEFAULT_DIR / "matrix.json"
TARGET_PLAYERS = {
    "luna": "gpt-5.6-luna (medium)",
    "terra": "gpt-5.6-terra (low)",
    "sol": "gpt-5.6-sol (high)",
}
CLASS_ORDER = [
    "movement_rule_or_blocked",
    "source_empty",
    "king_safety",
    "destination_own_piece",
    "wrong_side_piece",
]


def base_model(player_id: str) -> str:
    for name in TARGET_PLAYERS:
        if player_id.startswith(f"gpt-5.6-{name}"):
            return name
    raise ValueError(f"Unexpected GPT-5.6 player ID: {player_id}")


def stable_key(seed: int, *parts: Any) -> str:
    text = ":".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(text.encode()).hexdigest()


def fen_key(fen: str) -> str:
    return " ".join(fen.split()[:4])


def read_game(pgn_text: str) -> tuple[chess.pgn.Game, list[chess.Move]]:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        raise ValueError("Could not parse PGN")
    return game, list(game.mainline_moves())


def reconstruct_state(pgn_text: str, prospective_ply: int) -> dict[str, Any]:
    game, moves = read_game(pgn_text)
    board = game.board()
    history: list[str] = []
    for move in moves[: prospective_ply - 1]:
        history.append(move.uci())
        board.push(move)
    captured, checked = last_move_features(board)
    return {
        "fen": board.fen(),
        "move_history": history,
        "side": "white" if board.turn == chess.WHITE else "black",
        "ply_number": prospective_ply,
        "history_plies": len(history),
        "ply_phase": ply_phase(prospective_ply),
        "material_phase": material_phase(len(board.piece_map())),
        "piece_count": len(board.piece_map()),
        "legal_move_count": board.legal_moves.count(),
        "castling_rights": board.has_castling_rights(board.turn),
        "last_move_capture": captured,
        "last_move_check": checked,
        "in_check": board.is_check(),
    }


def choose_failures(
    events: list[dict[str, Any]],
    *,
    per_base_model: int,
    seed: int,
) -> list[dict[str, Any]]:
    eligible = [
        event
        for event in events
        if event.get("attempt_kind") == "first_attempt"
        and event.get("well_formed_uci") is True
        and event.get("primary_class") in CLASS_ORDER
    ]
    selected: list[dict[str, Any]] = []
    used_fens: set[str] = set()
    for family in TARGET_PLAYERS:
        family_events = [
            event for event in eligible if base_model(str(event["player_id"])) == family
        ]
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in family_events:
            buckets[str(event["primary_class"])].append(event)
        for bucket in buckets.values():
            bucket.sort(
                key=lambda event: stable_key(
                    seed,
                    family,
                    event["game_id"],
                    event["ply_number"],
                    event["fen"],
                )
            )

        family_selected: list[dict[str, Any]] = []
        offset = 0
        while len(family_selected) < per_base_model:
            added = False
            for failure_class in CLASS_ORDER:
                candidates = buckets.get(failure_class, [])
                while offset < len(candidates):
                    candidate = candidates[offset]
                    key = fen_key(str(candidate["fen"]))
                    if key not in used_fens:
                        family_selected.append(candidate)
                        used_fens.add(key)
                        added = True
                        break
                    offset += 1
                if len(family_selected) == per_base_model:
                    break
            if not added:
                remaining = [
                    event
                    for event in family_events
                    if fen_key(str(event["fen"])) not in used_fens
                ]
                remaining.sort(
                    key=lambda event: stable_key(
                        seed,
                        "remainder",
                        event["game_id"],
                        event["ply_number"],
                    )
                )
                if not remaining:
                    break
                family_selected.append(remaining[0])
                used_fens.add(fen_key(str(remaining[0]["fen"])))
            offset += 1
        if len(family_selected) != per_base_model:
            raise ValueError(
                f"Only selected {len(family_selected)}/{per_base_model} {family} failures"
            )
        selected.extend(family_selected)
    return selected


def legal_control_states(
    results: dict[str, dict[str, Any]],
    pgns: dict[str, str],
    illegal_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    illegal_turns = {
        (str(event["game_id"]), str(event["player_id"]), int(event["ply_number"]))
        for event in illegal_events
        if event.get("attempt_kind") == "first_attempt"
    }
    controls: list[dict[str, Any]] = []
    for game_id, result in results.items():
        pgn = pgns.get(game_id)
        if not pgn:
            continue
        game, moves = read_game(pgn)
        board = game.board()
        history: list[str] = []
        sides = {
            "white": str(result.get("white_id") or ""),
            "black": str(result.get("black_id") or ""),
        }
        for ply_number, move in enumerate(moves, start=1):
            side = "white" if board.turn == chess.WHITE else "black"
            player_id = sides[side]
            if player_id.startswith("gpt-5.6-") and (
                game_id,
                player_id,
                ply_number,
            ) not in illegal_turns:
                captured, checked = last_move_features(board)
                controls.append(
                    {
                        "game_id": game_id,
                        "player_id": player_id,
                        "base_model": base_model(player_id),
                        "fen": board.fen(),
                        "move_history": list(history),
                        "side": side,
                        "ply_number": ply_number,
                        "history_plies": len(history),
                        "ply_phase": ply_phase(ply_number),
                        "material_phase": material_phase(len(board.piece_map())),
                        "piece_count": len(board.piece_map()),
                        "legal_move_count": board.legal_moves.count(),
                        "castling_rights": board.has_castling_rights(board.turn),
                        "last_move_capture": captured,
                        "last_move_check": checked,
                        "in_check": board.is_check(),
                    }
                )
            history.append(move.uci())
            board.push(move)
    return controls


def control_distance(failure: dict[str, Any], control: dict[str, Any]) -> float:
    score = 0.0
    score += 100.0 * (failure["side"] != control["side"])
    score += 40.0 * (failure["ply_phase"] != control["ply_phase"])
    score += 30.0 * (failure["castling_rights"] != control["castling_rights"])
    score += 20.0 * (failure["last_move_capture"] != control["last_move_capture"])
    score += 20.0 * (failure["in_check"] != control["in_check"])
    score += 10.0 * (failure["material_phase"] != control["material_phase"])
    score += 2.0 * abs(failure["piece_count"] - control["piece_count"])
    score += 0.5 * abs(failure["legal_move_count"] - control["legal_move_count"])
    score += 0.2 * abs(failure["history_plies"] - control["history_plies"])
    return score


def match_controls(
    failures: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_fens = {fen_key(str(failure["fen"])) for failure in failures}
    for failure in failures:
        candidates = [
            control
            for control in controls
            if control["player_id"] == failure["player_id"]
            and control["game_id"] != failure["game_id"]
            and fen_key(str(control["fen"])) not in used_fens
        ]
        candidates.sort(
            key=lambda control: (
                control_distance(failure, control),
                stable_key(
                    seed,
                    "control",
                    failure["game_id"],
                    failure["ply_number"],
                    control["game_id"],
                    control["ply_number"],
                ),
            )
        )
        if not candidates:
            raise ValueError(f"No matched control for {failure['candidate_id']}")
        control = dict(candidates[0])
        control["candidate_kind"] = "matched_control"
        control["matched_failure_id"] = failure["candidate_id"]
        control["match_distance"] = control_distance(failure, control)
        control["candidate_id"] = failure["candidate_id"].replace(
            "failure-", "control-"
        )
        used_fens.add(fen_key(str(control["fen"])))
        selected.append(control)
    return selected


def score_position(
    candidate: dict[str, Any],
    stockfish: chess.engine.SimpleEngine,
    *,
    depth: int,
) -> dict[str, Any]:
    board = chess.Board(str(candidate["fen"]))
    info = stockfish.analyse(board, chess.engine.Limit(depth=depth))
    pv = info.get("pv") or []
    if not pv:
        raise ValueError(f"No Stockfish PV for {candidate['candidate_id']}")
    best_move = pv[0]
    output = dict(candidate)
    output.update(
        {
            "position_id": candidate["candidate_id"],
            "panel": "failure-transfer-screen-v1",
            "type": "transfer-screen",
            "side_to_move": candidate["side"],
            "best_move": best_move.uci(),
            "best_move_san": board.san(best_move),
            "eval_before": eval_to_cp(info, board.turn),
            "blunder_move": "",
            "blunder_move_san": "",
            "stockfish_setup_depth": depth,
        }
    )
    replay = replay_position_board(output)
    if fen_key(replay.fen()) != fen_key(str(output["fen"])):
        raise ValueError(f"History replay mismatch for {candidate['candidate_id']}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--matrix-output", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--per-base-model", type=int, default=4)
    parser.add_argument("--seed", type=int, default=5615)
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--stockfish-path", default="stockfish")
    args = parser.parse_args()

    audit = json.loads(args.audit.read_text())
    events = audit["illegal_events"]
    results, pgns, source_results = load_games_and_pgns(
        results_input=None,
        pgns_input=None,
        player_prefixes=("gpt-5.6-",),
    )
    failures = choose_failures(
        events,
        per_base_model=args.per_base_model,
        seed=args.seed,
    )
    reconstructed: list[dict[str, Any]] = []
    family_counts: dict[str, int] = defaultdict(int)
    for event in failures:
        state = reconstruct_state(pgns[str(event["game_id"])], int(event["ply_number"]))
        if fen_key(state["fen"]) != fen_key(str(event["fen"])):
            raise ValueError(
                f"Audit/PGN mismatch for {event['game_id']} ply {event['ply_number']}"
            )
        family = base_model(str(event["player_id"]))
        family_counts[family] += 1
        reconstructed.append(
            {
                **state,
                "candidate_id": (
                    f"failure-transfer-{family}-{family_counts[family]:03d}"
                ),
                "candidate_kind": "source_failure",
                "game_id": event["game_id"],
                "player_id": event["player_id"],
                "base_model": family,
                "source_failure_primary_class": event["primary_class"],
                "source_failure_detailed_class": event["detailed_class"],
                "source_illegal_move": event["parsed_move"],
                "source_stale_board_signal": event["stale_board_signal"],
            }
        )

    controls = legal_control_states(results, pgns, events)
    matched = match_controls(reconstructed, controls, seed=args.seed)
    all_candidates = reconstructed + matched
    stockfish = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    try:
        scored = [
            score_position(candidate, stockfish, depth=args.depth)
            for candidate in all_candidates
        ]
    finally:
        stockfish.quit()
    by_id = {candidate["candidate_id"]: candidate for candidate in scored}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_files: dict[str, str] = {}
    test_matrix: dict[str, list[str]] = {}
    for target_family, target_player in TARGET_PLAYERS.items():
        target_failures = [
            failure
            for failure in reconstructed
            if failure["base_model"] != target_family
        ]
        position_ids: list[str] = []
        for failure in target_failures:
            position_ids.extend(
                [
                    failure["candidate_id"],
                    failure["candidate_id"].replace("failure-", "control-"),
                ]
            )
        positions = []
        for panel_index, position_id in enumerate(position_ids):
            position = dict(by_id[position_id])
            position["panel_index"] = panel_index
            positions.append(position)
        target_path = args.output_dir / f"{target_family}.json"
        target_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "panel": "failure-transfer-screen-v1",
                        "status": "research-screen",
                        "production_effect": "none",
                        "selection_policy": "leave-one-base-model-out-v1",
                        "target_player": target_player,
                        "excluded_source_base_model": target_family,
                        "position_count": len(positions),
                        "source_failure_count": len(target_failures),
                        "matched_control_count": len(target_failures),
                        "stockfish_setup_depth": args.depth,
                    },
                    "positions": positions,
                },
                indent=2,
            )
            + "\n"
        )
        target_files[target_family] = str(target_path.relative_to(ROOT))
        test_matrix[target_player] = position_ids

    matrix = {
        "metadata": {
            "screen_version": "failure-transfer-screen-v1",
            "status": "frozen-before-target-calls",
            "production_effect": "none",
            "created_from_audit": str(args.audit.relative_to(ROOT)),
            "source_result_documents": source_results,
            "selected_failure_count": len(reconstructed),
            "matched_control_count": len(matched),
            "per_source_base_model": args.per_base_model,
            "seed": args.seed,
            "selection_eligibility": (
                "first-attempt, well-formed UCI, chess-legality failure"
            ),
            "target_policy": "exclude-entire-source-base-model-line-v1",
            "target_files": target_files,
            "planned_first_attempt_calls": sum(len(ids) for ids in test_matrix.values()),
        },
        "selected_failures": reconstructed,
        "matched_controls": matched,
        "test_matrix": test_matrix,
    }
    args.matrix_output.parent.mkdir(parents=True, exist_ok=True)
    args.matrix_output.write_text(json.dumps(matrix, indent=2) + "\n")
    print(
        f"Saved {len(reconstructed)} failures, {len(matched)} controls, and "
        f"{matrix['metadata']['planned_first_attempt_calls']} planned calls to "
        f"{args.matrix_output}"
    )


if __name__ == "__main__":
    main()
