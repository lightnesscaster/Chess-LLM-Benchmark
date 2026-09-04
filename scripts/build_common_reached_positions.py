#!/usr/bin/env python3
"""Build a shared-position dataset from exact FENs many models already reached."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def eval_to_cp(info: chess.engine.InfoDict, perspective: chess.Color) -> float:
    score = info.get("score")
    if score is None:
        return 0.0
    pov = score.pov(perspective)
    if pov.is_mate():
        mate = pov.mate()
        if mate is None:
            return 0.0
        if mate > 0:
            return 10000.0 - 10.0 * mate
        return -10000.0 - 10.0 * mate
    cp = pov.score()
    return float(cp if cp is not None else 0.0)


def cache_key(fen: str, depth: int, multipv: int) -> str:
    return hashlib.sha256(f"{depth}:{multipv}:{fen}".encode("utf-8")).hexdigest()


def move_cache_key(fen: str, move_uci: str, depth: int) -> str:
    return hashlib.sha256(f"{depth}:{fen}:{move_uci}".encode("utf-8")).hexdigest()


def analyze_position(
    *,
    fen: str,
    engine: chess.engine.SimpleEngine,
    depth: int,
    multipv: int,
    cache_path: Path,
) -> dict[str, Any]:
    if cache_path.exists():
        cached = load_json(cache_path)
        if (
            cached.get("fen") == fen
            and cached.get("depth") == depth
            and cached.get("multipv") == multipv
        ):
            return cached

    board = chess.Board(fen)
    width = max(1, min(multipv, board.legal_moves.count()))
    info_list = engine.analyse(
        board,
        chess.engine.Limit(depth=depth),
        multipv=width,
    )
    if isinstance(info_list, dict):
        info_list = [info_list]

    best_eval = eval_to_cp(info_list[0], board.turn) if info_list else 0.0
    moves: list[dict[str, Any]] = []
    for info in info_list:
        pv = info.get("pv") or []
        if not pv:
            continue
        move = pv[0]
        if move not in board.legal_moves:
            continue
        move_eval = eval_to_cp(info, board.turn)
        try:
            san = board.san(move)
        except ValueError:
            san = move.uci()
        moves.append(
            {
                "move": move.uci(),
                "san": san,
                "eval": move_eval,
                "delta_cp": max(0.0, best_eval - move_eval),
            }
        )

    analyzed = {
        "fen": fen,
        "depth": depth,
        "multipv": multipv,
        "best_eval": best_eval,
        "moves": moves,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(analyzed))
    return analyzed


def evaluate_move(
    *,
    fen: str,
    move_uci: str,
    engine: chess.engine.SimpleEngine,
    depth: int,
    cache_path: Path,
) -> float:
    if cache_path.exists():
        cached = load_json(cache_path)
        if (
            cached.get("fen") == fen
            and cached.get("move") == move_uci
            and cached.get("depth") == depth
        ):
            return float(cached["eval"])

    board = chess.Board(fen)
    move = chess.Move.from_uci(move_uci)
    board.push(move)
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    eval_after = -eval_to_cp(info, not board.turn)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {"fen": fen, "move": move_uci, "depth": depth, "eval": eval_after}
        )
    )
    return eval_after


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "total_positions": 0,
            "legal_moves": 0,
            "legal_pct": 0.0,
            "best_moves": 0,
            "best_pct": 0.0,
            "avg_cpl": 10000.0,
            "median_cpl": 10000.0,
        }
    cpls = sorted(float(item["cpl"]) for item in results)
    n = len(cpls)
    median = cpls[n // 2] if n % 2 else (cpls[n // 2 - 1] + cpls[n // 2]) / 2.0
    legal = sum(1 for item in results if item.get("is_legal", True))
    best = sum(1 for item in results if item.get("is_best", False))
    return {
        "total_positions": n,
        "legal_moves": legal,
        "legal_pct": 100.0 * legal / n,
        "best_moves": best,
        "best_pct": 100.0 * best / n,
        "avg_cpl": sum(cpls) / n,
        "median_cpl": median,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games-dir", type=Path, default=Path("position_benchmark/games"))
    parser.add_argument("--results-json", type=Path, default=Path("position_benchmark/games/_results.json"))
    parser.add_argument("--output-positions", type=Path, default=Path("position_benchmark/common_reached_positions.json"))
    parser.add_argument("--output-results", type=Path, default=Path("position_benchmark/common_reached_results.json"))
    parser.add_argument("--stockfish-path", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hash-mb", type=int, default=512)
    parser.add_argument("--max-plies", type=int, default=20)
    parser.add_argument("--min-model-coverage", type=int, default=10)
    parser.add_argument("--max-positions", type=int, default=40)
    parser.add_argument("--min-occurrences", type=int, default=10)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/common_reached_cache"))
    args = parser.parse_args()

    results = load_json(args.results_json)
    entries: dict[str, dict[str, Any]] = {}

    parsed = 0
    for game_id, meta in results.items():
        pgn_path = args.games_dir / f"{game_id}.pgn"
        if not pgn_path.exists():
            continue
        game = chess.pgn.read_game(io.StringIO(pgn_path.read_text()))
        if game is None:
            continue

        board = game.board()
        san_history: list[str] = []
        white_id = meta.get("white_id")
        black_id = meta.get("black_id")

        for ply, node in enumerate(game.mainline(), start=1):
            if ply > args.max_plies:
                break
            fen = board.fen()
            player_id = white_id if board.turn == chess.WHITE else black_id
            move = node.move
            try:
                san = board.san(move)
            except ValueError:
                san = move.uci()

            entry = entries.setdefault(
                fen,
                {
                    "fen": fen,
                    "move_history": list(san_history),
                    "side_to_move": "white" if board.turn == chess.WHITE else "black",
                    "moves_by_model": {},
                    "occurrences": 0,
                    "ply": ply,
                },
            )
            entry["occurrences"] += 1
            if player_id and player_id not in entry["moves_by_model"]:
                entry["moves_by_model"][player_id] = move.uci()

            san_history.append(san)
            board.push(move)

        parsed += 1
        if parsed % 1000 == 0:
            print(f"Parsed {parsed} games", flush=True)

    candidates = [
        entry
        for entry in entries.values()
        if len(entry["moves_by_model"]) >= args.min_model_coverage
        and entry["occurrences"] >= args.min_occurrences
    ]
    candidates.sort(
        key=lambda entry: (
            len(entry["moves_by_model"]),
            entry["occurrences"],
            -entry["ply"],
        ),
        reverse=True,
    )
    selected = candidates[: args.max_positions]

    print(
        f"Selected {len(selected)} positions from {len(candidates)} candidates "
        f"(coverage>={args.min_model_coverage}, occurrences>={args.min_occurrences})"
    )

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    try:
        try:
            engine.configure({"Threads": args.threads, "Hash": args.hash_mb})
        except chess.engine.EngineError:
            pass

        positions: list[dict[str, Any]] = []
        results_by_model: dict[str, dict[str, Any]] = defaultdict(lambda: {"results": []})

        pos_cache_dir = args.cache_dir / f"depth{args.depth}_multipv{args.multipv}"
        move_cache_dir = args.cache_dir / f"moveeval_depth{args.depth}"

        for pos_idx, entry in enumerate(selected):
            analyzed = analyze_position(
                fen=entry["fen"],
                engine=engine,
                depth=args.depth,
                multipv=args.multipv,
                cache_path=pos_cache_dir / f"{cache_key(entry['fen'], args.depth, args.multipv)}.json",
            )
            position = {
                "fen": entry["fen"],
                "move_history": entry["move_history"],
                "side_to_move": entry["side_to_move"],
                "type": "common_reached",
                "bucket": "common_reached",
                "eval_before": analyzed["best_eval"],
                "best_move": analyzed["moves"][0]["move"] if analyzed["moves"] else "",
                "best_move_san": analyzed["moves"][0]["san"] if analyzed["moves"] else "",
                "multipv": analyzed["moves"],
                "common_model_coverage": len(entry["moves_by_model"]),
                "common_occurrences": entry["occurrences"],
                "common_ply": entry["ply"],
            }
            positions.append(position)

            board = chess.Board(entry["fen"])
            by_move = {item["move"]: item for item in analyzed["moves"]}
            for player_id, model_move in sorted(entry["moves_by_model"].items()):
                try:
                    move = chess.Move.from_uci(model_move)
                    model_move_san = board.san(move)
                    is_legal = move in board.legal_moves
                except (ValueError, chess.InvalidMoveError):
                    model_move_san = ""
                    is_legal = False

                if is_legal and model_move in by_move:
                    eval_model = float(by_move[model_move]["eval"])
                elif is_legal:
                    eval_model = evaluate_move(
                        fen=entry["fen"],
                        move_uci=model_move,
                        engine=engine,
                        depth=args.depth,
                        cache_path=move_cache_dir / f"{move_cache_key(entry['fen'], model_move, args.depth)}.json",
                    )
                else:
                    eval_model = -5000.0
                cpl = max(0.0, float(analyzed["best_eval"]) - eval_model)

                results_by_model[player_id]["results"].append(
                    {
                        "position_idx": pos_idx,
                        "fen": entry["fen"],
                        "model_move": model_move,
                        "model_move_san": model_move_san,
                        "best_move": position["best_move"],
                        "best_move_san": position["best_move_san"],
                        "blunder_move": "",
                        "cpl": cpl,
                        "is_legal": is_legal,
                        "is_best": model_move == position["best_move"],
                        "avoided_blunder": True,
                        "eval_model": eval_model,
                        "eval_best": float(analyzed["best_eval"]),
                        "eval_before": float(analyzed["best_eval"]),
                    }
                )

            if (pos_idx + 1) % 10 == 0 or pos_idx + 1 == len(selected):
                print(f"Analyzed {pos_idx + 1}/{len(selected)} positions", flush=True)
    finally:
        engine.quit()

    for player_id, model_data in results_by_model.items():
        model_data["summary"] = build_summary(model_data["results"])
        model_data["summary"]["player_id"] = player_id

    positions_payload = {
        "metadata": {
            "description": "Exact FENs already reached by many models in local Firebase game snapshot",
            "source": str(args.results_json),
            "max_plies": args.max_plies,
            "min_model_coverage": args.min_model_coverage,
            "max_positions": args.max_positions,
            "depth": args.depth,
            "multipv": args.multipv,
            "position_count": len(positions),
        },
        "positions": positions,
    }

    args.output_positions.write_text(json.dumps(positions_payload, indent=2))
    args.output_results.write_text(json.dumps(dict(results_by_model), indent=2))


if __name__ == "__main__":
    main()
