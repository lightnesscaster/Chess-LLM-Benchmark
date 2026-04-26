#!/usr/bin/env python3
"""Add Stockfish MultiPV metadata to an existing position set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import chess
import chess.engine


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
    digest = hashlib.sha256(f"{depth}:{multipv}:{fen}".encode("utf-8")).hexdigest()
    return digest


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stockfish-path", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--hash-mb", type=int, default=512)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/position_multipv_cache"),
    )
    args = parser.parse_args()

    data = load_json(args.input)
    positions = data["positions"] if isinstance(data, dict) else data
    cache_dir = args.cache_dir / f"depth{args.depth}_multipv{args.multipv}"

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    try:
        try:
            engine.configure({"Threads": args.threads, "Hash": args.hash_mb})
        except chess.engine.EngineError:
            pass

        for idx, position in enumerate(positions, start=1):
            fen = position["fen"]
            analyzed = analyze_position(
                fen=fen,
                engine=engine,
                depth=args.depth,
                multipv=args.multipv,
                cache_path=cache_dir / f"{cache_key(fen, args.depth, args.multipv)}.json",
            )
            position["multipv"] = analyzed["moves"]
            position["multipv_depth"] = args.depth
            position["multipv_width"] = args.multipv
            if idx % 10 == 0 or idx == len(positions):
                print(f"Enriched {idx}/{len(positions)} positions", flush=True)
    finally:
        engine.quit()

    if isinstance(data, dict):
        metadata = data.setdefault("metadata", {})
        metadata["multipv_depth"] = args.depth
        metadata["multipv_width"] = args.multipv
        args.output.write_text(json.dumps(data, indent=2))
    else:
        args.output.write_text(json.dumps(positions, indent=2))


if __name__ == "__main__":
    main()
