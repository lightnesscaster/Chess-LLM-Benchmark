#!/usr/bin/env python3
"""Build a PGN-mined Regan-lite shared position set.

The output is compatible with position_benchmark/run_benchmark.py and includes
extra MultiPV metadata for later Regan-style scoring. Positions are sampled
from real benchmark PGNs, evaluated at Stockfish depth >= 16, then balanced
across natural chess buckets instead of being dominated by early-game equality.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine
import chess.pgn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_BUCKET_TARGETS = {
    "opening": 60,
    "quiet_equal": 70,
    "tactical_equal": 50,
    "advantage_conversion": 60,
    "defense": 60,
    "endgame": 50,
    "legal_stress": 30,
}


@dataclass(frozen=True)
class Candidate:
    game_id: str
    pgn_path: Path
    ply_before: int
    fullmove: int
    side_to_move: str
    source_player_id: str
    opponent_id: str
    fen: str
    move_history: tuple[str, ...]
    move_history_san: tuple[str, ...]
    played_move: str
    played_move_san: str
    pieces: int
    legal_moves: int
    in_check: bool


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
        return -10000.0 - 10.0 * abs(mate)
    cp = pov.score()
    return float(cp if cp is not None else 0.0)


def stable_fen_key(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def cache_key(fen: str, depth: int, multipv: int) -> str:
    digest = hashlib.sha256(f"{depth}:{multipv}:{fen}".encode("utf-8")).hexdigest()
    return digest[:24]


def material_piece_count(board: chess.Board) -> int:
    return len(board.piece_map())


def is_interesting_ply(board: chess.Board, ply_before: int) -> bool:
    if board.is_game_over(claim_draw=True):
        return False
    if board.legal_moves.count() < 2:
        return False
    if ply_before < 6:
        return False
    return True


def iter_candidates_from_pgn(path: Path) -> list[Candidate]:
    pgn = path.read_text(errors="replace")
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return []

    white_id = game.headers.get("White", "unknown")
    black_id = game.headers.get("Black", "unknown")
    game_id = path.stem
    board = game.board()
    move_history: list[str] = []
    move_history_san: list[str] = []
    out: list[Candidate] = []

    for ply_before, node in enumerate(game.mainline()):
        if is_interesting_ply(board, ply_before):
            side = board.turn
            move = node.move
            try:
                san = board.san(move)
            except ValueError:
                san = ""
            out.append(
                Candidate(
                    game_id=game_id,
                    pgn_path=path,
                    ply_before=ply_before,
                    fullmove=board.fullmove_number,
                    side_to_move="white" if side == chess.WHITE else "black",
                    source_player_id=white_id if side == chess.WHITE else black_id,
                    opponent_id=black_id if side == chess.WHITE else white_id,
                    fen=board.fen(),
                    move_history=tuple(move_history),
                    move_history_san=tuple(move_history_san),
                    played_move=move.uci(),
                    played_move_san=san,
                    pieces=material_piece_count(board),
                    legal_moves=board.legal_moves.count(),
                    in_check=board.is_check(),
                )
            )
        move = node.move
        try:
            move_history_san.append(board.san(move))
        except ValueError:
            move_history_san.append(move.uci())
        move_history.append(move.uci())
        board.push(move)

    return out


def collect_candidates(
    pgn_dirs: list[Path],
    max_games: int | None,
    seed: int,
) -> list[Candidate]:
    pgn_paths: list[Path] = []
    seen_ids: set[str] = set()
    for pgn_dir in pgn_dirs:
        if not pgn_dir.exists():
            continue
        for path in sorted(pgn_dir.glob("*.pgn")):
            if path.stem in seen_ids:
                continue
            seen_ids.add(path.stem)
            pgn_paths.append(path)

    rng = random.Random(seed)
    rng.shuffle(pgn_paths)
    if max_games is not None:
        pgn_paths = pgn_paths[:max_games]

    candidates: list[Candidate] = []
    seen_positions: set[str] = set()
    for path in pgn_paths:
        for candidate in iter_candidates_from_pgn(path):
            key = stable_fen_key(chess.Board(candidate.fen))
            if key in seen_positions:
                continue
            seen_positions.add(key)
            candidates.append(candidate)
    rng.shuffle(candidates)
    return candidates


def analyze_candidate(
    candidate: Candidate,
    engine: chess.engine.SimpleEngine,
    depth: int,
    multipv: int,
    cache_dir: Path,
) -> dict[str, Any] | None:
    board = chess.Board(candidate.fen)
    key = cache_key(candidate.fen, depth, multipv)
    cache_path = cache_dir / f"{key}.json"
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if (
                cached.get("fen") == candidate.fen
                and cached.get("depth") == depth
                and cached.get("multipv") == multipv
            ):
                return cached
        except json.JSONDecodeError:
            pass

    try:
        infos = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=min(multipv, board.legal_moves.count()),
        )
    except chess.engine.EngineTerminatedError:
        raise
    except chess.engine.EngineError:
        return None

    if isinstance(infos, dict):
        infos = [infos]
    if not infos:
        return None

    moves: list[dict[str, Any]] = []
    best_eval: float | None = None
    for info in infos:
        pv = info.get("pv") or []
        if not pv:
            continue
        move = pv[0]
        if move not in board.legal_moves:
            continue
        move_eval = eval_to_cp(info, board.turn)
        if best_eval is None:
            best_eval = move_eval
        try:
            san = board.san(move)
        except ValueError:
            san = move.uci()
        moves.append(
            {
                "move": move.uci(),
                "san": san,
                "eval": move_eval,
                "delta_cp": max(0.0, float(best_eval - move_eval)),
            }
        )

    if not moves or best_eval is None:
        return None

    analyzed = {
        "fen": candidate.fen,
        "depth": depth,
        "multipv": multipv,
        "eval_before": best_eval,
        "multipv_moves": moves,
        "best_move": moves[0]["move"],
        "best_move_san": moves[0]["san"],
        "second_gap_cp": moves[1]["delta_cp"] if len(moves) > 1 else 10000.0,
        "near_best_25": sum(1 for item in moves if item["delta_cp"] <= 25),
        "near_best_50": sum(1 for item in moves if item["delta_cp"] <= 50),
        "near_best_100": sum(1 for item in moves if item["delta_cp"] <= 100),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(analyzed))
    return analyzed


def matching_buckets(candidate: Candidate, analyzed: dict[str, Any]) -> list[str]:
    eval_before = float(analyzed["eval_before"])
    second_gap = float(analyzed["second_gap_cp"])
    pieces = candidate.pieces
    fullmove = candidate.fullmove
    legal_moves = candidate.legal_moves
    buckets: list[str] = []

    if 4 <= fullmove <= 12 and abs(eval_before) <= 250 and pieces >= 22:
        buckets.append("opening")
    if fullmove >= 10 and pieces > 12 and abs(eval_before) <= 160 and second_gap <= 80:
        buckets.append("quiet_equal")
    if fullmove >= 8 and pieces > 10 and abs(eval_before) <= 300 and second_gap >= 120:
        buckets.append("tactical_equal")
    if 180 <= eval_before <= 900 and fullmove >= 8:
        buckets.append("advantage_conversion")
    if -900 <= eval_before <= -180 and fullmove >= 8:
        buckets.append("defense")
    if (pieces <= 12 or fullmove >= 35) and abs(eval_before) <= 900:
        buckets.append("endgame")
    if (legal_moves <= 8 or candidate.in_check or pieces <= 8) and abs(eval_before) <= 1200:
        buckets.append("legal_stress")
    return buckets


def choose_bucket(
    matches: list[str],
    selected_counts: dict[str, int],
    targets: dict[str, int],
) -> str | None:
    available = [bucket for bucket in matches if selected_counts.get(bucket, 0) < targets.get(bucket, 0)]
    if not available:
        return None
    return max(
        available,
        key=lambda bucket: (targets[bucket] - selected_counts.get(bucket, 0), targets[bucket]),
    )


def build_position(candidate: Candidate, analyzed: dict[str, Any], bucket: str) -> dict[str, Any]:
    board = chess.Board(candidate.fen)
    played_cpl = None
    played_in_multipv = next(
        (item for item in analyzed["multipv_moves"] if item["move"] == candidate.played_move),
        None,
    )
    if played_in_multipv is not None:
        played_cpl = played_in_multipv["delta_cp"]

    return {
        "fen": candidate.fen,
        "move_history": list(candidate.move_history),
        "move_history_san": list(candidate.move_history_san),
        "side_to_move": candidate.side_to_move,
        "type": "equal",
        "regan_bucket": bucket,
        "best_move": analyzed["best_move"],
        "best_move_san": analyzed["best_move_san"],
        "eval_before": round(float(analyzed["eval_before"]), 2),
        "multipv": analyzed["multipv_moves"],
        "second_gap_cp": round(float(analyzed["second_gap_cp"]), 2),
        "near_best_25": int(analyzed["near_best_25"]),
        "near_best_50": int(analyzed["near_best_50"]),
        "near_best_100": int(analyzed["near_best_100"]),
        "legal_moves": candidate.legal_moves,
        "pieces": candidate.pieces,
        "in_check": candidate.in_check,
        "game_id": candidate.game_id,
        "ply": candidate.ply_before,
        "move_number": candidate.fullmove,
        "source_player_id": candidate.source_player_id,
        "opponent_id": candidate.opponent_id,
        "source_played_move": candidate.played_move,
        "source_played_move_san": candidate.played_move_san,
        "source_played_cpl_if_multipv": played_cpl,
        "fen_key": stable_fen_key(board),
    }


def parse_targets(value: str | None) -> dict[str, int]:
    if not value:
        return dict(DEFAULT_BUCKET_TARGETS)
    targets: dict[str, int] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        name, raw_count = part.split("=", 1)
        targets[name.strip()] = int(raw_count)
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pgn-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory of PGN files. Defaults to position_benchmark/games and data/games.",
    )
    parser.add_argument("--output", type=Path, default=Path("position_benchmark/regan_lite_positions.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/regan_lite_multipv_cache"))
    parser.add_argument("--stockfish-path", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=256)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-games", type=int)
    parser.add_argument("--max-analyzed", type=int, default=6000)
    parser.add_argument(
        "--targets",
        help=(
            "Comma-separated bucket targets, e.g. "
            "opening=40,quiet_equal=60,tactical_equal=40"
        ),
    )
    args = parser.parse_args()

    if args.depth < 16:
        raise SystemExit("Use --depth >= 16 for this benchmark-building pass.")
    if args.multipv < 2:
        raise SystemExit("Use --multipv >= 2 so position ambiguity can be measured.")

    pgn_dirs = args.pgn_dir or [Path("position_benchmark/games"), Path("data/games")]
    targets = parse_targets(args.targets)

    print("Collecting PGN candidates...")
    candidates = collect_candidates(pgn_dirs, args.max_games, args.seed)
    print(f"Candidates: {len(candidates)} from {', '.join(str(p) for p in pgn_dirs)}")
    print(f"Targets: {targets}")

    cache_key_dir = args.cache_dir / f"depth{args.depth}_multipv{args.multipv}"
    selected: list[dict[str, Any]] = []
    selected_counts = {bucket: 0 for bucket in targets}
    analyzed_count = 0
    cache_hits = 0
    skipped = 0

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    try:
        try:
            engine.configure({"Threads": args.threads, "Hash": args.hash_mb})
        except chess.engine.EngineError:
            pass

        for candidate in candidates:
            if all(selected_counts[bucket] >= targets[bucket] for bucket in targets):
                break
            if analyzed_count >= args.max_analyzed:
                break

            cache_path = cache_key_dir / f"{cache_key(candidate.fen, args.depth, args.multipv)}.json"
            was_cached = cache_path.exists()
            analyzed = analyze_candidate(
                candidate,
                engine,
                args.depth,
                args.multipv,
                cache_key_dir,
            )
            analyzed_count += 1
            if was_cached:
                cache_hits += 1
            if analyzed is None:
                skipped += 1
                continue

            bucket = choose_bucket(matching_buckets(candidate, analyzed), selected_counts, targets)
            if bucket is None:
                skipped += 1
                continue

            selected_counts[bucket] += 1
            selected.append(build_position(candidate, analyzed, bucket))

            if analyzed_count % 100 == 0 or len(selected) % 25 == 0:
                print(
                    f"Analyzed={analyzed_count} selected={len(selected)} "
                    f"cache_hits={cache_hits} skipped={skipped} counts={selected_counts}",
                    flush=True,
                )
    finally:
        engine.quit()

    selected.sort(key=lambda item: (item["regan_bucket"], item["game_id"], item["ply"]))
    for idx, item in enumerate(selected):
        item["position_id"] = f"regan-lite-{idx:04d}"

    metadata = {
        "description": "PGN-mined natural positions with Stockfish MultiPV metadata for Regan-lite scoring",
        "depth": args.depth,
        "multipv": args.multipv,
        "targets": targets,
        "selected_counts": selected_counts,
        "positions": len(selected),
        "analyzed_candidates": analyzed_count,
        "cache_hits": cache_hits,
        "skipped_candidates": skipped,
        "pgn_dirs": [str(path) for path in pgn_dirs],
        "seed": args.seed,
    }
    output = {"metadata": metadata, "positions": selected}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))

    print()
    print(f"Wrote {len(selected)} positions to {args.output}")
    print(f"Bucket counts: {selected_counts}")
    print(f"Analyzed candidates: {analyzed_count}, cache_hits={cache_hits}, skipped={skipped}")


if __name__ == "__main__":
    main()
