#!/usr/bin/env python3
"""Build a small 2-step non-opening sequence screening set from raw PGNs.

Each selected sequence contributes two ordinary benchmark positions that can be
run through the existing benchmark runner. The positions are linked with
sequence metadata so later analysis can aggregate them as a micro-sequence
instead of as isolated one-shot items.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_regan_lite_positions import (  # type: ignore
    Candidate,
    analyze_candidate,
    build_position,
    iter_candidates_from_pgn,
    matching_buckets,
)


DEFAULT_SEQUENCE_TARGETS = {
    "quiet_quiet": 3,
    "quiet_tactical": 4,
    "tactical_tactical": 2,
    "tactical_defense": 4,
    "quiet_defense": 3,
}
ALLOWED_BUCKETS = {"quiet_equal", "tactical_equal", "defense"}


def parse_targets(value: str | None) -> dict[str, int]:
    if not value:
        return dict(DEFAULT_SEQUENCE_TARGETS)
    targets: dict[str, int] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        name, raw_count = part.split("=", 1)
        targets[name.strip()] = int(raw_count)
    return targets


def model_family(player_id: str) -> str:
    player = player_id.lower()
    for prefix in (
        "gemini-3.1",
        "gemini-3",
        "gemini-2.5",
        "gemini-2.0",
        "gpt-5.2",
        "gpt-5.1",
        "gpt-5",
        "grok-4.1",
        "grok-4",
        "grok-3",
        "deepseek-v3.2",
        "deepseek-v3.1",
        "deepseek-chat",
        "claude",
        "llama-4",
        "maia",
        "random",
        "survival",
        "eubos",
    ):
        if player.startswith(prefix):
            return prefix
    return player.split()[0].split("-")[0]


def list_pgn_paths(pgn_dirs: list[Path], seed: int, max_games: int | None) -> list[Path]:
    paths: list[Path] = []
    seen_ids: set[str] = set()
    for pgn_dir in pgn_dirs:
        if not pgn_dir.exists():
            continue
        for path in sorted(pgn_dir.glob("*.pgn")):
            if path.stem in seen_ids:
                continue
            seen_ids.add(path.stem)
            paths.append(path)
    rng = random.Random(seed)
    rng.shuffle(paths)
    if max_games is not None:
        paths = paths[:max_games]
    return paths


def choose_primary_bucket(candidate: dict[str, Any]) -> str | None:
    matches = set(candidate["matches"])
    eval_before = float(candidate["eval_before"])
    second_gap = float(candidate["second_gap_cp"])
    near_best_25 = int(candidate["near_best_25"])
    legal_moves = int(candidate["legal_moves"])
    pieces = int(candidate["pieces"])
    move_number = int(candidate["move_number"])

    if "quiet_equal" in matches:
        if (
            move_number >= 10
            and pieces > 14
            and abs(eval_before) <= 130
            and 18 <= second_gap <= 85
            and near_best_25 <= 2
            and legal_moves >= 16
        ):
            return "quiet_equal"

    if "tactical_equal" in matches:
        if (
            move_number >= 10
            and pieces > 12
            and abs(eval_before) <= 240
            and 120 <= second_gap <= 340
            and near_best_25 <= 1
            and legal_moves >= 10
        ):
            return "tactical_equal"

    if "defense" in matches:
        if (
            move_number >= 10
            and pieces > 10
            and -750 <= eval_before <= -180
            and 18 <= second_gap <= 180
            and legal_moves >= 10
        ):
            return "defense"

    return None


def step_score(bucket: str, position: dict[str, Any]) -> float:
    eval_before = float(position["eval_before"])
    second_gap = float(position["second_gap_cp"])
    near_best_25 = int(position["near_best_25"])
    near_best_50 = int(position["near_best_50"])
    legal_moves = int(position["legal_moves"])
    source_cpl = position.get("source_played_cpl_if_multipv")
    score = 0.0

    if bucket == "quiet_equal":
        if 20 <= second_gap <= 70:
            score += 1.2
        if near_best_25 <= 2 and near_best_50 <= 4:
            score += 1.0
        if legal_moves >= 18:
            score += 0.8
        score -= abs(eval_before) / 220.0
    elif bucket == "tactical_equal":
        if 130 <= second_gap <= 300:
            score += 1.2
        if near_best_25 <= 1 and near_best_50 <= 2:
            score += 1.0
        if legal_moves >= 12:
            score += 0.8
        score -= abs(eval_before) / 320.0
    elif bucket == "defense":
        if -650 <= eval_before <= -220:
            score += 1.2
        if 20 <= second_gap <= 140:
            score += 1.0
        if legal_moves >= 12:
            score += 0.8
        score -= abs(abs(eval_before) - 350.0) / 400.0

    if source_cpl is None:
        score -= 0.6
    else:
        score -= min(float(source_cpl), 400.0) / 220.0
    return float(score)


def sequence_type(first_bucket: str, second_bucket: str) -> str:
    names = tuple(sorted([first_bucket.replace("_equal", ""), second_bucket.replace("_equal", "")]))
    if names == ("defense", "quiet"):
        return "quiet_defense"
    if names == ("defense", "tactical"):
        return "tactical_defense"
    if names == ("quiet", "quiet"):
        return "quiet_quiet"
    if names == ("quiet", "tactical"):
        return "quiet_tactical"
    if names == ("tactical", "tactical"):
        return "tactical_tactical"
    return "_".join(names)


def sequence_bonus(seq_type: str, ply_gap: int) -> float:
    bonus = {
        "quiet_quiet": 0.8,
        "quiet_tactical": 1.2,
        "tactical_tactical": 0.7,
        "tactical_defense": 1.3,
        "quiet_defense": 1.0,
    }.get(seq_type, 0.0)
    if ply_gap == 2:
        bonus += 0.6
    elif ply_gap == 4:
        bonus += 0.3
    else:
        bonus -= 0.2
    return bonus


def annotate_candidate(
    candidate: Candidate,
    analyzed: dict[str, Any],
) -> dict[str, Any]:
    position = build_position(candidate, analyzed, bucket="")
    position["matches"] = matching_buckets(candidate, analyzed)
    return position


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pgn-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory of PGN files. Defaults to position_benchmark/games and data/games.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("position_benchmark/nonopening_sequence_screening_positions.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("position_benchmark/nonopening_sequence_screening_report.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/regan_lite_multipv_cache"))
    parser.add_argument("--stockfish-path", default="/opt/homebrew/bin/stockfish")
    parser.add_argument("--depth", type=int, default=16)
    parser.add_argument("--multipv", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=256)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--max-games", type=int, default=250)
    parser.add_argument("--max-analyzed", type=int, default=2500)
    parser.add_argument("--max-ply-gap", type=int, default=6)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--max-per-source-family", type=int, default=3)
    args = parser.parse_args()

    if args.depth < 16:
        raise SystemExit("Use --depth >= 16 for sequence screening.")
    if args.multipv < 2:
        raise SystemExit("Use --multipv >= 2 so rank-style scoring stays available.")

    targets = parse_targets(args.targets)
    pgn_dirs = args.pgn_dir or [Path("position_benchmark/games"), Path("data/games")]
    pgn_paths = list_pgn_paths(pgn_dirs, args.seed, args.max_games)
    print(f"Scanning {len(pgn_paths)} PGNs from {', '.join(str(path) for path in pgn_dirs)}")
    print(f"Targets: {targets}")

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
    analyzed_cache: dict[tuple[str, int], dict[str, Any] | None] = {}
    analyzed_count = 0
    raw_sequences: list[dict[str, Any]] = []

    try:
        try:
            engine.configure({"Threads": args.threads, "Hash": args.hash_mb})
        except chess.engine.EngineError:
            pass

        for path in pgn_paths:
            if analyzed_count >= args.max_analyzed:
                break

            candidates = iter_candidates_from_pgn(path)
            groups: dict[tuple[str, str], list[Candidate]] = defaultdict(list)
            for candidate in candidates:
                groups[(candidate.source_player_id, candidate.side_to_move)].append(candidate)

            for group_candidates in groups.values():
                group_candidates.sort(key=lambda item: item.ply_before)
                for idx in range(len(group_candidates) - 1):
                    first = group_candidates[idx]
                    second = group_candidates[idx + 1]
                    ply_gap = second.ply_before - first.ply_before
                    if ply_gap <= 0 or ply_gap > args.max_ply_gap:
                        continue

                    for candidate in (first, second):
                        if analyzed_count >= args.max_analyzed:
                            break
                        key = (candidate.fen, candidate.ply_before)
                        if key not in analyzed_cache:
                            analyzed_cache[key] = analyze_candidate(
                                candidate,
                                engine,
                                args.depth,
                                args.multipv,
                                args.cache_dir / f"depth{args.depth}_multipv{args.multipv}",
                            )
                            analyzed_count += 1
                            if analyzed_count % 100 == 0:
                                print(
                                    f"Analyzed={analyzed_count} raw_sequences={len(raw_sequences)}",
                                    flush=True,
                                )
                    if analyzed_count >= args.max_analyzed:
                        break
                    analyzed_first = analyzed_cache[(first.fen, first.ply_before)]
                    analyzed_second = analyzed_cache[(second.fen, second.ply_before)]
                    if analyzed_first is None or analyzed_second is None:
                        continue

                    first_position = annotate_candidate(first, analyzed_first)
                    second_position = annotate_candidate(second, analyzed_second)
                    first_bucket = choose_primary_bucket(first_position)
                    second_bucket = choose_primary_bucket(second_position)
                    if first_bucket not in ALLOWED_BUCKETS or second_bucket not in ALLOWED_BUCKETS:
                        continue

                    first_position["regan_bucket"] = first_bucket
                    second_position["regan_bucket"] = second_bucket

                    first_score = step_score(first_bucket, first_position)
                    second_score = step_score(second_bucket, second_position)
                    if min(first_score, second_score) < 1.2:
                        continue

                    seq_type = sequence_type(first_bucket, second_bucket)
                    if seq_type not in targets:
                        continue

                    avg_source_cpl = np.mean(
                        [
                            200.0 if pos.get("source_played_cpl_if_multipv") is None else float(pos["source_played_cpl_if_multipv"])
                            for pos in (first_position, second_position)
                        ]
                    )
                    if avg_source_cpl > 180.0:
                        continue

                    total_score = first_score + second_score + sequence_bonus(seq_type, ply_gap)
                    raw_sequences.append(
                        {
                            "game_id": first.game_id,
                            "source_player_id": first.source_player_id,
                            "source_family": model_family(first.source_player_id),
                            "side_to_move": first.side_to_move,
                            "ply_gap": ply_gap,
                            "sequence_type": seq_type,
                            "score": float(total_score),
                            "steps": [first_position, second_position],
                            "step_buckets": [first_bucket, second_bucket],
                            "step_scores": [float(first_score), float(second_score)],
                        }
                    )
                if analyzed_count >= args.max_analyzed:
                    break
    finally:
        engine.quit()

    raw_sequences.sort(key=lambda item: item["score"], reverse=True)

    selected_sequences: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    source_family_counts: Counter[str] = Counter()
    seen_games: set[str] = set()

    for sequence in raw_sequences:
        seq_type = sequence["sequence_type"]
        if selected_counts[seq_type] >= targets.get(seq_type, 0):
            continue
        if sequence["game_id"] in seen_games:
            continue
        if source_family_counts[sequence["source_family"]] >= args.max_per_source_family:
            continue
        selected_sequences.append(sequence)
        selected_counts[seq_type] += 1
        source_family_counts[sequence["source_family"]] += 1
        seen_games.add(sequence["game_id"])
        if all(selected_counts[name] >= targets[name] for name in targets):
            break

    flattened_positions: list[dict[str, Any]] = []
    sequence_summaries: list[dict[str, Any]] = []
    for seq_idx, sequence in enumerate(selected_sequences):
        sequence_id = f"seq-screen-{seq_idx:03d}"
        sequence_summaries.append(
            {
                "sequence_id": sequence_id,
                "sequence_type": sequence["sequence_type"],
                "score": round(float(sequence["score"]), 3),
                "game_id": sequence["game_id"],
                "source_player_id": sequence["source_player_id"],
                "source_family": sequence["source_family"],
                "side_to_move": sequence["side_to_move"],
                "ply_gap": sequence["ply_gap"],
                "step_buckets": sequence["step_buckets"],
                "step_scores": [round(float(score), 3) for score in sequence["step_scores"]],
            }
        )
        for step_idx, position in enumerate(sequence["steps"], start=1):
            item = dict(position)
            item["sequence_id"] = sequence_id
            item["sequence_step"] = step_idx
            item["sequence_length"] = len(sequence["steps"])
            item["sequence_type"] = sequence["sequence_type"]
            item["sequence_rank_score"] = round(float(sequence["score"]), 3)
            item["sequence_step_bucket"] = sequence["step_buckets"][step_idx - 1]
            item["position_id"] = f"{sequence_id}-step{step_idx}"
            item["type"] = "equal"
            flattened_positions.append(item)

    flattened_positions.sort(key=lambda item: (item["sequence_id"], item["sequence_step"]))
    metadata = {
        "description": "Two-step non-opening sequence screening set mined from raw PGNs",
        "positions": len(flattened_positions),
        "sequences": len(sequence_summaries),
        "targets": targets,
        "selected_counts": dict(selected_counts),
        "source_family_counts": dict(source_family_counts),
        "max_games": args.max_games,
        "max_analyzed": args.max_analyzed,
        "analyzed_positions": analyzed_count,
        "raw_sequences": len(raw_sequences),
        "depth": args.depth,
        "multipv": args.multipv,
        "seed": args.seed,
        "pgn_dirs": [str(path) for path in pgn_dirs],
    }

    args.output.write_text(json.dumps({"metadata": metadata, "positions": flattened_positions}, indent=2))
    args.report.write_text(json.dumps({"metadata": metadata, "sequences": sequence_summaries}, indent=2))

    print()
    print(f"Wrote {len(flattened_positions)} positions across {len(sequence_summaries)} sequences to {args.output}")
    print(f"Wrote sequence report to {args.report}")
    print(f"Selected counts: {dict(selected_counts)}")
    print(f"Source family counts: {dict(source_family_counts)}")


if __name__ == "__main__":
    main()
