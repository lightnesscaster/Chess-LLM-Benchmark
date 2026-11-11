"""Rescore saved static-panel responses with Stockfish, without any model calls.

The required isolated output is a crash-resumable copy of the input registry.
Only complete, current history-replay records are eligible. Legacy records retain
all their original data, and screening MultiPV ranks are retained as research
metadata rather than being represented as corrected depth-30 rankings.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
from typing import Any, Callable

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from position_benchmark.evaluation import ConsistentEvaluator, SCORING_VERSION
from position_benchmark.layout import (
    CORE_POSITIONS_PATH, CORE_RESULTS_PATH, GAME_LIKE_POSITIONS_PATH,
    GAME_LIKE_RESULTS_PATH, BLUNDER_POSITIONS_PATH, BLUNDER_RESULTS_PATH, RESULTS_DIR,
)
from position_benchmark.predictions import benchmark_result_readiness, result_row_is_current
from position_benchmark.run_benchmark import _calculate_median, replay_position_board, stamp_scoring_summary


SCORING_FIELDS = frozenset({
    "best_move", "best_move_san", "eval_before", "eval_best", "eval_model",
    "cpl", "is_best", "scoring_version", "scoring_profile",
})
PANEL_PATHS = {
    "core": (CORE_POSITIONS_PATH, CORE_RESULTS_PATH, 50),
    "game_like": (GAME_LIKE_POSITIONS_PATH, GAME_LIKE_RESULTS_PATH, 48),
    "blunder": (BLUNDER_POSITIONS_PATH, BLUNDER_RESULTS_PATH, 25),
}
RANK_SOURCE = "screening-multipv"


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _atomic_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _aggregate(rows: list[dict]) -> dict:
    """Match the static runner's evaluation aggregates without altering usage."""
    count = len(rows)
    legal = [row for row in rows if row["is_legal"]]
    losses = [row["cpl"] for row in rows]
    legal_losses = [row["cpl"] for row in legal]
    best = sum(bool(row["is_best"]) for row in rows)
    avoided = sum(bool(row.get("avoided_blunder")) for row in rows)
    return {
        "total_positions": count,
        "legal_moves": len(legal),
        "legal_pct": len(legal) / count * 100 if count else 0,
        "best_moves": best,
        "best_pct": best / count * 100 if count else 0,
        "avoided_blunders": avoided,
        "avoided_pct": avoided / count * 100 if count else 0,
        "avg_cpl": sum(losses) / count if count else 10000,
        "avg_cpl_legal": sum(legal_losses) / len(legal_losses) if legal_losses else 10000,
        "median_cpl": _calculate_median(losses),
    }


def _positions_for_rows(rows: list[dict], positions: list[dict]) -> list[dict]:
    by_id = {position.get("position_id"): position for position in positions}
    if None in by_id or len(by_id) != len(positions):
        raise ValueError("Panel must have unique stable position_id values")
    resolved = []
    for row in rows:
        position = by_id.get(row.get("position_id"))
        if position is None:
            raise ValueError(f"Unknown or missing position_id: {row.get('position_id')!r}")
        if row.get("fen") != position["fen"]:
            raise ValueError(f"FEN mismatch for position_id {row['position_id']}")
        if not result_row_is_current(row):
            raise ValueError(f"Refusing legacy row in ready record: {row['position_id']}")
        resolved.append(position)
    return resolved


def rescore_record(record: dict, positions: list[dict], evaluator: Any, depth: int = 30) -> dict:
    """Replace only scoring fields and aggregates on an eligible saved record."""
    corrected = copy.deepcopy(record)
    rows = corrected["results"]
    row_positions = _positions_for_rows(rows, positions)
    for row, position in zip(rows, row_positions):
        board = replay_position_board(position)
        move = None
        if row["is_legal"]:
            try:
                move = chess.Move.from_uci(row["model_move"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"Saved legal move is malformed at {row['position_id']}") from error
            if move not in board.legal_moves:
                raise ValueError(f"Saved legal move is illegal at {row['position_id']}: {move}")
        score = evaluator.score(board, move)
        if score.get("scoring_version") != SCORING_VERSION or score.get("scoring_profile") != evaluator.scoring_profile:
            raise ValueError("Evaluator returned inconsistent scoring provenance")
        row.update({key: score[key] for key in SCORING_FIELDS})
        if row.get("move_rank") is not None:
            row["rank_metadata_source"] = RANK_SOURCE
    summary = corrected.setdefault("summary", {})
    engine_profile = getattr(evaluator, "engine_profile", None)
    if engine_profile is not None:
        summary["scoring_engine_profile"] = copy.deepcopy(engine_profile)
    summary.update(_aggregate(rows))
    stamp_scoring_summary(summary, rows)
    for kind in ("equal", "blunder"):
        subset = [row for row, position in zip(rows, row_positions) if position.get("type") == kind]
        if subset:
            subsum = summary.setdefault(kind, {})
            subsum.update(_aggregate(subset))
            stamp_scoring_summary(subsum, subset)
            if any(row.get("move_rank") is not None for row in subset):
                subsum["rank_metadata_source"] = RANK_SOURCE
    if any(row.get("move_rank") is not None for row in rows):
        summary["rank_metadata_source"] = RANK_SOURCE
    return corrected


class _EngineEvaluator:
    """Own one persistent engine and scorer for exactly one worker thread."""

    def __init__(self, stockfish: str, depth: int, cache_path: Path | None) -> None:
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish)
        try:
            self.evaluator = ConsistentEvaluator(self.engine, depth=depth, cache_path=cache_path)
        except BaseException:
            self.engine.quit()
            raise
        self.scoring_profile = self.evaluator.scoring_profile
        self.engine_profile = self.evaluator.engine_profile

    def score(self, board: chess.Board, move: chess.Move | None) -> dict:
        return self.evaluator.score(board, move)

    def close(self) -> None:
        self.engine.quit()


def default_stockfish_path() -> str:
    """Use an explicit configured Stockfish binary, then the available binary."""
    import yaml
    config_path = Path(__file__).resolve().parents[1] / "config" / "benchmark.yaml"
    config = yaml.safe_load(config_path.read_text()) or {}
    for engine in config.get("engines", []):
        if engine.get("type") == "stockfish":
            configured = engine.get("engine_path") or engine.get("stockfish_path") or engine.get("path")
            if configured:
                path = Path(configured).expanduser()
                return str(path if path.is_absolute() else config_path.parent.parent / path)
    return shutil.which("stockfish") or "stockfish"


def rescore_file(
    *, panel: str, output_path: Path, input_path: Path | None = None,
    stockfish: str | None = None, workers: int = 4, depth: int = 30,
    cache_path: Path | None = None, positions: list[dict] | None = None,
    evaluator_factory: Callable[[], Any] | None = None,
) -> dict:
    """Rescore eligible records and atomically checkpoint each completed model."""
    if workers < 1 or depth < 30:
        raise ValueError("Require at least one worker and Stockfish depth >= 30")
    positions_path, canonical_input, minimum = PANEL_PATHS[panel]
    source = Path(input_path or canonical_input).resolve()
    output = Path(output_path).resolve()
    if output == source or output.is_relative_to(RESULTS_DIR.resolve()):
        raise ValueError("Output must be an isolated copy outside production results and input")
    inputs = json.loads(source.read_text())
    if positions is None:
        positions = json.loads(positions_path.read_text())["positions"]
    previous = json.loads(output.read_text()) if output.exists() else {}
    result = copy.deepcopy(inputs)
    eligible = []
    for player, record in inputs.items():
        readiness = benchmark_result_readiness(
            record, positions, min_equal_positions=minimum,
            position_type="blunder" if panel == "blunder" else "equal",
        )
        if readiness.is_ready:
            # Validate identity and legal flags before starting expensive searches.
            resolved = _positions_for_rows(record["results"], positions)
            for row, position in zip(record["results"], resolved):
                if row.get("is_legal"):
                    board = replay_position_board(position)
                    try:
                        move = chess.Move.from_uci(row["model_move"])
                    except (ValueError, TypeError) as error:
                        raise ValueError(f"Saved legal move is malformed for {player}") from error
                    if move not in board.legal_moves:
                        raise ValueError(f"Saved legal move is illegal for {player}: {move}")
            eligible.append(player)
        else:
            print(f"skip {player}: {readiness.reason}", flush=True)
    if not eligible:
        _atomic_write(output, result)
        return result
    factory = evaluator_factory or (lambda: _EngineEvaluator(stockfish or default_stockfish_path(), depth, cache_path))
    first = factory()
    resources = [first]
    lock = threading.Lock()
    local = threading.local()
    unused_first = [first]
    profile = copy.deepcopy(first.scoring_profile)
    positions_fingerprint = _fingerprint(positions)

    def provenance(player: str) -> dict:
        return {
            "input_sha256": _fingerprint(inputs[player]),
            "positions_sha256": positions_fingerprint,
            "scoring_version": SCORING_VERSION, "scoring_profile": profile,
            "stockfish_depth": depth, "panel": panel,
        }

    def rescore(player: str) -> tuple[str, dict]:
        if not hasattr(local, "evaluator"):
            with lock:
                if unused_first:
                    local.evaluator = unused_first.pop()
                else:
                    local.evaluator = factory()
                    resources.append(local.evaluator)
            if local.evaluator.scoring_profile != profile:
                raise ValueError("Worker engine profiles differ")
        corrected = rescore_record(inputs[player], positions, local.evaluator, depth)
        corrected.pop("rescoring", None)
        corrected["rescoring"] = {**provenance(player), "output_sha256": _fingerprint(corrected)}
        return player, corrected

    try:
        pending = []
        for player in eligible:
            saved = previous.get(player, {})
            expected = provenance(player)
            if (
                {key: value for key, value in saved.get("rescoring", {}).items() if key != "output_sha256"} == expected
                and saved.get("rescoring", {}).get("output_sha256") == _fingerprint(
                    {key: value for key, value in saved.items() if key != "rescoring"}
                )
                and saved.get("summary", {}).get("scoring_version") == SCORING_VERSION
                and saved.get("summary", {}).get("scoring_profile") == profile
                and all(row.get("scoring_version") == SCORING_VERSION and row.get("scoring_profile") == profile
                        for row in saved.get("results", []))
                and len(saved.get("results", [])) == len(inputs[player]["results"])
            ):
                result[player] = saved
                print(f"resume {player}: corrected", flush=True)
            else:
                pending.append(player)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(rescore, player) for player in pending]
            for completed, future in enumerate(as_completed(futures), start=1):
                player, corrected = future.result()
                result[player] = corrected
                _atomic_write(output, result)
                print(f"[{completed}/{len(pending)}] {player}: {len(corrected['results'])} rescored", flush=True)
        # Also checkpoint original skipped records and removals on an all-resumed run.
        _atomic_write(output, result)
    finally:
        for resource in resources:
            close = getattr(resource, "close", None)
            if close:
                close()
    return result


def main() -> None:
    """Run CPU-only, full-depth rescoring into a required isolated output file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", choices=PANEL_PATHS, required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stockfish", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--depth", type=int, default=30)
    parser.add_argument("--cache-path", type=Path)
    args = parser.parse_args()
    rescore_file(panel=args.panel, input_path=args.input, output_path=args.output,
                 stockfish=args.stockfish, workers=args.workers, depth=args.depth,
                 cache_path=args.cache_path)


if __name__ == "__main__":
    main()
