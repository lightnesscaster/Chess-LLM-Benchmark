#!/usr/bin/env python3
"""Re-evaluate all position benchmark CPL calculations at Stockfish depth 30.

Uses 8 parallel Stockfish instances for throughput.
Re-evaluates:
  1. Position files: eval_before, best_move verification
  2. Result files: eval_model (re-evaluate each model's recorded move), recalculate CPL

Usage:
    python position_benchmark/reeval_depth30.py [--depth 30] [--workers 8]
"""

import argparse
import chess
import chess.engine
import json
import threading
import time
from pathlib import Path
from collections import defaultdict


DEPTH = 30
NUM_WORKERS = 8
HASH_MB = 128


def eval_to_cp(info, perspective: chess.Color) -> float:
    """Convert engine info to centipawns from perspective's point of view."""
    score = info["score"].pov(perspective)
    if score.is_mate():
        mate_in = score.mate()
        if mate_in > 0:
            return 10000 + (100 - mate_in) * 10
        else:
            return -10000 - mate_in * 10
    cp = score.score()
    return cp if cp is not None else 0.0


class ParallelEvaluator:
    """Evaluate positions in parallel using multiple Stockfish instances."""

    def __init__(self, num_workers=NUM_WORKERS, depth=DEPTH, hash_mb=HASH_MB):
        self.num_workers = num_workers
        self.depth = depth
        self.hash_mb = hash_mb
        self.engines = []
        self.lock = threading.Lock()
        self.completed = 0
        self.total = 0

    def start(self):
        print(f"Starting {self.num_workers} Stockfish instances (depth={self.depth}, hash={self.hash_mb}MB)...")
        for _ in range(self.num_workers):
            engine = chess.engine.SimpleEngine.popen_uci("stockfish")
            engine.configure({"Hash": self.hash_mb})
            self.engines.append(engine)

    def stop(self):
        for engine in self.engines:
            engine.quit()
        self.engines = []

    def evaluate_positions(self, tasks):
        """Evaluate a list of (task_id, fen, move_uci_or_None) tuples.

        If move_uci is None, evaluates the position (eval_before + best move).
        If move_uci is provided, evaluates after that move is played.

        Returns dict of task_id -> result dict.
        """
        self.completed = 0
        self.total = len(tasks)
        results = {}
        results_lock = threading.Lock()
        self.start_time = time.time()

        # Distribute tasks across workers
        chunks = [[] for _ in range(self.num_workers)]
        for i, task in enumerate(tasks):
            chunks[i % self.num_workers].append(task)

        def worker(worker_id, worker_tasks):
            engine = self.engines[worker_id]
            for task_id, fen, move_uci in worker_tasks:
                board = chess.Board(fen)
                perspective = board.turn

                if move_uci is None:
                    # Evaluate position: get eval_before and best move
                    info = engine.analyse(board, chess.engine.Limit(depth=self.depth))
                    eval_cp = eval_to_cp(info, perspective)
                    best_move = info["pv"][0]
                    result = {
                        "eval": eval_cp,
                        "best_move_uci": best_move.uci(),
                        "best_move_san": board.san(best_move),
                    }
                else:
                    # Evaluate a specific move
                    try:
                        move = chess.Move.from_uci(move_uci)
                        if move not in board.legal_moves:
                            result = {"eval": None, "illegal": True}
                        else:
                            board.push(move)
                            info = engine.analyse(board, chess.engine.Limit(depth=self.depth))
                            # Negate because we want eval from the mover's perspective
                            eval_cp = -eval_to_cp(info, not perspective)
                            result = {"eval": eval_cp, "illegal": False}
                    except (ValueError, chess.InvalidMoveError):
                        result = {"eval": None, "illegal": True}

                with results_lock:
                    results[task_id] = result
                    self.completed += 1
                    if self.completed % 50 == 0 or self.completed == self.total:
                        elapsed = time.time() - self.start_time
                        rate = self.completed / elapsed if elapsed > 0 else 0
                        eta = (self.total - self.completed) / rate if rate > 0 else 0
                        print(f"  {self.completed}/{self.total} done ({rate:.1f}/s, ETA {eta:.0f}s)", flush=True)

        threads = [threading.Thread(target=worker, args=(w, chunks[w])) for w in range(self.num_workers)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.time() - t0
        print(f"  Completed {len(results)} evaluations in {elapsed:.1f}s ({elapsed/len(results):.2f}s/eval)")

        return results


def reeval_position_file(evaluator, filepath, position_key="positions"):
    """Re-evaluate eval_before and best_move in a position file."""
    print(f"\n{'='*60}")
    print(f"Re-evaluating position file: {filepath}")
    print(f"{'='*60}")

    with open(filepath) as f:
        data = json.load(f)

    positions = data[position_key]
    print(f"  {len(positions)} positions")

    # Build tasks
    tasks = []
    for i, pos in enumerate(positions):
        tasks.append((i, pos["fen"], None))

    results = evaluator.evaluate_positions(tasks)

    # Update positions
    changes = 0
    for i, pos in enumerate(positions):
        r = results[i]
        old_eval = pos.get("eval_before", pos.get("eval"))
        new_eval = int(r["eval"])
        new_best_uci = r["best_move_uci"]
        new_best_san = r["best_move_san"]

        if "eval_before" in pos:
            if pos["eval_before"] != new_eval:
                changes += 1
            pos["eval_before"] = new_eval
        elif "eval" in pos:
            if pos["eval"] != new_eval:
                changes += 1
            pos["eval"] = new_eval

        old_best = pos.get("best_move", pos.get("best_move_uci"))
        if old_best != new_best_uci:
            changes += 1

        if "best_move" in pos:
            pos["best_move"] = new_best_uci
        elif "best_move_uci" in pos:
            pos["best_move_uci"] = new_best_uci

        pos["best_move_san"] = new_best_san

    print(f"  {changes} values changed")

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {filepath}")


def reeval_results_file(evaluator, results_filepath, positions_filepath, fen_cache=None):
    """Re-evaluate all model moves in a results file."""
    print(f"\n{'='*60}")
    print(f"Re-evaluating results: {results_filepath}")
    print(f"{'='*60}")

    with open(results_filepath) as f:
        all_results = json.load(f)

    with open(positions_filepath) as f:
        pos_data = json.load(f)
    positions = pos_data["positions"]

    # Collect all evaluation tasks (only legal moves need re-evaluation)
    tasks = []

    for model_name, model_data in all_results.items():
        results = model_data.get("results", [])
        if isinstance(model_data, list):
            results = model_data

        for j, result in enumerate(results):
            fen = result["fen"]
            model_move = result.get("model_move", "")
            is_legal = result.get("is_legal", True)

            if not is_legal or not model_move:
                continue

            task_id = f"{model_name}:{j}"
            tasks.append((task_id, fen, model_move))

    print(f"  {len(all_results)} models, {len(tasks)} legal moves to re-evaluate")

    if tasks:
        eval_results = evaluator.evaluate_positions(tasks)
    else:
        eval_results = {}

    # Build FEN lookup (use cache if provided, otherwise evaluate)
    if fen_cache is None:
        unique_fens = set()
        for model_data in all_results.values():
            results = model_data.get("results", []) if isinstance(model_data, dict) else model_data if isinstance(model_data, list) else []
            for r in results:
                unique_fens.add(r["fen"])

        print(f"  Re-evaluating {len(unique_fens)} unique position FENs for eval_before...")
        pos_tasks = [(f"pos:{fen}", fen, None) for fen in unique_fens]
        pos_results = evaluator.evaluate_positions(pos_tasks)

        fen_cache = {}
        for fen in unique_fens:
            r = pos_results[f"pos:{fen}"]
            fen_cache[fen] = {
                "eval_before": int(r["eval"]),
                "best_move_uci": r["best_move_uci"],
                "best_move_san": r["best_move_san"],
            }

    # Update all results
    for model_name, model_data in all_results.items():
        results = model_data.get("results", [])
        if isinstance(model_data, list):
            results = model_data

        for j, result in enumerate(results):
            fen = result["fen"]

            if fen not in fen_cache:
                continue

            cached = fen_cache[fen]
            new_eval_before = cached["eval_before"]
            new_best_uci = cached["best_move_uci"]
            new_best_san = cached["best_move_san"]

            # Update eval_before and eval_best
            result["eval_before"] = new_eval_before
            result["eval_best"] = new_eval_before

            # Update best_move
            result["best_move"] = new_best_uci
            result["best_move_san"] = new_best_san

            # Update is_best
            model_move = result.get("model_move", "")
            result["is_best"] = model_move == new_best_uci

            task_id = f"{model_name}:{j}"
            if task_id in eval_results:
                er = eval_results[task_id]
                if er.get("illegal"):
                    result["is_legal"] = False
                    result["eval_model"] = -5000
                    result["cpl"] = new_eval_before + 5000
                else:
                    new_eval_model = int(er["eval"])
                    result["eval_model"] = new_eval_model
                    result["cpl"] = max(0, new_eval_before - new_eval_model)
            else:
                # Illegal move - recalculate CPL with new eval_before
                if not result.get("is_legal", True):
                    result["eval_model"] = -5000
                    result["cpl"] = new_eval_before + 5000

        # Recalculate summary
        if isinstance(model_data, dict) and "summary" in model_data:
            summary = model_data["summary"]
            _recalc_summary(summary, results, positions)

    with open(results_filepath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved {results_filepath}")


def _recalc_summary(summary, results, positions=None):
    """Recalculate summary statistics from results."""
    n = len(results)
    if n == 0:
        return

    legal = [r for r in results if r.get("is_legal", True)]
    all_cpls = [r["cpl"] for r in results]
    legal_cpls = [r["cpl"] for r in legal]

    summary["total_positions"] = n
    summary["legal_moves"] = len(legal)
    summary["legal_pct"] = len(legal) / n * 100
    summary["best_moves"] = sum(1 for r in results if r.get("is_best", False))
    summary["best_pct"] = summary["best_moves"] / n * 100
    summary["avoided_blunders"] = sum(1 for r in results if r.get("avoided_blunder", True))
    summary["avoided_pct"] = summary["avoided_blunders"] / n * 100
    summary["avg_cpl"] = sum(all_cpls) / n
    if legal_cpls:
        summary["avg_cpl_legal"] = sum(legal_cpls) / len(legal_cpls)
    summary["median_cpl"] = sorted(all_cpls)[n // 2]

    # Per-type breakdowns if positions available
    if positions:
        pos_type_by_fen = {p["fen"]: p.get("type", "") for p in positions}
        for type_name in ["blunder", "equal", "puzzle"]:
            type_results = [r for r in results if pos_type_by_fen.get(r["fen"]) == type_name]
            if type_results:
                type_legal = [r for r in type_results if r.get("is_legal", True)]
                tn = len(type_results)
                summary[type_name] = {
                    "total_positions": tn,
                    "legal_moves": len(type_legal),
                    "legal_pct": len(type_legal) / tn * 100,
                    "best_moves": sum(1 for r in type_results if r.get("is_best", False)),
                    "best_pct": sum(1 for r in type_results if r.get("is_best", False)) / tn * 100,
                    "avg_cpl": sum(r["cpl"] for r in type_results) / tn,
                }


def main():
    parser = argparse.ArgumentParser(description="Re-evaluate all CPL at higher Stockfish depth")
    parser.add_argument("--depth", type=int, default=DEPTH, help=f"Stockfish depth (default: {DEPTH})")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS, help=f"Parallel workers (default: {NUM_WORKERS})")
    args = parser.parse_args()

    evaluator = ParallelEvaluator(num_workers=args.workers, depth=args.depth)
    evaluator.start()

    base = Path("position_benchmark")

    try:
        # Phase 1: Re-evaluate position files
        print("\n" + "=" * 60)
        print("PHASE 1: Re-evaluating position files")
        print("=" * 60)

        position_files = [
            base / "positions.json",
            base / "lichess_puzzles.json",
        ]
        for pf in position_files:
            if pf.exists():
                reeval_position_file(evaluator, pf)

        # Game blunders has slightly different structure
        gb_path = base / "game_blunders.json"
        if gb_path.exists():
            reeval_position_file(evaluator, gb_path)

        # Phase 2: Build FEN eval cache from all result files first
        print("\n" + "=" * 60)
        print("PHASE 2: Building FEN eval cache")
        print("=" * 60)

        # Collect ALL unique FENs across all result files
        all_unique_fens = set()
        result_files_to_process = []

        results_path = base / "results.json"
        if results_path.exists():
            result_files_to_process.append((results_path, base / "positions.json"))
            with open(results_path) as f:
                d = json.load(f)
            for mdata in d.values():
                for r in mdata.get("results", []):
                    all_unique_fens.add(r["fen"])

        lichess_files = sorted(f for f in base.glob("lichess_results_*.json") if "backup" not in f.name)
        for lf in lichess_files:
            result_files_to_process.append((lf, base / "lichess_puzzles.json"))
            with open(lf) as f:
                d = json.load(f)
            for mdata in d.values():
                results = mdata.get("results", []) if isinstance(mdata, dict) else mdata if isinstance(mdata, list) else []
                for r in results:
                    all_unique_fens.add(r["fen"])

        print(f"  {len(all_unique_fens)} unique FENs across all result files")
        pos_tasks = [(f"pos:{fen}", fen, None) for fen in all_unique_fens]
        pos_results = evaluator.evaluate_positions(pos_tasks)

        fen_cache = {}
        for fen in all_unique_fens:
            r = pos_results[f"pos:{fen}"]
            fen_cache[fen] = {
                "eval_before": int(r["eval"]),
                "best_move_uci": r["best_move_uci"],
                "best_move_san": r["best_move_san"],
            }

        # Phase 3: Re-evaluate result files using cached FEN evals
        print("\n" + "=" * 60)
        print("PHASE 3: Re-evaluating result files")
        print("=" * 60)

        for rf, pf in result_files_to_process:
            reeval_results_file(evaluator, rf, pf, fen_cache=fen_cache)

    finally:
        evaluator.stop()

    print("\n" + "=" * 60)
    print("DONE! All evaluations updated to depth " + str(args.depth))
    print("=" * 60)


if __name__ == "__main__":
    main()
