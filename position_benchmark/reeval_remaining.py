#!/usr/bin/env python3
"""Continue depth-30 re-evaluation for remaining lichess result files only."""

import json
from pathlib import Path
from reeval_depth30 import ParallelEvaluator, reeval_results_file

ALREADY_DONE = {
    "lichess_results_batch1.json",
    "lichess_results_batch2.json",
    "lichess_results_ds31.json",
    "lichess_results_gemini25flash.json",
    "lichess_results_gemini2flash.json",
    "lichess_results_gemini3.json",
}

base = Path("position_benchmark")

# Collect remaining lichess files
remaining = []
for f in sorted(base.glob("lichess_results_*.json")):
    if "backup" in f.name:
        continue
    if f.name in ALREADY_DONE:
        continue
    remaining.append(f)

print(f"Remaining files to process: {len(remaining)}")
for f in remaining:
    print(f"  {f.name}")

# Rebuild FEN cache from all result files (needed for eval_before values)
print("\nBuilding FEN eval cache...")
all_unique_fens = set()

# Collect FENs from remaining files
for lf in remaining:
    with open(lf) as f:
        d = json.load(f)
    for mdata in d.values():
        results = mdata.get("results", []) if isinstance(mdata, dict) else mdata if isinstance(mdata, list) else []
        for r in results:
            all_unique_fens.add(r["fen"])

print(f"  {len(all_unique_fens)} unique FENs to evaluate")

evaluator = ParallelEvaluator(num_workers=8, depth=30)
evaluator.start()

try:
    # Build cache
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

    # Process remaining files
    for rf in remaining:
        reeval_results_file(evaluator, rf, base / "lichess_puzzles.json", fen_cache=fen_cache)

finally:
    evaluator.stop()

print("\nDONE! All remaining files updated to depth 30.")
