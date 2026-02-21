#!/usr/bin/env python3
"""
Verify unified position benchmark data consistency.

Validates:
1. positions.json has correct structure (100 blunder + 200 equal = 300)
2. results.json has 300 results per model with correct per-type breakdowns
3. Summary stats match per-position data
4. CPL formulas are correct
5. Position indices are valid
"""

import json
import sys
from pathlib import Path


def calculate_median(values: list[float]) -> float:
    if not values:
        return 10000
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 0:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2
    return sorted_vals[n // 2]


def verify_positions(base_path: Path) -> list[str]:
    """Verify positions.json structure."""
    issues = []
    positions_file = base_path / "positions.json"

    if not positions_file.exists():
        return [f"positions.json not found at {positions_file}"]

    with open(positions_file) as f:
        data = json.load(f)

    positions = data.get("positions", [])
    metadata = data.get("metadata", {})

    blunders = [p for p in positions if p.get("type") == "blunder"]
    equals = [p for p in positions if p.get("type") == "equal"]
    untyped = [p for p in positions if p.get("type") not in ("blunder", "equal")]

    print(f"  Total positions: {len(positions)}")
    print(f"  Blunder: {len(blunders)}, Equal: {len(equals)}, Untyped: {len(untyped)}")

    if len(blunders) != metadata.get("blunder_count", -1):
        issues.append(f"Metadata blunder_count ({metadata.get('blunder_count')}) != actual ({len(blunders)})")
    if len(equals) != metadata.get("equal_count", -1):
        issues.append(f"Metadata equal_count ({metadata.get('equal_count')}) != actual ({len(equals)})")
    if untyped:
        issues.append(f"{len(untyped)} positions have no type field")

    # Verify required fields
    required_blunder = ["fen", "eval_before", "best_move", "best_move_san", "move_history", "side_to_move", "blunder_move"]
    required_equal = ["fen", "eval_before", "best_move", "best_move_san", "move_history", "side_to_move"]

    for i, pos in enumerate(blunders):
        for field in required_blunder:
            if field not in pos:
                issues.append(f"Blunder position {i} missing field: {field}")
                break

    for i, pos in enumerate(equals):
        for field in required_equal:
            if field not in pos:
                issues.append(f"Equal position {i} missing field: {field}")
                break

    # Check FEN uniqueness
    fens = [p["fen"] for p in positions]
    if len(fens) != len(set(fens)):
        dup_count = len(fens) - len(set(fens))
        issues.append(f"{dup_count} duplicate FENs in positions.json")

    return issues


def verify_results(base_path: Path) -> list[str]:
    """Verify results.json structure and data consistency."""
    issues = []
    results_file = base_path / "results.json"

    if not results_file.exists():
        return [f"results.json not found at {results_file}"]

    with open(results_file) as f:
        data = json.load(f)

    # Load positions to know expected counts
    positions_file = base_path / "positions.json"
    if positions_file.exists():
        with open(positions_file) as f:
            pos_data = json.load(f)
        positions = pos_data["positions"]
        expected_total = len(positions)
        expected_blunder = sum(1 for p in positions if p.get("type") == "blunder")
        expected_equal = sum(1 for p in positions if p.get("type") == "equal")
    else:
        expected_total = 300
        expected_blunder = 100
        expected_equal = 200

    print(f"  Models: {len(data)}")
    print(f"  Expected: {expected_total} positions per model ({expected_blunder} blunder + {expected_equal} equal)")

    # Build position type lookup by index
    pos_type_by_idx = {i: p.get("type") for i, p in enumerate(positions)} if positions_file.exists() else {}

    for model_name, model_data in data.items():
        summary = model_data.get("summary", {})
        results = model_data.get("results", [])
        skipped = summary.get("positions_skipped", 0)

        # Check result count (allow skipped positions)
        if len(results) + skipped != expected_total and len(results) != expected_total:
            issues.append(f"{model_name}: {len(results)} results + {skipped} skipped != {expected_total} expected")
            continue

        if skipped > 0:
            print(f"  {model_name}: {skipped} positions skipped (API errors)")

        # Check position indices are valid (subset of expected range)
        indices = [r["position_idx"] for r in results]
        out_of_range = [i for i in indices if i < 0 or i >= expected_total]
        if out_of_range:
            issues.append(f"{model_name}: position indices out of range: {out_of_range[:5]}")
        if len(indices) != len(set(indices)):
            dup_count = len(indices) - len(set(indices))
            issues.append(f"{model_name}: {dup_count} duplicate position indices")

        # Verify required fields in results
        required_fields = ["position_idx", "fen", "model_move", "best_move", "cpl", "is_legal", "is_best"]
        for r in results[:5]:
            for field in required_fields:
                if field not in r:
                    issues.append(f"{model_name}: result missing field '{field}'")
                    break

        # Verify overall summary matches per-position data
        if not results:
            continue

        legal = [r for r in results if r["is_legal"]]
        all_cpls = [r["cpl"] for r in results]

        computed_legal_pct = len(legal) / len(results) * 100
        if abs(computed_legal_pct - summary["legal_pct"]) > 0.01:
            issues.append(f"{model_name}: legal_pct mismatch: summary={summary['legal_pct']}, computed={computed_legal_pct}")

        computed_avg_cpl = sum(all_cpls) / len(all_cpls)
        if abs(computed_avg_cpl - summary["avg_cpl"]) > 0.1:
            issues.append(f"{model_name}: avg_cpl mismatch: summary={summary['avg_cpl']:.1f}, computed={computed_avg_cpl:.1f}")

        computed_best_count = sum(1 for r in results if r["is_best"])
        computed_best_pct = computed_best_count / len(results) * 100
        if abs(computed_best_pct - summary["best_pct"]) > 0.01:
            issues.append(f"{model_name}: best_pct mismatch: summary={summary['best_pct']}, computed={computed_best_pct}")

        # Verify per-type breakdowns using position_idx to look up type
        for ptype in ["blunder", "equal"]:
            if ptype not in summary:
                issues.append(f"{model_name}: missing '{ptype}' breakdown in summary")
                continue

            type_summary = summary[ptype]
            type_results = [r for r in results if pos_type_by_idx.get(r["position_idx"]) == ptype]

            if not type_results:
                continue

            if type_summary["total_positions"] != len(type_results):
                issues.append(f"{model_name}.{ptype}: total_positions={type_summary['total_positions']}, actual results={len(type_results)}")

            type_legal = [r for r in type_results if r["is_legal"]]
            type_cpls = [r["cpl"] for r in type_results]

            computed_type_legal_pct = len(type_legal) / len(type_results) * 100
            if abs(computed_type_legal_pct - type_summary["legal_pct"]) > 0.01:
                issues.append(f"{model_name}.{ptype}: legal_pct mismatch: summary={type_summary['legal_pct']}, computed={computed_type_legal_pct}")

            computed_type_avg_cpl = sum(type_cpls) / len(type_cpls)
            if abs(computed_type_avg_cpl - type_summary["avg_cpl"]) > 0.1:
                issues.append(f"{model_name}.{ptype}: avg_cpl mismatch: summary={type_summary['avg_cpl']:.1f}, computed={computed_type_avg_cpl:.1f}")

        # Verify CPL formula for illegal moves: cpl = eval_before + 5000
        for r in results:
            if not r["is_legal"] and "eval_before" in r:
                expected_cpl = r["eval_before"] + 5000
                if abs(r["cpl"] - expected_cpl) > 0.1:
                    issues.append(f"{model_name} pos {r['position_idx']}: illegal CPL={r['cpl']}, expected eval_before+5000={expected_cpl}")
                    break  # One example is enough

    return issues


def main():
    base_path = Path(__file__).parent

    print("=" * 70)
    print("UNIFIED POSITION BENCHMARK VERIFICATION")
    print("=" * 70)

    all_issues = []

    print("\n1. POSITIONS FILE")
    print("-" * 70)
    pos_issues = verify_positions(base_path)
    all_issues.extend(pos_issues)
    if pos_issues:
        for issue in pos_issues:
            print(f"  FAIL: {issue}")
    else:
        print("  PASS")

    print("\n2. RESULTS FILE")
    print("-" * 70)
    res_issues = verify_results(base_path)
    all_issues.extend(res_issues)
    if res_issues:
        for issue in res_issues:
            print(f"  FAIL: {issue}")
    else:
        print("  PASS")

    print("\n" + "=" * 70)
    if all_issues:
        print(f"FAILED: {len(all_issues)} issues found")
        sys.exit(1)
    else:
        print("ALL CHECKS PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
