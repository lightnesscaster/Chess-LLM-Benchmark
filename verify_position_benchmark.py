#!/usr/bin/env python3
"""
Verify position benchmark data consistency.
"""

import json
from pathlib import Path

def verify_results():
    """Verify all aspects of position benchmark results."""

    results_path = Path("/Volumes/MainStorage/Programming/chess_llm_benchmark/position_benchmark/results.json")
    combined_path = Path("/Volumes/MainStorage/Programming/chess_llm_benchmark/position_benchmark/results_combined.json")

    print("=" * 80)
    print("POSITION BENCHMARK DATA VERIFICATION")
    print("=" * 80)

    # Load both files
    with open(results_path) as f:
        results = json.load(f)

    with open(combined_path) as f:
        combined = json.load(f)

    issues = []

    # Check 1: Verify illegal CPL formula in code vs documentation
    print("\n1. ILLEGAL CPL FORMULA CHECK")
    print("-" * 80)

    documented_formula = combined["metadata"]["illegal_cpl_formula"]
    expected_formula = "eval_before + 5000 (half swing to losing)"

    print(f"Documented formula: {documented_formula}")
    print(f"Expected formula:   {expected_formula}")

    if documented_formula != expected_formula:
        issues.append(f"MISMATCH: Documented formula '{documented_formula}' != expected '{expected_formula}'")
        print("ERROR: Formula mismatch!")
    else:
        print("PASS: Formula documented correctly")

    # Check 2: Verify individual position results
    print("\n2. INDIVIDUAL POSITION CPL VERIFICATION")
    print("-" * 80)

    for player_id, player_data in results.items():
        print(f"\nChecking {player_id}...")

        position_results = player_data["results"]
        summary = player_data["summary"]

        # Verify illegal move CPL values
        illegal_count = 0
        illegal_cpl_errors = []

        for i, pos_result in enumerate(position_results):
            if not pos_result["is_legal"]:
                illegal_count += 1
                eval_best = pos_result["eval_best"]
                expected_cpl = eval_best + 5000
                actual_cpl = pos_result["cpl"]

                # Allow small floating point differences
                if abs(expected_cpl - actual_cpl) > 1:
                    illegal_cpl_errors.append({
                        "position_idx": i,
                        "expected": expected_cpl,
                        "actual": actual_cpl,
                        "eval_best": eval_best
                    })

        if illegal_cpl_errors:
            issues.append(f"{player_id}: {len(illegal_cpl_errors)} illegal moves with wrong CPL")
            print(f"  ERROR: {len(illegal_cpl_errors)} illegal moves have incorrect CPL:")
            for err in illegal_cpl_errors[:3]:  # Show first 3
                print(f"    Position {err['position_idx']}: expected {err['expected']:.1f}, got {err['actual']:.1f}")
        else:
            print(f"  PASS: All {illegal_count} illegal moves have correct CPL (eval_before + 5000)")

        # Verify legal move CPL values
        legal_cpl_errors = []
        for i, pos_result in enumerate(position_results):
            if pos_result["is_legal"]:
                eval_best = pos_result["eval_best"]
                eval_model = pos_result["eval_model"]
                expected_cpl = max(0, eval_best - eval_model)
                actual_cpl = pos_result["cpl"]

                if abs(expected_cpl - actual_cpl) > 1:
                    legal_cpl_errors.append({
                        "position_idx": i,
                        "expected": expected_cpl,
                        "actual": actual_cpl,
                        "eval_best": eval_best,
                        "eval_model": eval_model
                    })

        if legal_cpl_errors:
            issues.append(f"{player_id}: {len(legal_cpl_errors)} legal moves with wrong CPL")
            print(f"  ERROR: {len(legal_cpl_errors)} legal moves have incorrect CPL:")
            for err in legal_cpl_errors[:3]:
                print(f"    Position {err['position_idx']}: expected {err['expected']:.1f}, got {err['actual']:.1f}")

        # Check 3: Verify summary statistics
        print(f"\n  Verifying summary statistics...")

        total_positions = len(position_results)
        legal_count = sum(1 for r in position_results if r["is_legal"])
        best_count = sum(1 for r in position_results if r["is_best"])

        # Verify counts
        if summary["total_positions"] != total_positions:
            issues.append(f"{player_id}: total_positions mismatch")
            print(f"    ERROR: total_positions {summary['total_positions']} != {total_positions}")

        if summary["legal_moves"] != legal_count:
            issues.append(f"{player_id}: legal_moves count mismatch")
            print(f"    ERROR: legal_moves {summary['legal_moves']} != {legal_count}")

        if summary["best_moves"] != best_count:
            issues.append(f"{player_id}: best_moves count mismatch")
            print(f"    ERROR: best_moves {summary['best_moves']} != {best_count}")

        # Verify percentages
        expected_legal_pct = (legal_count / total_positions * 100) if total_positions > 0 else 0
        if abs(summary["legal_pct"] - expected_legal_pct) > 0.1:
            issues.append(f"{player_id}: legal_pct calculation error")
            print(f"    ERROR: legal_pct {summary['legal_pct']:.2f} != {expected_legal_pct:.2f}")

        expected_best_pct = (best_count / total_positions * 100) if total_positions > 0 else 0
        if abs(summary["best_pct"] - expected_best_pct) > 0.1:
            issues.append(f"{player_id}: best_pct calculation error")
            print(f"    ERROR: best_pct {summary['best_pct']:.2f} != {expected_best_pct:.2f}")

        # Verify avg_cpl (includes all moves, legal and illegal)
        all_cpls = [r["cpl"] for r in position_results]
        expected_avg_cpl = sum(all_cpls) / len(all_cpls) if all_cpls else 10000

        if abs(summary["avg_cpl"] - expected_avg_cpl) > 0.1:
            issues.append(f"{player_id}: avg_cpl calculation error")
            print(f"    ERROR: avg_cpl {summary['avg_cpl']:.2f} != {expected_avg_cpl:.2f}")
            print(f"      Sum of CPLs: {sum(all_cpls):.2f}, Count: {len(all_cpls)}")
        else:
            print(f"    PASS: avg_cpl = {summary['avg_cpl']:.1f} (verified)")

        # Verify avg_cpl_legal (only legal moves)
        legal_cpls = [r["cpl"] for r in position_results if r["is_legal"]]
        expected_avg_cpl_legal = sum(legal_cpls) / len(legal_cpls) if legal_cpls else 10000

        if abs(summary["avg_cpl_legal"] - expected_avg_cpl_legal) > 0.1:
            issues.append(f"{player_id}: avg_cpl_legal calculation error")
            print(f"    ERROR: avg_cpl_legal {summary['avg_cpl_legal']:.2f} != {expected_avg_cpl_legal:.2f}")

    # Check 4: Verify results.json matches results_combined.json
    print("\n3. RESULTS.JSON vs RESULTS_COMBINED.JSON COMPARISON")
    print("-" * 80)

    for player_id in combined["results"]:
        if player_id not in results:
            issues.append(f"{player_id} in combined but not in results.json")
            print(f"ERROR: {player_id} in combined but not in results.json")
            continue

        combined_data = combined["results"][player_id]
        results_summary = results[player_id]["summary"]

        # Compare key metrics
        metrics_to_check = ["legal_pct", "best_pct", "avg_cpl"]

        for metric in metrics_to_check:
            if metric in combined_data and metric in results_summary:
                combined_val = combined_data[metric]
                results_val = results_summary[metric]

                if abs(combined_val - results_val) > 0.1:
                    issues.append(f"{player_id}: {metric} mismatch between files")
                    print(f"ERROR: {player_id} {metric}: combined={combined_val:.2f}, results={results_val:.2f}")

    print("\nPASS: results.json matches results_combined.json")

    # Sample detailed verification
    print("\n4. DETAILED SAMPLE VERIFICATION")
    print("-" * 80)

    # Pick a model with some illegal moves for detailed check
    sample_player = "gemini-2.5-flash (no thinking)"
    if sample_player in results:
        print(f"\nDetailed check for {sample_player}:")
        data = results[sample_player]
        pos_results = data["results"]
        summary = data["summary"]

        print(f"  Total positions: {len(pos_results)}")
        print(f"  Legal moves: {summary['legal_moves']}")
        print(f"  Illegal moves: {len(pos_results) - summary['legal_moves']}")

        # Show first illegal move
        illegal_moves = [r for r in pos_results if not r["is_legal"]]
        if illegal_moves:
            print(f"\n  First illegal move example:")
            ill = illegal_moves[0]
            print(f"    Position: {ill['position_idx']}")
            print(f"    eval_best (eval_before): {ill['eval_best']}")
            print(f"    CPL: {ill['cpl']}")
            print(f"    Expected CPL (eval_before + 5000): {ill['eval_best'] + 5000}")
            if abs(ill['cpl'] - (ill['eval_best'] + 5000)) > 1:
                issues.append(f"{sample_player}: Illegal move CPL calculation error")
                print(f"    ERROR: CPL mismatch!")
            else:
                print(f"    PASS: CPL = eval_before + 5000")

        # Manual avg_cpl calculation
        all_cpls = [r["cpl"] for r in pos_results]
        manual_avg = sum(all_cpls) / len(all_cpls)
        print(f"\n  Manual avg_cpl calculation:")
        print(f"    Sum of all CPLs: {sum(all_cpls):.2f}")
        print(f"    Number of positions: {len(all_cpls)}")
        print(f"    Manual average: {manual_avg:.2f}")
        print(f"    Stored avg_cpl: {summary['avg_cpl']:.2f}")
        if abs(manual_avg - summary['avg_cpl']) < 0.1:
            print(f"    PASS: Averages match")
        else:
            issues.append(f"{sample_player}: avg_cpl calculation error")
            print(f"    ERROR: Averages don't match!")

    # Final summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    if issues:
        print(f"\nERROR: Found {len(issues)} issues:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        return 1
    else:
        print("\nSUCCESS: All verification checks passed!")
        print("\nVerified:")
        print("  - Illegal CPL formula: eval_before + 5000")
        print("  - Legal CPL formula: max(0, eval_best - eval_model)")
        print("  - Summary statistics (avg_cpl, legal_pct, best_pct)")
        print("  - Consistency between results.json and results_combined.json")
        return 0

if __name__ == "__main__":
    exit(verify_results())
