#!/usr/bin/env python3
"""
Quick verification script for position benchmark data integrity.
Run this anytime to verify data consistency.
"""

import json
from pathlib import Path

def quick_verify():
    """Quick data integrity check."""
    base_path = Path(__file__).parent

    print("Position Benchmark Data Quick Verification")
    print("=" * 60)

    try:
        # Load files
        with open(base_path / "equal_results.json") as f:
            equal_results = json.load(f)
        with open(base_path / "results.json") as f:
            blunder_results = json.load(f)
        with open(base_path / "results_combined.json") as f:
            combined_results = json.load(f)

        issues = []

        # Check 1: Model counts
        equal_models = set(equal_results.keys())
        blunder_models = set(blunder_results.keys())
        combined_models = set(combined_results.get("results", {}).keys())

        if len(equal_models) != 24:
            issues.append(f"equal_results.json has {len(equal_models)} models (expected 24)")
        if len(blunder_models) != 24:
            issues.append(f"results.json has {len(blunder_models)} models (expected 24)")
        if len(combined_models) != 24:
            issues.append(f"results_combined.json has {len(combined_models)} models (expected 24)")

        if equal_models != blunder_models or equal_models != combined_models:
            issues.append("Model sets differ between files")

        # Check 2: Position counts
        for model in equal_models:
            equal_count = len(equal_results[model].get('results', []))
            blunder_count = len(blunder_results[model].get('results', []))

            if equal_count != 100:
                issues.append(f"{model}: equal_results has {equal_count} positions (expected 100)")
            if blunder_count != 100:
                issues.append(f"{model}: results has {blunder_count} positions (expected 100)")

        # Check 3: Required fields (sample check)
        required_fields = ["position_idx", "fen", "model_move", "best_move", "cpl", "is_legal", "is_best"]

        for model in list(equal_models)[:3]:  # Sample 3 models
            sample_pos = equal_results[model]['results'][0]
            missing = [f for f in required_fields if f not in sample_pos]
            if missing:
                issues.append(f"{model}: missing fields {missing}")

        # Print results
        if not issues:
            print("✅ ALL CHECKS PASSED")
            print(f"\n24 models × 100 positions × 2 datasets = 4,800 verified records")
            print("\nFiles are consistent and ready for use.")
        else:
            print("❌ ISSUES FOUND:")
            for issue in issues:
                print(f"  - {issue}")
            print(f"\nTotal issues: {len(issues)}")
            return False

        print("=" * 60)
        return True

    except FileNotFoundError as e:
        print(f"❌ ERROR: File not found - {e}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON - {e}")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    import sys
    success = quick_verify()
    sys.exit(0 if success else 1)
