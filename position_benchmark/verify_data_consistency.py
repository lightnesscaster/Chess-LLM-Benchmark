#!/usr/bin/env python3
"""
Data consistency verification script for position benchmark results.
Checks that all three result files have consistent data.
"""

import json
from pathlib import Path
from collections import defaultdict

def load_json(filepath):
    """Load and return JSON data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def verify_data_consistency():
    """Main verification function."""
    base_path = Path(__file__).parent

    # Load all three files
    print("Loading files...")
    equal_results = load_json(base_path / "equal_results.json")
    blunder_results = load_json(base_path / "results.json")
    combined_results = load_json(base_path / "results_combined.json")

    print("\n" + "="*80)
    print("POSITION BENCHMARK DATA CONSISTENCY REPORT")
    print("="*80)

    issues = []

    # 1. Check model counts across all three files
    print("\n1. MODEL COUNT VERIFICATION")
    print("-" * 80)

    equal_models = set(equal_results.keys())
    blunder_models = set(blunder_results.keys())
    combined_models = set(combined_results.get("results", {}).keys())

    print(f"equal_results.json: {len(equal_models)} models")
    print(f"results.json: {len(blunder_models)} models")
    print(f"results_combined.json: {len(combined_models)} models")

    # Check if all have 24 models
    for name, model_set, expected in [
        ("equal_results.json", equal_models, 24),
        ("results.json", blunder_models, 24),
        ("results_combined.json", combined_models, 24)
    ]:
        if len(model_set) != expected:
            issues.append(f"❌ {name} has {len(model_set)} models, expected {expected}")
        else:
            print(f"✓ {name} has exactly {expected} models")

    # Check if the same models are in all files
    all_models = equal_models | blunder_models | combined_models
    print(f"\nTotal unique models across all files: {len(all_models)}")
    print(f"Models: {sorted(all_models)}")

    # Find discrepancies
    only_equal = equal_models - blunder_models - combined_models
    only_blunder = blunder_models - equal_models - combined_models
    only_combined = combined_models - equal_models - blunder_models

    if only_equal:
        issues.append(f"❌ Models only in equal_results.json: {only_equal}")
    if only_blunder:
        issues.append(f"❌ Models only in results.json: {only_blunder}")
    if only_combined:
        issues.append(f"❌ Models only in results_combined.json: {only_combined}")

    missing_in_equal = (blunder_models | combined_models) - equal_models
    missing_in_blunder = (equal_models | combined_models) - blunder_models
    missing_in_combined = (equal_models | blunder_models) - combined_models

    if missing_in_equal:
        issues.append(f"❌ Models missing in equal_results.json: {missing_in_equal}")
    if missing_in_blunder:
        issues.append(f"❌ Models missing in results.json: {missing_in_blunder}")
    if missing_in_combined:
        issues.append(f"❌ Models missing in results_combined.json: {missing_in_combined}")

    if equal_models == blunder_models == combined_models:
        print("✓ All three files contain the same set of models")
    else:
        print("❌ Model sets differ between files")

    # 2. Check position counts for each model
    print("\n2. POSITION COUNT VERIFICATION")
    print("-" * 80)

    equal_position_issues = []
    blunder_position_issues = []

    for model in sorted(all_models):
        # Handle nested structure with 'results' array
        equal_data = equal_results.get(model, {})
        blunder_data = blunder_results.get(model, {})

        equal_count = len(equal_data.get('results', [])) if isinstance(equal_data, dict) else 0
        blunder_count = len(blunder_data.get('results', [])) if isinstance(blunder_data, dict) else 0

        status_equal = "✓" if equal_count == 100 else "❌"
        status_blunder = "✓" if blunder_count == 100 else "❌"

        if equal_count != 100 or blunder_count != 100:
            print(f"{status_equal} {model}: equal={equal_count}/100, blunder={blunder_count}/100")

            if equal_count != 100:
                equal_position_issues.append(f"{model}: {equal_count} positions")
            if blunder_count != 100:
                blunder_position_issues.append(f"{model}: {blunder_count} positions")

    if not equal_position_issues and not blunder_position_issues:
        print("✓ All models have exactly 100 positions in both files")
    else:
        if equal_position_issues:
            issues.append(f"❌ equal_results.json position count issues: {len(equal_position_issues)}")
            for issue in equal_position_issues[:5]:  # Show first 5
                print(f"  - {issue}")
        if blunder_position_issues:
            issues.append(f"❌ results.json position count issues: {len(blunder_position_issues)}")
            for issue in blunder_position_issues[:5]:  # Show first 5
                print(f"  - {issue}")

    # 3. Check required fields in per-position data
    print("\n3. REQUIRED FIELDS VERIFICATION")
    print("-" * 80)

    required_fields = ["position_idx", "fen", "model_move", "best_move", "cpl", "is_legal", "is_best"]

    field_issues = defaultdict(set)

    # Check equal_results.json
    for model in equal_models:
        model_data = equal_results.get(model, {})
        positions = model_data.get('results', []) if isinstance(model_data, dict) else []
        for pos_data in positions[:5]:  # Check first 5 positions
            for field in required_fields:
                if field not in pos_data:
                    field_issues[f"equal_results.json:{model}"].add(field)

    # Check results.json (blunder)
    for model in blunder_models:
        model_data = blunder_results.get(model, {})
        positions = model_data.get('results', []) if isinstance(model_data, dict) else []
        for pos_data in positions[:5]:  # Check first 5 positions
            for field in required_fields:
                if field not in pos_data:
                    field_issues[f"results.json:{model}"].add(field)

    if not field_issues:
        print("✓ All models have required fields in per-position data")
    else:
        print(f"❌ Found {len(field_issues)} models with missing fields")
        issues.append(f"❌ Missing required fields in {len(field_issues)} model entries")
        for key, missing_fields in list(field_issues.items())[:10]:
            print(f"  - {key}: missing {missing_fields}")

    # 4. Check for duplicate position_idx values
    print("\n4. POSITION INDEX UNIQUENESS VERIFICATION")
    print("-" * 80)

    duplicate_issues = []

    # Check equal_results.json
    for model in equal_models:
        model_data = equal_results.get(model, {})
        positions = model_data.get('results', []) if isinstance(model_data, dict) else []
        indices = [p.get("position_idx") for p in positions if "position_idx" in p]

        if len(indices) != len(set(indices)):
            duplicates = [idx for idx in set(indices) if indices.count(idx) > 1]
            duplicate_issues.append(f"equal_results.json:{model} has duplicate indices: {duplicates[:5]}")

    # Check results.json
    for model in blunder_models:
        model_data = blunder_results.get(model, {})
        positions = model_data.get('results', []) if isinstance(model_data, dict) else []
        indices = [p.get("position_idx") for p in positions if "position_idx" in p]

        if len(indices) != len(set(indices)):
            duplicates = [idx for idx in set(indices) if indices.count(idx) > 1]
            duplicate_issues.append(f"results.json:{model} has duplicate indices: {duplicates[:5]}")

    if not duplicate_issues:
        print("✓ All models have unique position_idx values (0-99)")
    else:
        print(f"❌ Found {len(duplicate_issues)} models with duplicate indices")
        issues.append(f"❌ Duplicate position indices in {len(duplicate_issues)} model entries")
        for issue in duplicate_issues[:10]:
            print(f"  - {issue}")

    # 5. Verify position_idx ranges (should be 0-99)
    print("\n5. POSITION INDEX RANGE VERIFICATION")
    print("-" * 80)

    range_issues = []

    for model in equal_models:
        model_data = equal_results.get(model, {})
        positions = model_data.get('results', []) if isinstance(model_data, dict) else []
        indices = [p.get("position_idx") for p in positions if "position_idx" in p]
        if indices:
            min_idx, max_idx = min(indices), max(indices)
            if min_idx != 0 or max_idx != 99:
                range_issues.append(f"equal_results.json:{model} range: [{min_idx}, {max_idx}]")

    for model in blunder_models:
        model_data = blunder_results.get(model, {})
        positions = model_data.get('results', []) if isinstance(model_data, dict) else []
        indices = [p.get("position_idx") for p in positions if "position_idx" in p]
        if indices:
            min_idx, max_idx = min(indices), max(indices)
            if min_idx != 0 or max_idx != 99:
                range_issues.append(f"results.json:{model} range: [{min_idx}, {max_idx}]")

    if not range_issues:
        print("✓ All models have position_idx in range [0, 99]")
    else:
        print(f"❌ Found {len(range_issues)} models with incorrect index ranges")
        issues.append(f"❌ Incorrect position_idx ranges in {len(range_issues)} model entries")
        for issue in range_issues[:10]:
            print(f"  - {issue}")

    # 6. Verify results_combined.json has summary stats for all models
    print("\n6. COMBINED RESULTS SUMMARY STATS VERIFICATION")
    print("-" * 80)

    required_stats = ["legal_pct", "best_pct", "avg_cpl", "avoided_pct", "median_cpl", "type"]

    stats_issues = []

    for model in combined_models:
        model_data = combined_results.get("results", {}).get(model, {})
        missing_stats = [stat for stat in required_stats if stat not in model_data]
        if missing_stats:
            stats_issues.append(f"{model}: missing {missing_stats}")

    if not stats_issues:
        print("✓ All models in results_combined.json have required summary stats")
    else:
        print(f"❌ Found {len(stats_issues)} models with missing stats")
        issues.append(f"❌ Missing summary stats in {len(stats_issues)} models")
        for issue in stats_issues[:10]:
            print(f"  - {issue}")

    # FINAL SUMMARY
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)

    if not issues:
        print("\n✅ ALL CHECKS PASSED - Data is consistent across all three files")
    else:
        print(f"\n❌ FOUND {len(issues)} ISSUES:")
        for i, issue in enumerate(issues, 1):
            print(f"{i}. {issue}")

    print("\n" + "="*80)

    return len(issues) == 0

if __name__ == "__main__":
    success = verify_data_consistency()
    exit(0 if success else 1)
