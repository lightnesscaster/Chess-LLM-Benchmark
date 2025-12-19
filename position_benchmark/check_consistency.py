#!/usr/bin/env python3
"""
Check data consistency of position benchmark results files.
"""
import json
from collections import defaultdict
from pathlib import Path

def load_json(filepath):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def check_consistency():
    """Check data consistency across all three results files."""
    base_dir = Path(__file__).parent

    # Load all three files
    print("Loading files...")
    equal_results = load_json(base_dir / "equal_results.json")
    results = load_json(base_dir / "results.json")
    combined = load_json(base_dir / "results_combined.json")

    issues = []

    # Extract model lists
    equal_models = set(equal_results.keys())
    results_models = set(results.keys())
    combined_models = set(combined.get("results", {}).keys())

    print(f"\n{'='*80}")
    print("1. MODEL COUNT CHECK")
    print(f"{'='*80}")
    print(f"equal_results.json: {len(equal_models)} models")
    print(f"results.json: {len(results_models)} models")
    print(f"results_combined.json: {len(combined_models)} models")

    # Check if all have 24 models
    if len(equal_models) != 24:
        issues.append(f"equal_results.json has {len(equal_models)} models, expected 24")
    if len(results_models) != 24:
        issues.append(f"results.json has {len(results_models)} models, expected 24")
    if len(combined_models) != 24:
        issues.append(f"results_combined.json has {len(combined_models)} models, expected 24")

    # Check model consistency across files
    print(f"\n{'='*80}")
    print("2. MODEL NAME CONSISTENCY CHECK")
    print(f"{'='*80}")

    all_models = equal_models | results_models | combined_models
    print(f"Total unique models across all files: {len(all_models)}")

    only_in_equal = equal_models - results_models - combined_models
    only_in_results = results_models - equal_models - combined_models
    only_in_combined = combined_models - equal_models - results_models

    if only_in_equal:
        issues.append(f"Models only in equal_results.json: {only_in_equal}")
        print(f"ISSUE: Models only in equal_results.json: {only_in_equal}")

    if only_in_results:
        issues.append(f"Models only in results.json: {only_in_results}")
        print(f"ISSUE: Models only in results.json: {only_in_results}")

    if only_in_combined:
        issues.append(f"Models only in results_combined.json: {only_in_combined}")
        print(f"ISSUE: Models only in results_combined.json: {only_in_combined}")

    if not (only_in_equal or only_in_results or only_in_combined):
        print("✓ All files have the same models")

    # Check position counts
    print(f"\n{'='*80}")
    print("3. POSITION COUNT CHECK")
    print(f"{'='*80}")

    print("\nequal_results.json (should be 100 per model):")
    equal_position_counts = {}
    for model, data in equal_results.items():
        if isinstance(data, dict) and 'results' in data:
            count = len(data['results'])
            equal_position_counts[model] = count
            if count != 100:
                issues.append(f"equal_results.json - {model}: {count} positions (expected 100)")
                print(f"  ISSUE: {model}: {count} positions")
            else:
                print(f"  ✓ {model}: {count} positions")
        else:
            issues.append(f"equal_results.json - {model}: missing 'results' field")
            print(f"  ISSUE: {model}: missing 'results' field, has: {data.keys() if isinstance(data, dict) else type(data)}")

    print("\nresults.json (should be 100 per model):")
    results_position_counts = {}
    for model, data in results.items():
        if isinstance(data, dict) and 'results' in data:
            count = len(data['results'])
            results_position_counts[model] = count
            if count != 100:
                issues.append(f"results.json - {model}: {count} positions (expected 100)")
                print(f"  ISSUE: {model}: {count} positions")
            else:
                print(f"  ✓ {model}: {count} positions")
        else:
            issues.append(f"results.json - {model}: missing 'results' field")
            print(f"  ISSUE: {model}: missing 'results' field, has: {data.keys() if isinstance(data, dict) else type(data)}")

    # Check required fields in per-position data
    print(f"\n{'='*80}")
    print("4. PER-POSITION DATA FIELD CHECK")
    print(f"{'='*80}")

    required_fields = ['position_idx', 'fen', 'model_move', 'best_move', 'cpl', 'is_legal', 'is_best']

    print("\nChecking equal_results.json...")
    for model, data in equal_results.items():
        if isinstance(data, dict) and 'results' in data:
            positions = data['results']
            if positions:
                sample = positions[0]
                missing = [f for f in required_fields if f not in sample]
                if missing:
                    issues.append(f"equal_results.json - {model}: missing fields {missing}")
                    print(f"  ISSUE: {model}: missing fields {missing}")
                else:
                    print(f"  ✓ {model}: all required fields present")

    print("\nChecking results.json...")
    for model, data in results.items():
        if isinstance(data, dict) and 'results' in data:
            positions = data['results']
            if positions:
                sample = positions[0]
                missing = [f for f in required_fields if f not in sample]
                if missing:
                    issues.append(f"results.json - {model}: missing fields {missing}")
                    print(f"  ISSUE: {model}: missing fields {missing}")
                else:
                    print(f"  ✓ {model}: all required fields present")

    # Check results_combined summary stats
    print(f"\n{'='*80}")
    print("5. SUMMARY STATS CHECK (results_combined.json)")
    print(f"{'='*80}")

    combined_fields = ['legal_pct', 'best_pct', 'avg_cpl', 'avoided_pct', 'median_cpl', 'type']

    for model, stats in combined.get("results", {}).items():
        missing = [f for f in combined_fields if f not in stats]
        if missing:
            issues.append(f"results_combined.json - {model}: missing fields {missing}")
            print(f"  ISSUE: {model}: missing fields {missing}")
        else:
            print(f"  ✓ {model}: all summary fields present")

    # Check for duplicate position indices
    print(f"\n{'='*80}")
    print("6. DUPLICATE POSITION INDEX CHECK")
    print(f"{'='*80}")

    print("\nChecking equal_results.json for duplicates...")
    for model, data in equal_results.items():
        if isinstance(data, dict) and 'results' in data:
            indices = [p.get('position_idx') for p in data['results']]
            if len(indices) != len(set(indices)):
                duplicates = [idx for idx in set(indices) if indices.count(idx) > 1]
                issues.append(f"equal_results.json - {model}: duplicate position indices {duplicates}")
                print(f"  ISSUE: {model}: duplicate position indices {duplicates}")
            else:
                print(f"  ✓ {model}: no duplicates")

    print("\nChecking results.json for duplicates...")
    for model, data in results.items():
        if isinstance(data, dict) and 'results' in data:
            indices = [p.get('position_idx') for p in data['results']]
            if len(indices) != len(set(indices)):
                duplicates = [idx for idx in set(indices) if indices.count(idx) > 1]
                issues.append(f"results.json - {model}: duplicate position indices {duplicates}")
                print(f"  ISSUE: {model}: duplicate position indices {duplicates}")
            else:
                print(f"  ✓ {model}: no duplicates")

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    if issues:
        print(f"\n❌ Found {len(issues)} issues:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\n✓ All consistency checks passed!")

    return len(issues) == 0

if __name__ == "__main__":
    success = check_consistency()
    exit(0 if success else 1)
