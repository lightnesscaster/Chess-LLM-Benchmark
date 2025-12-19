#!/usr/bin/env python3
"""
Detailed data quality report for position benchmark results.
Provides additional insights beyond basic consistency checks.
"""

import json
from pathlib import Path
from collections import defaultdict

def load_json(filepath):
    """Load and return JSON data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def detailed_data_report():
    """Generate detailed data quality report."""
    base_path = Path(__file__).parent

    # Load all three files
    print("Loading files...")
    equal_results = load_json(base_path / "equal_results.json")
    blunder_results = load_json(base_path / "results.json")
    combined_results = load_json(base_path / "results_combined.json")

    print("\n" + "="*80)
    print("DETAILED DATA QUALITY REPORT")
    print("="*80)

    models = sorted(equal_results.keys())

    # 1. Verify consistency between per-position and summary stats
    print("\n1. SUMMARY STATS vs PER-POSITION DATA CONSISTENCY")
    print("-" * 80)

    stats_mismatches = []

    for model in models:
        # Get per-position data
        equal_data = equal_results[model]
        blunder_data = blunder_results[model]

        equal_positions = equal_data.get('results', [])
        blunder_positions = blunder_data.get('results', [])

        # Calculate stats from per-position data
        equal_legal = sum(1 for p in equal_positions if p.get('is_legal'))
        equal_best = sum(1 for p in equal_positions if p.get('is_best'))

        blunder_legal = sum(1 for p in blunder_positions if p.get('is_legal'))
        blunder_best = sum(1 for p in blunder_positions if p.get('is_best'))
        blunder_avoided = sum(1 for p in blunder_positions if p.get('avoided_blunder'))

        # Get summary stats
        equal_summary = equal_data.get('summary', {})
        blunder_summary = blunder_data.get('summary', {})
        combined_stats = combined_results['results'].get(model, {})

        # Verify equal_results.json consistency
        if equal_summary.get('legal_moves') != equal_legal:
            stats_mismatches.append(
                f"equal_results.json:{model} legal_moves: summary={equal_summary.get('legal_moves')} "
                f"vs calculated={equal_legal}"
            )
        if equal_summary.get('best_moves') != equal_best:
            stats_mismatches.append(
                f"equal_results.json:{model} best_moves: summary={equal_summary.get('best_moves')} "
                f"vs calculated={equal_best}"
            )

        # Verify results.json consistency
        if blunder_summary.get('legal_moves') != blunder_legal:
            stats_mismatches.append(
                f"results.json:{model} legal_moves: summary={blunder_summary.get('legal_moves')} "
                f"vs calculated={blunder_legal}"
            )
        if blunder_summary.get('best_moves') != blunder_best:
            stats_mismatches.append(
                f"results.json:{model} best_moves: summary={blunder_summary.get('best_moves')} "
                f"vs calculated={blunder_best}"
            )
        if blunder_summary.get('avoided_blunders') != blunder_avoided:
            stats_mismatches.append(
                f"results.json:{model} avoided_blunders: summary={blunder_summary.get('avoided_blunders')} "
                f"vs calculated={blunder_avoided}"
            )

    if not stats_mismatches:
        print("✓ All summary stats match calculated values from per-position data")
    else:
        print(f"❌ Found {len(stats_mismatches)} mismatches:")
        for mismatch in stats_mismatches[:10]:
            print(f"  - {mismatch}")

    # 2. Check for data type consistency
    print("\n2. DATA TYPE CONSISTENCY")
    print("-" * 80)

    type_issues = []

    for model in models:
        equal_positions = equal_results[model].get('results', [])
        blunder_positions = blunder_results[model].get('results', [])

        for i, pos in enumerate(equal_positions[:10]):  # Check first 10
            if not isinstance(pos.get('position_idx'), int):
                type_issues.append(f"equal_results.json:{model}[{i}].position_idx is not int")
            if not isinstance(pos.get('fen'), str):
                type_issues.append(f"equal_results.json:{model}[{i}].fen is not str")
            if not isinstance(pos.get('cpl'), (int, float)):
                type_issues.append(f"equal_results.json:{model}[{i}].cpl is not numeric")
            if not isinstance(pos.get('is_legal'), bool):
                type_issues.append(f"equal_results.json:{model}[{i}].is_legal is not bool")
            if not isinstance(pos.get('is_best'), bool):
                type_issues.append(f"equal_results.json:{model}[{i}].is_best is not bool")

        for i, pos in enumerate(blunder_positions[:10]):  # Check first 10
            if not isinstance(pos.get('position_idx'), int):
                type_issues.append(f"results.json:{model}[{i}].position_idx is not int")
            if not isinstance(pos.get('avoided_blunder'), bool):
                type_issues.append(f"results.json:{model}[{i}].avoided_blunder is not bool")

    if not type_issues:
        print("✓ All data types are consistent")
    else:
        print(f"❌ Found {len(type_issues)} type inconsistencies:")
        for issue in type_issues[:10]:
            print(f"  - {issue}")

    # 3. Check for logical consistency
    print("\n3. LOGICAL CONSISTENCY")
    print("-" * 80)

    logic_issues = []

    for model in models:
        equal_positions = equal_results[model].get('results', [])
        blunder_positions = blunder_results[model].get('results', [])

        for i, pos in enumerate(equal_positions):
            # If is_best is True, is_legal must also be True
            if pos.get('is_best') and not pos.get('is_legal'):
                logic_issues.append(
                    f"equal_results.json:{model}[{i}]: is_best=True but is_legal=False"
                )
            # If is_best is True, cpl should be 0
            if pos.get('is_best') and pos.get('cpl') != 0:
                logic_issues.append(
                    f"equal_results.json:{model}[{i}]: is_best=True but cpl={pos.get('cpl')} (expected 0)"
                )
            # If is_legal is False, cpl should be very high (likely 5000+)
            if not pos.get('is_legal') and pos.get('cpl', 0) < 5000:
                logic_issues.append(
                    f"equal_results.json:{model}[{i}]: is_legal=False but cpl={pos.get('cpl')} (expected >=5000)"
                )

        for i, pos in enumerate(blunder_positions):
            # Same checks for blunder positions
            if pos.get('is_best') and not pos.get('is_legal'):
                logic_issues.append(
                    f"results.json:{model}[{i}]: is_best=True but is_legal=False"
                )
            if pos.get('is_best') and pos.get('cpl') != 0:
                logic_issues.append(
                    f"results.json:{model}[{i}]: is_best=True but cpl={pos.get('cpl')} (expected 0)"
                )
            # avoided_blunder logic: if the move is different from blunder_move, it should be avoided
            # Note: This is tricky because blunder_move might be empty string

    if not logic_issues:
        print("✓ All logical relationships are consistent")
    else:
        print(f"❌ Found {len(logic_issues)} logical inconsistencies:")
        for issue in logic_issues[:15]:
            print(f"  - {issue}")

    # 4. Check FEN validity (basic check)
    print("\n4. FEN FORMAT VALIDATION")
    print("-" * 80)

    fen_issues = []

    for model in models[:3]:  # Check first 3 models only
        equal_positions = equal_results[model].get('results', [])
        blunder_positions = blunder_results[model].get('results', [])

        for i, pos in enumerate(equal_positions):
            fen = pos.get('fen', '')
            # Basic FEN check: should have 6 space-separated fields
            if len(fen.split()) != 6:
                fen_issues.append(f"equal_results.json:{model}[{i}]: invalid FEN format")

        for i, pos in enumerate(blunder_positions):
            fen = pos.get('fen', '')
            if len(fen.split()) != 6:
                fen_issues.append(f"results.json:{model}[{i}]: invalid FEN format")

    if not fen_issues:
        print("✓ FEN formats appear valid (sampled)")
    else:
        print(f"❌ Found {len(fen_issues)} FEN format issues:")
        for issue in fen_issues[:10]:
            print(f"  - {issue}")

    # 5. Statistics summary
    print("\n5. DATASET STATISTICS SUMMARY")
    print("-" * 80)

    # Group models by type
    llm_models = []
    engine_models = []

    for model in models:
        model_type = combined_results['results'][model].get('type', 'unknown')
        if model_type == 'llm':
            llm_models.append(model)
        elif model_type == 'engine':
            engine_models.append(model)

    print(f"Total models: {len(models)}")
    print(f"  - LLMs: {len(llm_models)}")
    print(f"  - Engines: {len(engine_models)}")
    print(f"\nPositions per dataset: 100")
    print(f"Total data points per file: {len(models)} models × 100 positions = {len(models) * 100}")

    # Best performers
    print("\n6. TOP PERFORMERS (by best move percentage in blunder positions)")
    print("-" * 80)

    model_best_pcts = [(m, combined_results['results'][m]['best_pct']) for m in models]
    model_best_pcts.sort(key=lambda x: x[1], reverse=True)

    for i, (model, best_pct) in enumerate(model_best_pcts[:10], 1):
        model_type = combined_results['results'][model].get('type', 'unknown')
        legal_pct = combined_results['results'][model]['legal_pct']
        print(f"{i}. {model:45} | type={model_type:6} | best={best_pct:5.1f}% | legal={legal_pct:5.1f}%")

    # Worst performers by illegal moves
    print("\n7. MODELS WITH MOST ILLEGAL MOVES (in blunder positions)")
    print("-" * 80)

    model_illegal_pcts = [(m, 100 - combined_results['results'][m]['legal_pct']) for m in models]
    model_illegal_pcts.sort(key=lambda x: x[1], reverse=True)

    for i, (model, illegal_pct) in enumerate(model_illegal_pcts[:10], 1):
        model_type = combined_results['results'][model].get('type', 'unknown')
        legal_pct = 100 - illegal_pct
        print(f"{i}. {model:45} | type={model_type:6} | illegal={illegal_pct:5.1f}% | legal={legal_pct:5.1f}%")

    print("\n" + "="*80)
    print("END OF REPORT")
    print("="*80)

if __name__ == "__main__":
    detailed_data_report()
