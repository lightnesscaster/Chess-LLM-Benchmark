#!/usr/bin/env python3
"""
Regression analysis: predict model strength from reasoning tokens per move.

Groups models by family, compares thinking vs non-thinking variants using
actual tokens/move rather than categorical effort levels.
"""

import json
import re
import numpy as np


def load_data():
    from firebase_client import get_firestore_client, RESULTS_COLLECTION, RATINGS_COLLECTION
    db = get_firestore_client()

    results = {}
    for doc in db.collection(RESULTS_COLLECTION).stream():
        results[doc.id] = doc.to_dict()
    print(f"Loaded {len(results)} game results from Firestore")

    ratings = {}
    for doc in db.collection(RATINGS_COLLECTION).stream():
        ratings[doc.id] = doc.to_dict()
    print(f"Loaded {len(ratings)} ratings from Firestore")

    return results, ratings


def get_model_family(player_id):
    """Extract base model family from player_id, stripping effort/thinking suffixes."""
    # Remove parenthetical suffixes like (high), (no thinking), (medium), (minimal), (default)
    base = re.sub(r'\s*\(.*?\)\s*$', '', player_id).strip()
    return base


def compute_tokens_per_move(results):
    """Compute average completion tokens per move for each player."""
    player_tokens = {}  # player_id -> list of (completion_tokens, moves)

    for game_id, g in results.items():
        for side in ['white', 'black']:
            player_id = g.get(f'{side}_id')
            tokens = g.get(f'tokens_{side}')
            moves = g.get(f'total_moves_{side}', 0)

            if not player_id or not tokens or moves == 0:
                continue

            comp = tokens.get('completion_tokens', 0)
            if comp == 0:
                continue

            if player_id not in player_tokens:
                player_tokens[player_id] = []
            player_tokens[player_id].append((comp, moves))

    # Compute averages
    avg_tokens = {}
    for player_id, games in player_tokens.items():
        total_comp = sum(c for c, m in games)
        total_moves = sum(m for c, m in games)
        avg_tokens[player_id] = total_comp / total_moves if total_moves > 0 else 0

    return avg_tokens


def main():
    results, ratings = load_data()
    avg_tokens = compute_tokens_per_move(results)

    # Build dataset: only include players that have both rating and token data
    data = []
    for player_id, rating_info in ratings.items():
        rating = rating_info.get('rating', 0)
        games = rating_info.get('games_played', 0)
        rd = rating_info.get('rating_deviation', 350)
        tokens = avg_tokens.get(player_id)

        if tokens is None or rd >= 150:
            continue

        family = get_model_family(player_id)
        data.append({
            'player_id': player_id,
            'family': family,
            'rating': rating,
            'tokens_per_move': tokens,
            'games': games,
            'rd': rd,
        })

    # Sort by family then tokens
    data.sort(key=lambda x: (x['family'], x['tokens_per_move']))

    # Print all models with their data
    print("=" * 100)
    print(f"{'Player':<40} {'Family':<20} {'Rating':>7} {'Tok/Move':>10} {'Games':>6} {'RD':>6}")
    print("=" * 100)
    for d in data:
        print(f"{d['player_id']:<40} {d['family']:<20} {d['rating']:>7.0f} {d['tokens_per_move']:>10.0f} {d['games']:>6} {d['rd']:>6.0f}")

    # Find families with multiple variants
    from collections import defaultdict
    families = defaultdict(list)
    for d in data:
        families[d['family']].append(d)

    print("\n\n" + "=" * 100)
    print("MODEL FAMILIES WITH MULTIPLE REASONING VARIANTS")
    print("=" * 100)

    family_data = []  # For regression: (family, tokens_per_move, rating) pairs

    for family, variants in sorted(families.items()):
        if len(variants) < 2:
            continue

        variants.sort(key=lambda x: x['tokens_per_move'])
        print(f"\n{family}:")
        for v in variants:
            marker = " *" if v['rd'] > 150 else ""
            print(f"  {v['player_id']:<40} rating={v['rating']:>7.0f}  tok/move={v['tokens_per_move']:>8.0f}  games={v['games']}{marker}")
            family_data.append((family, v['tokens_per_move'], v['rating']))

        # Show delta from lowest to highest tokens variant
        low = variants[0]
        high = variants[-1]
        token_ratio = high['tokens_per_move'] / low['tokens_per_move'] if low['tokens_per_move'] > 0 else float('inf')
        rating_delta = high['rating'] - low['rating']
        print(f"  -> {token_ratio:.1f}x more tokens = {rating_delta:+.0f} rating ({low['player_id']} -> {high['player_id']})")

    # Regression: within each family, does more tokens = higher rating?
    print("\n\n" + "=" * 100)
    print("REGRESSION: LOG(TOKENS/MOVE) vs RATING")
    print("=" * 100)

    # Collect within-family deltas for paired analysis
    deltas = []  # (log_token_ratio, rating_delta, family)
    for family, variants in sorted(families.items()):
        if len(variants) < 2:
            continue
        variants.sort(key=lambda x: x['tokens_per_move'])
        base = variants[0]
        for v in variants[1:]:
            if base['tokens_per_move'] > 0 and v['tokens_per_move'] > 0:
                log_ratio = np.log(v['tokens_per_move'] / base['tokens_per_move'])
                delta_r = v['rating'] - base['rating']
                deltas.append((log_ratio, delta_r, family))

    if not deltas:
        print("Not enough data for regression")
        return

    log_ratios = np.array([d[0] for d in deltas])
    rating_deltas = np.array([d[1] for d in deltas])
    n = len(deltas)

    # Also collect absolute values for models that use absolute tokens (not ratios)
    abs_data = []  # (log_tokens, rating, family) for each variant
    for family, variants in sorted(families.items()):
        if len(variants) < 2:
            continue
        for v in variants:
            if v['tokens_per_move'] > 0:
                abs_data.append((np.log(v['tokens_per_move']), v['rating'], family))

    def adj_r2(r2, n, k):
        """Adjusted R² penalizing extra parameters."""
        if n - k - 1 <= 0:
            return float('nan')
        return 1 - (1 - r2) * (n - 1) / (n - k - 1)

    def leave_one_out_mse(X, y):
        """LOO cross-validation MSE."""
        mse = 0
        for i in range(len(y)):
            mask = np.ones(len(y), dtype=bool)
            mask[i] = False
            c = np.linalg.lstsq(X[mask], y[mask], rcond=None)[0]
            pred = X[i] @ c
            mse += (y[i] - pred) ** 2
        return mse / len(y)

    def aic(ss_res, n, k):
        """AIC (lower is better)."""
        if ss_res <= 0 or n <= 0:
            return float('inf')
        return n * np.log(ss_res / n) + 2 * k

    print(f"\nPaired analysis: {n} pairs from {len([f for f, vs in families.items() if len(vs) >= 2])} families")
    print(f"\n{'='*90}")
    print("COMPARING REGRESSION FORMULAS (paired: rating_delta ~ f(token_ratio))")
    print(f"{'='*90}")

    ss_tot = np.sum((rating_deltas - np.mean(rating_deltas)) ** 2)
    results = []

    # --- Model 1: Intercept only (baseline) ---
    mean_delta = np.mean(rating_deltas)
    ss_res = np.sum((rating_deltas - mean_delta) ** 2)
    r2 = 0.0
    X_m1 = np.ones((n, 1))
    loo = leave_one_out_mse(X_m1, rating_deltas)
    results.append(("Baseline (mean)", f"Δ = {mean_delta:.0f}", 0, r2, adj_r2(r2, n, 0), loo, aic(ss_res, n, 1)))

    # --- Model 2: Linear in log(ratio), no intercept ---
    beta = np.sum(log_ratios * rating_deltas) / np.sum(log_ratios ** 2)
    pred = beta * log_ratios
    ss_res = np.sum((rating_deltas - pred) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    X_m2 = log_ratios.reshape(-1, 1)
    loo = leave_one_out_mse(X_m2, rating_deltas)
    results.append(("log(ratio), no intercept", f"Δ = {beta:.1f} * ln(r)", 1, r2, adj_r2(r2, n, 1), loo, aic(ss_res, n, 1)))

    # --- Model 3: Linear in log(ratio), with intercept ---
    X_m3 = np.column_stack([np.ones(n), log_ratios])
    c3 = np.linalg.lstsq(X_m3, rating_deltas, rcond=None)[0]
    pred3 = X_m3 @ c3
    ss_res3 = np.sum((rating_deltas - pred3) ** 2)
    r2_3 = 1 - ss_res3 / ss_tot if ss_tot > 0 else 0
    loo3 = leave_one_out_mse(X_m3, rating_deltas)
    results.append(("log(ratio) + intercept", f"Δ = {c3[0]:.0f} + {c3[1]:.0f} * ln(r)", 2, r2_3, adj_r2(r2_3, n, 2), loo3, aic(ss_res3, n, 2)))

    # --- Model 4: sqrt(ratio) ---
    sqrt_ratios = np.sqrt(np.exp(log_ratios))
    X_m4 = np.column_stack([np.ones(n), sqrt_ratios])
    c4 = np.linalg.lstsq(X_m4, rating_deltas, rcond=None)[0]
    pred4 = X_m4 @ c4
    ss_res4 = np.sum((rating_deltas - pred4) ** 2)
    r2_4 = 1 - ss_res4 / ss_tot if ss_tot > 0 else 0
    loo4 = leave_one_out_mse(X_m4, rating_deltas)
    results.append(("sqrt(ratio) + intercept", f"Δ = {c4[0]:.0f} + {c4[1]:.1f} * sqrt(r)", 2, r2_4, adj_r2(r2_4, n, 2), loo4, aic(ss_res4, n, 2)))

    # --- Model 5: Linear in ratio (no log) ---
    raw_ratios = np.exp(log_ratios)
    X_m5 = np.column_stack([np.ones(n), raw_ratios])
    c5 = np.linalg.lstsq(X_m5, rating_deltas, rcond=None)[0]
    pred5 = X_m5 @ c5
    ss_res5 = np.sum((rating_deltas - pred5) ** 2)
    r2_5 = 1 - ss_res5 / ss_tot if ss_tot > 0 else 0
    loo5 = leave_one_out_mse(X_m5, rating_deltas)
    results.append(("linear ratio + intercept", f"Δ = {c5[0]:.0f} + {c5[1]:.2f} * r", 2, r2_5, adj_r2(r2_5, n, 2), loo5, aic(ss_res5, n, 2)))

    # --- Model 6: log(ratio) + log(ratio)^2 (quadratic in log space) ---
    X_m6 = np.column_stack([np.ones(n), log_ratios, log_ratios ** 2])
    c6 = np.linalg.lstsq(X_m6, rating_deltas, rcond=None)[0]
    pred6 = X_m6 @ c6
    ss_res6 = np.sum((rating_deltas - pred6) ** 2)
    r2_6 = 1 - ss_res6 / ss_tot if ss_tot > 0 else 0
    loo6 = leave_one_out_mse(X_m6, rating_deltas)
    results.append(("log(ratio) quadratic", f"Δ = {c6[0]:.0f} + {c6[1]:.0f}*ln(r) + {c6[2]:.1f}*ln(r)²", 3, r2_6, adj_r2(r2_6, n, 3), loo6, aic(ss_res6, n, 3)))

    # --- Model 7: Power law: rating_delta ~ a * ratio^b (fit via log-log on positive deltas) ---
    # Only works for positive deltas
    pos_mask = rating_deltas > 0
    if np.sum(pos_mask) >= 3:
        lr_pos = log_ratios[pos_mask]
        rd_pos = np.log(rating_deltas[pos_mask])
        X_m7 = np.column_stack([np.ones(len(lr_pos)), lr_pos])
        c7 = np.linalg.lstsq(X_m7, rd_pos, rcond=None)[0]
        pred7_all = np.exp(c7[0]) * np.exp(log_ratios) ** c7[1]
        ss_res7 = np.sum((rating_deltas - pred7_all) ** 2)
        r2_7 = 1 - ss_res7 / ss_tot if ss_tot > 0 else 0
        results.append(("power law (a * r^b)", f"Δ = {np.exp(c7[0]):.0f} * r^{c7[1]:.2f}", 2, r2_7, adj_r2(r2_7, n, 2), float('nan'), aic(ss_res7, n, 2)))

    # --- Model 8: Absolute tokens (not ratio) predicting absolute rating ---
    abs_log_tok = np.array([d[0] for d in abs_data])
    abs_ratings = np.array([d[1] for d in abs_data])
    n_abs = len(abs_data)
    ss_tot_abs = np.sum((abs_ratings - np.mean(abs_ratings)) ** 2)
    X_m8 = np.column_stack([np.ones(n_abs), abs_log_tok])
    c8 = np.linalg.lstsq(X_m8, abs_ratings, rcond=None)[0]
    pred8 = X_m8 @ c8
    ss_res8 = np.sum((abs_ratings - pred8) ** 2)
    r2_8 = 1 - ss_res8 / ss_tot_abs if ss_tot_abs > 0 else 0
    loo8 = leave_one_out_mse(X_m8, abs_ratings)
    results.append(("ABSOLUTE: log(tok) -> rating", f"R = {c8[0]:.0f} + {c8[1]:.0f} * ln(tok)", 2, r2_8, adj_r2(r2_8, n_abs, 2), loo8, aic(ss_res8, n_abs, 2)))

    # Print comparison table
    print(f"\n{'Model':<30} {'Formula':<40} {'k':>3} {'R²':>7} {'Adj R²':>7} {'LOO MSE':>12} {'AIC':>8}")
    print("-" * 110)
    for name, formula, k, r2, ar2, loo, a in results:
        loo_str = f"{loo:>12.0f}" if not np.isnan(loo) else "         N/A"
        print(f"{name:<30} {formula:<40} {k:>3} {r2:>7.3f} {ar2:>7.3f} {loo_str} {a:>8.1f}")

    # Print best model's predictions
    print(f"\n{'='*90}")
    print("BEST MODEL PREDICTIONS (by adjusted R²)")
    print(f"{'='*90}")

    # Find best by adj R² among paired models (exclude absolute)
    paired_results = [r for r in results if not r[0].startswith("ABSOLUTE")]
    best = max(paired_results, key=lambda r: r[4] if not np.isnan(r[4]) else -999)
    print(f"\nBest: {best[0]} — {best[1]}  (Adj R² = {best[4]:.3f})")

    # Recompute predictions for each pair using the best model
    # Use model 2 (no intercept) or 3 (with intercept) based on which won
    best_name = best[0]
    print(f"\n{'Family':<25} {'Low variant':<35} {'High variant':<35} {'Tok ratio':>10} {'Actual Δ':>9} {'Pred Δ':>9} {'Error':>9}")
    print("-" * 135)

    for lr, rd, fam in deltas:
        fam_variants = sorted(families[fam], key=lambda x: x['tokens_per_move'])
        low_name = fam_variants[0]['player_id']
        high_name = "?"
        for v in fam_variants[1:]:
            if abs(np.log(v['tokens_per_move'] / fam_variants[0]['tokens_per_move']) - lr) < 0.01:
                high_name = v['player_id']
                break

        if best_name == "log(ratio), no intercept":
            pred = beta * lr
        elif best_name == "log(ratio) + intercept":
            pred = c3[0] + c3[1] * lr
        elif best_name == "sqrt(ratio) + intercept":
            pred = c4[0] + c4[1] * np.sqrt(np.exp(lr))
        elif best_name == "linear ratio + intercept":
            pred = c5[0] + c5[1] * np.exp(lr)
        elif best_name == "log(ratio) quadratic":
            pred = c6[0] + c6[1] * lr + c6[2] * lr ** 2
        elif best_name == "power law (a * r^b)":
            pred = np.exp(c7[0]) * np.exp(lr) ** c7[1]
        else:
            pred = mean_delta

        print(f"{fam:<25} {low_name:<35} {high_name:<35} {np.exp(lr):>8.1f}x {rd:>+9.0f} {pred:>+9.0f} {rd - pred:>+9.0f}")

    # =====================================================================
    # PREDICTIVE ACCURACY: predict higher variant's absolute rating
    # using lower variant's rating + token scaling
    # Use leave-one-out to avoid overfitting
    # =====================================================================
    print(f"\n\n{'='*90}")
    print("PREDICTING HIGHER VARIANT RATING FROM LOWER VARIANT")
    print("Formula: predicted_high_rating = low_rating + β * ln(tok_high / tok_low)")
    print(f"{'='*90}")

    # Build pair records with absolute ratings
    pairs = []  # (low_rating, low_tok, high_rating, high_tok, family)
    for family, variants in sorted(families.items()):
        if len(variants) < 2:
            continue
        variants_sorted = sorted(variants, key=lambda x: x['tokens_per_move'])
        base = variants_sorted[0]
        for v in variants_sorted[1:]:
            if base['tokens_per_move'] > 0 and v['tokens_per_move'] > 0:
                pairs.append((
                    base['rating'], base['tokens_per_move'],
                    v['rating'], v['tokens_per_move'],
                    family, base['player_id'], v['player_id']
                ))

    # Leave-one-out cross-validation
    print(f"\nLeave-one-out cross-validation ({len(pairs)} pairs):")
    print(f"\n{'Family':<25} {'Low variant':<35} {'Low R':>6} {'High variant':<35} {'Actual':>7} {'LOO Pred':>9} {'Error':>7}")
    print("-" * 130)

    loo_errors = []
    for i in range(len(pairs)):
        # Fit beta on all pairs except i
        train_lr = []
        train_rd = []
        for j in range(len(pairs)):
            if j == i:
                continue
            lr_j = np.log(pairs[j][3] / pairs[j][1])
            rd_j = pairs[j][2] - pairs[j][0]
            train_lr.append(lr_j)
            train_rd.append(rd_j)

        train_lr = np.array(train_lr)
        train_rd = np.array(train_rd)
        beta_loo = np.sum(train_lr * train_rd) / np.sum(train_lr ** 2)

        # Predict held-out pair
        low_r, low_t, high_r, high_t, fam, low_name, high_name = pairs[i]
        lr_i = np.log(high_t / low_t)
        pred_high = low_r + beta_loo * lr_i
        error = high_r - pred_high
        loo_errors.append(error)

        print(f"{fam:<25} {low_name:<35} {low_r:>6.0f} {high_name:<35} {high_r:>7.0f} {pred_high:>+9.0f} {error:>+7.0f}")

    loo_errors = np.array(loo_errors)
    rmse = np.sqrt(np.mean(loo_errors ** 2))
    mae = np.mean(np.abs(loo_errors))
    median_ae = np.median(np.abs(loo_errors))

    print(f"\n  RMSE:      {rmse:.0f} rating points")
    print(f"  MAE:       {mae:.0f} rating points")
    print(f"  Median AE: {median_ae:.0f} rating points")
    print(f"  β (full):  {beta:.1f} (Δ rating per unit ln(token_ratio))")
    print(f"\n  Interpretation: predictions are typically off by ~{mae:.0f} rating points")
    print(f"  (for context, 200 rating points ≈ difference between adjacent skill tiers)")


if __name__ == "__main__":
    main()
