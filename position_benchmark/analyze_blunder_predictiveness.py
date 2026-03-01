#!/usr/bin/env python3
"""Analyze how well blunder vs equal positions predict chess game-play ratings."""

import json
import math
import numpy as np
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────────
base = Path(__file__).parent
with open(base / "results.json") as f:
    results = json.load(f)
with open(base / "positions.json") as f:
    positions_data = json.load(f)

# Build position type lookup
pos_type = {}
for i, p in enumerate(positions_data["positions"]):
    pos_type[i] = p["type"]  # "equal" or "blunder"

# Format: {model_name: (rating, games_rd)}
# games_rd tracks RD from game results only (not benchmark seeding)
# Models with games_rd >= 100 excluded (unreliable game-play ratings)
MAX_RD = 100

_ALL_BENCHMARK_RATINGS = {
    "eubos": (2211, 30),
    "gemini-3-pro-preview (high)": (1998, 79),
    "gemini-3.1-pro-preview (medium)": (1954, 153),
    "maia-1900": (1816, 30),
    "survival-bot": (1659, 45),
    "maia-1100": (1628, 30),
    "grok-4.1-fast": (1389, 59),
    "gemini-3.1-pro-preview (high)": (1198, 348),
    "gpt-oss-120b (high)": (735, 97),
    "gpt-5.2 (no thinking)": (595, 53),
    "gemini-3-pro-preview (medium)": (537, 92),
    "gpt-5.1-chat": (526, 45),
    "random-bot": (400, 30),
    "gemini-2.5-flash (no thinking)": (350, 45),
    "gpt-5-chat": (285, 45),
    "gemini-2.0-flash-001": (270, 45),
    "grok-3-mini": (58, 70),
    "kimi-k2": (-149, 45),
    "llama-4-maverick": (-166, 45),
    "deepseek-v3.2 (no thinking)": (-182, 96),
    "deepseek-chat-v3-0324": (-183, 45),
    "kimi-k2-0905": (-216, 45),
    "glm-4.6 (thinking)": (-221, 82),
    "gpt-3.5-turbo-0613": (-240, 45),
    "deepseek-v3.1-terminus (no thinking)": (-248, 45),
    "mistral-medium-3": (-251, 45),
    "deepseek-r1-distill-qwen-32b": (-262, 77),
    "glm-4.6 (no thinking)": (-328, 45),
    "deepseek-chat-v3.1 (no thinking)": (-356, 45),
    "qwen3-235b-a22b-2507": (-403, 90),
    "gpt-3.5-turbo": (-500, 45),
    "llama-3.3-70b-instruct": (-500, 45),
}

BENCHMARK_RATINGS = {
    name: rating for name, (rating, games_rd) in _ALL_BENCHMARK_RATINGS.items()
    if games_rd < MAX_RD
}

# ── Compute per-model features ────────────────────────────────────────────
models = []
y_ratings = []
features = {}  # model_name -> dict of features

for model_name, rating in BENCHMARK_RATINGS.items():
    if model_name not in results:
        print(f"WARNING: {model_name} not found in results.json, skipping")
        continue

    model_results = results[model_name]["results"]

    eq_cpls = []
    bl_cpls = []
    eq_best = []
    bl_best = []

    for r in model_results:
        idx = r["position_idx"]
        ptype = pos_type[idx]
        cpl = r["cpl"]
        is_best = r["is_best"]

        if ptype == "equal":
            eq_cpls.append(cpl)
            eq_best.append(1 if is_best else 0)
        else:
            bl_cpls.append(cpl)
            bl_best.append(1 if is_best else 0)

    mean_eq_cpl = np.mean(eq_cpls) if eq_cpls else 0
    mean_bl_cpl = np.mean(bl_cpls) if bl_cpls else 0
    best_eq_pct = np.mean(eq_best) * 100 if eq_best else 0
    best_bl_pct = np.mean(bl_best) * 100 if bl_best else 0

    models.append(model_name)
    y_ratings.append(rating)
    features[model_name] = {
        "log_eq_cpl": math.log(mean_eq_cpl + 1),
        "log_bl_cpl": math.log(mean_bl_cpl + 1),
        "best_eq_pct": best_eq_pct,
        "best_bl_pct": best_bl_pct,
        "mean_eq_cpl": mean_eq_cpl,
        "mean_bl_cpl": mean_bl_cpl,
    }

n = len(models)
y = np.array(y_ratings, dtype=float)

print(f"Models with benchmark ratings found in results: {n}")
print()

# ── Print per-model features ──────────────────────────────────────────────
print(f"{'Model':<42} {'Rating':>6}  {'EqCPL':>7} {'BlCPL':>7}  {'Eq%':>6} {'Bl%':>6}")
print("-" * 85)
for i, m in enumerate(models):
    f = features[m]
    print(f"{m:<42} {y[i]:>6.0f}  {f['mean_eq_cpl']:>7.0f} {f['mean_bl_cpl']:>7.0f}  {f['best_eq_pct']:>5.1f}% {f['best_bl_pct']:>5.1f}%")
print()


# ── LOO regression helpers ────────────────────────────────────────────────
def loo_single_feature(X_col, y_arr):
    """LOO cross-validation for single feature linear regression."""
    n = len(y_arr)
    errors = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = X_col[mask].reshape(-1, 1)
        y_train = y_arr[mask]

        # Fit OLS: y = a + b*x
        X_aug = np.column_stack([np.ones(n - 1), X_train])
        coeffs = np.linalg.lstsq(X_aug, y_train, rcond=None)[0]

        x_test = np.array([1.0, X_col[i]])
        pred = x_test @ coeffs
        errors[i] = y_arr[i] - pred

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    return r2, rmse, mae, errors


def loo_multi_feature(X_mat, y_arr):
    """LOO cross-validation for multi-feature linear regression."""
    n = len(y_arr)
    errors = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_train = X_mat[mask]
        y_train = y_arr[mask]

        X_aug = np.column_stack([np.ones(n - 1), X_train])
        coeffs = np.linalg.lstsq(X_aug, y_train, rcond=None)[0]

        x_test = np.concatenate([[1.0], X_mat[i]])
        pred = x_test @ coeffs
        errors[i] = y_arr[i] - pred

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    return r2, rmse, mae, errors


def loo_weighted_average(X_eq, X_bl, y_arr, eq_weight):
    """
    LOO: fit separate models on equal and blunder features,
    then combine predictions as weighted average.
    X_eq, X_bl can be 1D (single feature) or 2D (multi-feature).
    """
    n = len(y_arr)
    bl_weight = 1 - eq_weight
    errors = np.zeros(n)

    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        y_train = y_arr[mask]

        # Fit equal model
        if X_eq.ndim == 1:
            Xeq_train = X_eq[mask].reshape(-1, 1)
            Xeq_aug = np.column_stack([np.ones(n - 1), Xeq_train])
            coeffs_eq = np.linalg.lstsq(Xeq_aug, y_train, rcond=None)[0]
            xeq_test = np.array([1.0, X_eq[i]])
        else:
            Xeq_train = X_eq[mask]
            Xeq_aug = np.column_stack([np.ones(n - 1), Xeq_train])
            coeffs_eq = np.linalg.lstsq(Xeq_aug, y_train, rcond=None)[0]
            xeq_test = np.concatenate([[1.0], X_eq[i]])

        pred_eq = xeq_test @ coeffs_eq

        # Fit blunder model
        if X_bl.ndim == 1:
            Xbl_train = X_bl[mask].reshape(-1, 1)
            Xbl_aug = np.column_stack([np.ones(n - 1), Xbl_train])
            coeffs_bl = np.linalg.lstsq(Xbl_aug, y_train, rcond=None)[0]
            xbl_test = np.array([1.0, X_bl[i]])
        else:
            Xbl_train = X_bl[mask]
            Xbl_aug = np.column_stack([np.ones(n - 1), Xbl_train])
            coeffs_bl = np.linalg.lstsq(Xbl_aug, y_train, rcond=None)[0]
            xbl_test = np.concatenate([[1.0], X_bl[i]])

        pred_bl = xbl_test @ coeffs_bl

        pred = eq_weight * pred_eq + bl_weight * pred_bl
        errors[i] = y_arr[i] - pred

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_arr - np.mean(y_arr)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    return r2, rmse, mae


# ── Build feature arrays ──────────────────────────────────────────────────
log_eq_cpl = np.array([features[m]["log_eq_cpl"] for m in models])
log_bl_cpl = np.array([features[m]["log_bl_cpl"] for m in models])
best_eq_pct = np.array([features[m]["best_eq_pct"] for m in models])
best_bl_pct = np.array([features[m]["best_bl_pct"] for m in models])

X_eq_both = np.column_stack([log_eq_cpl, best_eq_pct])
X_bl_both = np.column_stack([log_bl_cpl, best_bl_pct])

# ── Run all analyses ──────────────────────────────────────────────────────
print("=" * 80)
print("SINGLE-FEATURE REGRESSIONS (LOO Cross-Validation)")
print("=" * 80)
print(f"{'Approach':<45} {'R²':>7} {'RMSE':>7} {'MAE':>7}")
print("-" * 70)

r2_a, rmse_a, mae_a, err_a = loo_single_feature(log_eq_cpl, y)
print(f"{'(a) Equal-only: log(CPL+1)':<45} {r2_a:>7.3f} {rmse_a:>7.0f} {mae_a:>7.0f}")

r2_b, rmse_b, mae_b, err_b = loo_single_feature(log_bl_cpl, y)
print(f"{'(b) Blunder-only: log(CPL+1)':<45} {r2_b:>7.3f} {rmse_b:>7.0f} {mae_b:>7.0f}")

r2_c, rmse_c, mae_c, err_c = loo_single_feature(best_eq_pct, y)
print(f"{'(c) Equal-only: best_move_pct':<45} {r2_c:>7.3f} {rmse_c:>7.0f} {mae_c:>7.0f}")

r2_d, rmse_d, mae_d, err_d = loo_single_feature(best_bl_pct, y)
print(f"{'(d) Blunder-only: best_move_pct':<45} {r2_d:>7.3f} {rmse_d:>7.0f} {mae_d:>7.0f}")

print()
print("=" * 80)
print("MULTI-FEATURE REGRESSIONS (LOO Cross-Validation)")
print("=" * 80)
print(f"{'Approach':<45} {'R²':>7} {'RMSE':>7} {'MAE':>7}")
print("-" * 70)

r2_e, rmse_e, mae_e, err_e = loo_multi_feature(X_eq_both, y)
print(f"{'(e) Equal-only: log(CPL) + best_pct':<45} {r2_e:>7.3f} {rmse_e:>7.0f} {mae_e:>7.0f}")

r2_f, rmse_f, mae_f, err_f = loo_multi_feature(X_bl_both, y)
print(f"{'(f) Blunder-only: log(CPL) + best_pct':<45} {r2_f:>7.3f} {rmse_f:>7.0f} {mae_f:>7.0f}")

# Combined: all 4 features in one model
X_all4 = np.column_stack([log_eq_cpl, log_bl_cpl, best_eq_pct, best_bl_pct])
r2_all4, rmse_all4, mae_all4, err_all4 = loo_multi_feature(X_all4, y)
print(f"{'(+) All 4 features combined':<45} {r2_all4:>7.3f} {rmse_all4:>7.0f} {mae_all4:>7.0f}")

# Combined: both log_cpl features
X_both_cpl = np.column_stack([log_eq_cpl, log_bl_cpl])
r2_bc, rmse_bc, mae_bc, err_bc = loo_multi_feature(X_both_cpl, y)
print(f"{'(+) Both log(CPL) features':<45} {r2_bc:>7.3f} {rmse_bc:>7.0f} {mae_bc:>7.0f}")

# Combined: both best_pct features
X_both_bp = np.column_stack([best_eq_pct, best_bl_pct])
r2_bbp, rmse_bbp, mae_bbp, err_bbp = loo_multi_feature(X_both_bp, y)
print(f"{'(+) Both best_move_pct features':<45} {r2_bbp:>7.3f} {rmse_bbp:>7.0f} {mae_bbp:>7.0f}")

print()
print("=" * 80)
print("WEIGHTED AVERAGING: Equal(log_cpl) + Blunder(log_cpl)")
print("=" * 80)
print(f"{'Equal Weight':<15} {'Blunder Weight':<15} {'R²':>7} {'RMSE':>7} {'MAE':>7}")
print("-" * 55)

weights = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
best_g = (None, -999, 999)
for w in weights:
    r2_g, rmse_g, mae_g = loo_weighted_average(log_eq_cpl, log_bl_cpl, y, w)
    print(f"{w:<15.0%} {1 - w:<15.0%} {r2_g:>7.3f} {rmse_g:>7.0f} {mae_g:>7.0f}")
    if r2_g > best_g[1]:
        best_g = (w, r2_g, rmse_g)

print(f"\n  Best: {best_g[0]:.0%} equal / {1 - best_g[0]:.0%} blunder  (R²={best_g[1]:.3f}, RMSE={best_g[2]:.0f})")

print()
print("=" * 80)
print("WEIGHTED AVERAGING: Equal(log_cpl+best_pct) + Blunder(log_cpl+best_pct)")
print("=" * 80)
print(f"{'Equal Weight':<15} {'Blunder Weight':<15} {'R²':>7} {'RMSE':>7} {'MAE':>7}")
print("-" * 55)

best_h = (None, -999, 999)
for w in weights:
    r2_h, rmse_h, mae_h = loo_weighted_average(X_eq_both, X_bl_both, y, w)
    print(f"{w:<15.0%} {1 - w:<15.0%} {r2_h:>7.3f} {rmse_h:>7.0f} {mae_h:>7.0f}")
    if r2_h > best_h[1]:
        best_h = (w, r2_h, rmse_h)

print(f"\n  Best: {best_h[0]:.0%} equal / {1 - best_h[0]:.0%} blunder  (R²={best_h[1]:.3f}, RMSE={best_h[2]:.0f})")

# ── Per-model error analysis for key approaches ──────────────────────────
print()
print("=" * 80)
print("PER-MODEL ERRORS: Equal log(CPL) vs Blunder log(CPL) vs Equal(both) vs Blunder(both)")
print("=" * 80)
print(f"{'Model':<42} {'Rating':>6} {'EqCPL':>7} {'BlCPL':>7} {'Eq2F':>7} {'Bl2F':>7}")
print("-" * 80)

for i, m in enumerate(models):
    print(f"{m:<42} {y[i]:>6.0f} {err_a[i]:>+7.0f} {err_b[i]:>+7.0f} {err_e[i]:>+7.0f} {err_f[i]:>+7.0f}")

print()
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("Single-feature comparison:")
print(f"  Equal log(CPL):       R²={r2_a:.3f}  RMSE={rmse_a:.0f}")
print(f"  Blunder log(CPL):     R²={r2_b:.3f}  RMSE={rmse_b:.0f}")
print(f"  Equal best_pct:       R²={r2_c:.3f}  RMSE={rmse_c:.0f}")
print(f"  Blunder best_pct:     R²={r2_d:.3f}  RMSE={rmse_d:.0f}")
print()
print("Two-feature comparison:")
print(f"  Equal (CPL+best):     R²={r2_e:.3f}  RMSE={rmse_e:.0f}")
print(f"  Blunder (CPL+best):   R²={r2_f:.3f}  RMSE={rmse_f:.0f}")
print()
print("Best combined approaches:")
print(f"  All 4 features:       R²={r2_all4:.3f}  RMSE={rmse_all4:.0f}")
print(f"  Both log(CPL):        R²={r2_bc:.3f}  RMSE={rmse_bc:.0f}")
print(f"  Both best_pct:        R²={r2_bbp:.3f}  RMSE={rmse_bbp:.0f}")
print()
print(f"  Weighted avg (CPL):   Best at {best_g[0]:.0%} eq / {1 - best_g[0]:.0%} bl  R²={best_g[1]:.3f}  RMSE={best_g[2]:.0f}")
print(f"  Weighted avg (both):  Best at {best_h[0]:.0%} eq / {1 - best_h[0]:.0%} bl  R²={best_h[1]:.3f}  RMSE={best_h[2]:.0f}")
