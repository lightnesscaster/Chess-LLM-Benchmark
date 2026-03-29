"""
Comprehensive regression search for best position benchmark -> rating model.

Tries many model forms with Leave-One-Out cross-validation.
Uses equal position features only (allows more models).
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import binom
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


# Current ratings from leaderboard (Mar 2026)
# Format: {model_name: (rating, rd)}
MAX_RD = 100

_ALL_BENCHMARK_RATINGS = {
    # Engines (fixed ratings)
    "eubos": (2211, 30),
    "maia-1900": (1816, 30),
    "survival-bot": (1664, 45),
    "maia-1100": (1628, 30),
    "random-bot": (400, 30),
    # Gemini models
    "gemini-3.1-pro-preview (high)": (1986, 97),
    "gemini-3.1-pro-preview (medium)": (1918, 70),
    "gemini-3-pro-preview (high)": (1871, 65),
    "gemini-3-flash-preview (medium)": (1828, 85),
    "gemini-3-flash-preview (high)": (1792, 130),
    "gemini-3-pro-preview (medium)": (658, 70),
    "gemini-2.5-pro (medium)": (382, 59),
    "gemini-2.5-flash (no thinking)": (338, 45),
    "gemini-3.1-flash-lite-preview": (301, 77),
    "gemini-2.0-flash-001": (223, 45),
    # GPT models
    "gpt-5.1 (high)": (1245, 86),
    "gpt-5 (medium)": (1148, 137),
    "gpt-5.2 (high)": (1037, 89),
    "gpt-5.2-chat": (725, 61),
    "gpt-oss-120b (high)": (709, 69),
    "gpt-5.2 (no thinking)": (615, 53),
    "gpt-5.1-chat": (537, 45),
    "gpt-5-chat": (246, 45),
    "gpt-3.5-turbo-0613": (-282, 45),
    "gpt-3.5-turbo": (-500, 45),
    # Grok models
    "grok-4.1-fast": (1361, 56),
    "grok-4-fast": (1178, 64),
    "grok-3-mini": (5, 63),
    # Claude models
    "claude-opus-4.5 (high)": (653, 113),
    "claude-opus-4.6 (medium)": (903, 144),
    "claude-opus-4.6 (no thinking)": (180, 100),
    # Other models
    "kimi-k2": (-141, 45),
    "llama-4-maverick": (-202, 45),
    "glm-4.6 (thinking)": (-232, 77),
    "deepseek-chat-v3-0324": (-222, 45),
    "kimi-k2-0905": (-232, 45),
    "deepseek-v3.2 (no thinking)": (-259, 91),
    "mistral-medium-3": (-270, 45),
    "deepseek-v3.1-terminus (no thinking)": (-269, 45),
    "deepseek-r1-distill-qwen-32b": (-316, 71),
    "glm-4.6 (no thinking)": (-357, 45),
    "deepseek-chat-v3.1 (no thinking)": (-369, 45),
    "qwen3-235b-a22b-2507": (-451, 81),
    "llama-3.3-70b-instruct": (-500, 45),
}

BENCHMARK_RATINGS = {
    name: rating for name, (rating, rd) in _ALL_BENCHMARK_RATINGS.items()
    if rd < MAX_RD
}


def survival_probability(illegal_rate, game_length=40):
    if illegal_rate <= 0:
        return 100.0
    if illegal_rate >= 1:
        return 0.0
    return 100.0 * (binom.pmf(0, game_length, illegal_rate) +
                     binom.pmf(1, game_length, illegal_rate))


def load_data():
    """Load position benchmark results and extract equal-position features."""
    base_path = Path(__file__).parent
    with open(base_path / "results.json") as f:
        data = json.load(f)

    models = []
    for model_name, model_data in data.items():
        if model_name not in BENCHMARK_RATINGS:
            continue

        summary = model_data.get("summary", {})
        equal_stats = summary.get("equal")
        if not equal_stats:
            continue

        # Extract per-position data for richer features
        results = model_data.get("results", [])
        equal_results = [r for r in results if r.get("position_type") == "equal"
                         or (not r.get("position_type") and r.get("position_idx", 0) < 50)]

        # Compute CPL distribution features from individual results
        cpls = []
        legal_count = 0
        best_count = 0
        for r in equal_results:
            if r.get("is_legal", False):
                cpls.append(r.get("cpl", 0))
                legal_count += 1
                if r.get("is_best", False):
                    best_count += 1

        total = len(equal_results) if equal_results else 50

        # CPL distribution features (from legal moves only)
        if cpls:
            cpls_arr = np.array(cpls)
            median_cpl = np.median(cpls_arr)
            p75_cpl = np.percentile(cpls_arr, 75)
            p90_cpl = np.percentile(cpls_arr, 90)
            std_cpl = np.std(cpls_arr)
            pct_lt10 = 100.0 * np.sum(cpls_arr < 10) / total
            pct_lt50 = 100.0 * np.sum(cpls_arr < 50) / total
            pct_lt100 = 100.0 * np.sum(cpls_arr < 100) / total
            capped_cpl = np.mean(np.minimum(cpls_arr, 2000))
            # Geometric mean of CPL+1 (robust to outliers)
            geo_mean_cpl = np.exp(np.mean(np.log(cpls_arr + 1))) - 1
        else:
            median_cpl = equal_stats.get("median_cpl", equal_stats["avg_cpl"])
            p75_cpl = p90_cpl = std_cpl = 0
            pct_lt10 = pct_lt50 = pct_lt100 = 0
            capped_cpl = min(equal_stats["avg_cpl"], 2000)
            geo_mean_cpl = equal_stats["avg_cpl"]

        illegal_rate = 1.0 - equal_stats["legal_pct"] / 100.0
        surv_40 = survival_probability(illegal_rate, 40)
        surv_50 = survival_probability(illegal_rate, 50)

        models.append({
            "name": model_name,
            "rating": BENCHMARK_RATINGS[model_name],
            "rd": _ALL_BENCHMARK_RATINGS[model_name][1],
            # Basic stats
            "eq_avg_cpl": equal_stats["avg_cpl"],
            "eq_legal_pct": equal_stats["legal_pct"],
            "eq_best_pct": equal_stats["best_pct"],
            # Log transforms
            "log_eq_cpl": np.log(equal_stats["avg_cpl"] + 1),
            "log_capped_cpl": np.log(capped_cpl + 1),
            "log_geo_cpl": np.log(geo_mean_cpl + 1),
            # Distribution features
            "eq_median_cpl": median_cpl,
            "log_median_cpl": np.log(median_cpl + 1) if median_cpl > 0 else 0,
            "eq_p75_cpl": p75_cpl,
            "eq_p90_cpl": p90_cpl,
            "eq_std_cpl": std_cpl,
            "pct_lt10": pct_lt10,
            "pct_lt50": pct_lt50,
            "pct_lt100": pct_lt100,
            # Survival
            "surv_40": surv_40,
            "surv_50": surv_50,
            "illegal_rate": illegal_rate,
            # Capped
            "capped_cpl": capped_cpl,
            "geo_mean_cpl": geo_mean_cpl,
            # Sqrt transform
            "sqrt_eq_cpl": np.sqrt(equal_stats["avg_cpl"]),
            # Inverse
            "inv_cpl": 1.0 / (equal_stats["avg_cpl"] + 1),
        })

    return models


def loo_cv(X, y, model_class=LinearRegression, **model_kwargs):
    """Leave-one-out cross-validation. Returns predictions array."""
    n = len(y)
    predictions = np.zeros(n)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        predictions[i] = model.predict(X[i:i+1])[0]
    return predictions


def loo_cv_weighted(X, y, weights, model_class=LinearRegression, **model_kwargs):
    """Leave-one-out cross-validation with sample weights."""
    n = len(y)
    predictions = np.zeros(n)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        w_train = np.delete(weights, i)
        model = model_class(**model_kwargs)
        model.fit(X_train, y_train, sample_weight=w_train)
        predictions[i] = model.predict(X[i:i+1])[0]
    return predictions


def loo_cv_poly(X, y, degree=2):
    """LOO CV with polynomial features."""
    n = len(y)
    predictions = np.zeros(n)
    for i in range(n):
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y, i)
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X[i:i+1])
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        predictions[i] = model.predict(X_test_poly)[0]
    return predictions


def evaluate_model(y_true, y_pred, label=""):
    """Return dict of metrics."""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    max_err = np.max(np.abs(y_true - y_pred))
    return {"label": label, "r2": r2, "rmse": rmse, "mae": mae, "max_err": max_err}


def main():
    models = load_data()
    n = len(models)

    print("=" * 80)
    print(f"COMPREHENSIVE REGRESSION SEARCH ({n} models, RD < {MAX_RD})")
    print("=" * 80)

    names = [m["name"] for m in models]
    y = np.array([m["rating"] for m in models])

    # Build feature matrix
    feature_names = [
        "log_eq_cpl", "eq_best_pct", "eq_legal_pct", "pct_lt10", "pct_lt50",
        "pct_lt100", "surv_40", "surv_50", "log_capped_cpl", "log_geo_cpl",
        "log_median_cpl", "sqrt_eq_cpl", "inv_cpl", "eq_median_cpl", "capped_cpl",
        "illegal_rate",
    ]
    features = {}
    for fname in feature_names:
        features[fname] = np.array([m[fname] for m in models])

    # RD-based weights (higher weight for more certain ratings)
    rd_weights = np.array([1.0 / m["rd"] for m in models])
    rd_weights = rd_weights / rd_weights.mean()

    results = []

    # =========================================================================
    # PHASE 1: Single-feature models
    # =========================================================================
    print("\n" + "-" * 80)
    print("PHASE 1: SINGLE-FEATURE MODELS (LOO CV)")
    print("-" * 80)

    single_features = [
        "log_eq_cpl", "eq_best_pct", "eq_legal_pct", "pct_lt10", "pct_lt50",
        "pct_lt100", "surv_40", "surv_50", "log_capped_cpl", "log_geo_cpl",
        "log_median_cpl", "sqrt_eq_cpl", "inv_cpl",
    ]

    for fname in single_features:
        X = features[fname].reshape(-1, 1)
        y_pred = loo_cv(X, y)
        res = evaluate_model(y, y_pred, f"1F: {fname}")
        results.append(res)

    results_sorted = sorted(results, key=lambda r: -r["r2"])
    print(f"\n  {'Model':<45} {'LOO R²':>8} {'RMSE':>8} {'MAE':>8} {'MaxErr':>8}")
    print("  " + "-" * 79)
    for r in results_sorted:
        print(f"  {r['label']:<45} {r['r2']:>8.4f} {r['rmse']:>8.0f} {r['mae']:>8.0f} {r['max_err']:>8.0f}")

    # =========================================================================
    # PHASE 2: Two-feature models (all pairs)
    # =========================================================================
    print("\n" + "-" * 80)
    print("PHASE 2: TWO-FEATURE MODELS (LOO CV)")
    print("-" * 80)

    two_feat_candidates = [
        "log_eq_cpl", "eq_best_pct", "eq_legal_pct", "pct_lt10", "pct_lt50",
        "pct_lt100", "surv_40", "surv_50", "log_capped_cpl", "log_geo_cpl",
        "log_median_cpl", "sqrt_eq_cpl",
    ]

    results_2f = []
    for f1, f2 in combinations(two_feat_candidates, 2):
        X = np.column_stack([features[f1], features[f2]])
        y_pred = loo_cv(X, y)
        res = evaluate_model(y, y_pred, f"2F: {f1} + {f2}")
        results_2f.append(res)

    results_2f.sort(key=lambda r: -r["r2"])
    print(f"\n  {'Model':<65} {'LOO R²':>8} {'RMSE':>8} {'MAE':>8}")
    print("  " + "-" * 90)
    for r in results_2f[:20]:
        print(f"  {r['label']:<65} {r['r2']:>8.4f} {r['rmse']:>8.0f} {r['mae']:>8.0f}")

    # =========================================================================
    # PHASE 3: Three-feature models (top combinations)
    # =========================================================================
    print("\n" + "-" * 80)
    print("PHASE 3: THREE-FEATURE MODELS (LOO CV)")
    print("-" * 80)

    three_feat_candidates = [
        "log_eq_cpl", "eq_best_pct", "pct_lt10", "pct_lt50",
        "pct_lt100", "surv_40", "surv_50", "log_capped_cpl", "log_geo_cpl",
        "log_median_cpl", "eq_legal_pct", "sqrt_eq_cpl",
    ]

    results_3f = []
    for f1, f2, f3 in combinations(three_feat_candidates, 3):
        X = np.column_stack([features[f1], features[f2], features[f3]])
        y_pred = loo_cv(X, y)
        res = evaluate_model(y, y_pred, f"3F: {f1} + {f2} + {f3}")
        results_3f.append(res)

    results_3f.sort(key=lambda r: -r["r2"])
    print(f"\n  {'Model':<75} {'LOO R²':>8} {'RMSE':>8}")
    print("  " + "-" * 93)
    for r in results_3f[:20]:
        print(f"  {r['label']:<75} {r['r2']:>8.4f} {r['rmse']:>8.0f}")

    # =========================================================================
    # PHASE 4: Polynomial models on best features
    # =========================================================================
    print("\n" + "-" * 80)
    print("PHASE 4: POLYNOMIAL MODELS (LOO CV)")
    print("-" * 80)

    results_poly = []

    # Degree 2 on single features
    for fname in ["log_eq_cpl", "eq_best_pct", "pct_lt10", "surv_40", "log_capped_cpl"]:
        X = features[fname].reshape(-1, 1)
        y_pred = loo_cv_poly(X, y, degree=2)
        res = evaluate_model(y, y_pred, f"Poly2(1F): {fname}")
        results_poly.append(res)

    # Degree 2 on best two-feature combos
    best_2f_combos = [
        ("log_eq_cpl", "eq_best_pct"),
        ("log_eq_cpl", "pct_lt10"),
        ("log_capped_cpl", "pct_lt10"),
        ("log_geo_cpl", "pct_lt10"),
        ("log_eq_cpl", "pct_lt50"),
        ("log_eq_cpl", "surv_40"),
        ("eq_best_pct", "surv_40"),
        ("pct_lt10", "surv_40"),
    ]
    for f1, f2 in best_2f_combos:
        X = np.column_stack([features[f1], features[f2]])
        y_pred = loo_cv_poly(X, y, degree=2)
        res = evaluate_model(y, y_pred, f"Poly2(2F): {f1} + {f2}")
        results_poly.append(res)

    # Degree 3 on single features
    for fname in ["log_eq_cpl", "eq_best_pct", "pct_lt10"]:
        X = features[fname].reshape(-1, 1)
        y_pred = loo_cv_poly(X, y, degree=3)
        res = evaluate_model(y, y_pred, f"Poly3(1F): {fname}")
        results_poly.append(res)

    results_poly.sort(key=lambda r: -r["r2"])
    print(f"\n  {'Model':<65} {'LOO R²':>8} {'RMSE':>8} {'MAE':>8}")
    print("  " + "-" * 90)
    for r in results_poly:
        print(f"  {r['label']:<65} {r['r2']:>8.4f} {r['rmse']:>8.0f} {r['mae']:>8.0f}")

    # =========================================================================
    # PHASE 5: Robust regression (Huber) on best combos
    # =========================================================================
    print("\n" + "-" * 80)
    print("PHASE 5: ROBUST REGRESSION - HUBER (LOO CV)")
    print("-" * 80)

    results_robust = []
    robust_combos = [
        ("log_eq_cpl",),
        ("log_eq_cpl", "eq_best_pct"),
        ("log_eq_cpl", "pct_lt10"),
        ("log_eq_cpl", "pct_lt10", "surv_40"),
        ("log_eq_cpl", "eq_best_pct", "surv_40"),
        ("log_eq_cpl", "eq_best_pct", "surv_50"),
        ("log_capped_cpl", "pct_lt10"),
        ("log_capped_cpl", "pct_lt10", "surv_40"),
    ]
    for combo in robust_combos:
        X = np.column_stack([features[f] for f in combo])
        y_pred = loo_cv(X, y, model_class=HuberRegressor, epsilon=1.35)
        label = "Huber: " + " + ".join(combo)
        res = evaluate_model(y, y_pred, label)
        results_robust.append(res)

    results_robust.sort(key=lambda r: -r["r2"])
    print(f"\n  {'Model':<65} {'LOO R²':>8} {'RMSE':>8} {'MAE':>8}")
    print("  " + "-" * 90)
    for r in results_robust:
        print(f"  {r['label']:<65} {r['r2']:>8.4f} {r['rmse']:>8.0f} {r['mae']:>8.0f}")

    # =========================================================================
    # PHASE 6: Ridge regression on various combos
    # =========================================================================
    print("\n" + "-" * 80)
    print("PHASE 6: RIDGE REGRESSION (LOO CV)")
    print("-" * 80)

    results_ridge = []
    ridge_combos = [
        ("log_eq_cpl", "eq_best_pct", "eq_legal_pct"),
        ("log_eq_cpl", "eq_best_pct", "surv_40"),
        ("log_eq_cpl", "eq_best_pct", "surv_50"),
        ("log_eq_cpl", "pct_lt10", "surv_40"),
        ("log_eq_cpl", "pct_lt10", "surv_50"),
        ("log_eq_cpl", "pct_lt10", "surv_40", "eq_best_pct"),
        ("log_eq_cpl", "pct_lt10", "surv_50", "eq_best_pct"),
        ("log_eq_cpl", "eq_best_pct", "pct_lt10", "surv_40", "eq_legal_pct"),
        ("log_capped_cpl", "pct_lt10", "surv_40", "eq_best_pct"),
    ]
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        for combo in ridge_combos:
            X = np.column_stack([features[f] for f in combo])
            y_pred = loo_cv(X, y, model_class=Ridge, alpha=alpha)
            label = f"Ridge(a={alpha}): " + " + ".join(combo)
            res = evaluate_model(y, y_pred, label)
            results_ridge.append(res)

    results_ridge.sort(key=lambda r: -r["r2"])
    print(f"\n  {'Model':<75} {'LOO R²':>8} {'RMSE':>8}")
    print("  " + "-" * 93)
    for r in results_ridge[:15]:
        print(f"  {r['label']:<75} {r['r2']:>8.4f} {r['rmse']:>8.0f}")

    # =========================================================================
    # PHASE 7: RD-weighted regression (weight by 1/RD for more certain ratings)
    # =========================================================================
    print("\n" + "-" * 80)
    print("PHASE 7: RD-WEIGHTED REGRESSION (LOO CV)")
    print("-" * 80)

    results_weighted = []
    weighted_combos = [
        ("log_eq_cpl", "eq_best_pct", "surv_40"),
        ("log_eq_cpl", "eq_best_pct", "surv_50"),
        ("eq_best_pct", "surv_40"),
        ("eq_best_pct", "surv_50"),
        ("log_eq_cpl", "eq_best_pct"),
        ("log_eq_cpl", "eq_best_pct", "surv_40", "pct_lt10"),
        ("log_eq_cpl", "eq_best_pct", "surv_50", "pct_lt10"),
    ]
    for combo in weighted_combos:
        X = np.column_stack([features[f] for f in combo])
        y_pred = loo_cv_weighted(X, y, rd_weights)
        label = "WLS: " + " + ".join(combo)
        res = evaluate_model(y, y_pred, label)
        results_weighted.append(res)

    # Also try weighted Ridge
    for alpha in [1.0, 10.0]:
        for combo in [
            ("log_eq_cpl", "eq_best_pct", "surv_40"),
            ("log_eq_cpl", "eq_best_pct", "surv_50"),
        ]:
            X = np.column_stack([features[f] for f in combo])
            y_pred = loo_cv_weighted(X, y, rd_weights, model_class=Ridge, alpha=alpha)
            label = f"WLS+Ridge(a={alpha}): " + " + ".join(combo)
            res = evaluate_model(y, y_pred, label)
            results_weighted.append(res)

    results_weighted.sort(key=lambda r: -r["r2"])
    print(f"\n  {'Model':<75} {'LOO R²':>8} {'RMSE':>8} {'MAE':>8}")
    print("  " + "-" * 100)
    for r in results_weighted[:15]:
        print(f"  {r['label']:<75} {r['r2']:>8.4f} {r['rmse']:>8.0f} {r['mae']:>8.0f}")

    # =========================================================================
    # OVERALL BEST MODELS
    # =========================================================================
    all_results = results + results_2f + results_3f + results_poly + results_robust + results_ridge + results_weighted
    all_results.sort(key=lambda r: -r["r2"])

    print("\n" + "=" * 80)
    print("TOP 25 MODELS BY LOO CV R²")
    print("=" * 80)
    print(f"\n  {'Rank':<6} {'Model':<70} {'LOO R²':>8} {'RMSE':>8} {'MAE':>8} {'MaxErr':>8}")
    print("  " + "-" * 100)
    for i, r in enumerate(all_results[:25]):
        print(f"  {i+1:<6} {r['label']:<70} {r['r2']:>8.4f} {r['rmse']:>8.0f} {r['mae']:>8.0f} {r['max_err']:>8.0f}")

    # =========================================================================
    # FIT AND SHOW BEST MODEL DETAILS
    # =========================================================================
    best = all_results[0]
    print("\n" + "=" * 80)
    print(f"BEST MODEL: {best['label']}")
    print("=" * 80)
    print(f"  LOO CV R²:  {best['r2']:.4f}")
    print(f"  LOO RMSE:   {best['rmse']:.0f}")
    print(f"  LOO MAE:    {best['mae']:.0f}")
    print(f"  Max Error:  {best['max_err']:.0f}")

    # Parse the best model's features and fit on full data
    label = best["label"]
    # Extract feature names from label
    if ": " in label:
        feat_str = label.split(": ", 1)[1]
        feat_list = [f.strip() for f in feat_str.split(" + ")]
    else:
        feat_list = []

    if feat_list and feat_list[0] in features:
        X_best = np.column_stack([features[f] for f in feat_list])

        if "Poly" in label:
            degree = 3 if "Poly3" in label else 2
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            X_best_poly = poly.fit_transform(X_best)
            model = LinearRegression()
            model.fit(X_best_poly, y)
            y_pred_full = model.predict(X_best_poly)
            print(f"\n  Training R²: {r2_score(y, y_pred_full):.4f}")
            print(f"  Training RMSE: {np.sqrt(mean_squared_error(y, y_pred_full)):.0f}")
            fnames = poly.get_feature_names_out(feat_list)
            print("\n  Coefficients:")
            for fn, c in zip(fnames, model.coef_):
                print(f"    {fn}: {c:.4f}")
            print(f"    intercept: {model.intercept_:.4f}")
        elif "Huber" in label:
            model = HuberRegressor(epsilon=1.35)
            model.fit(X_best, y)
            y_pred_full = model.predict(X_best)
            print(f"\n  Training R²: {r2_score(y, y_pred_full):.4f}")
            print(f"  Training RMSE: {np.sqrt(mean_squared_error(y, y_pred_full)):.0f}")
            print("\n  Formula:")
            terms = []
            for fn, c in zip(feat_list, model.coef_):
                terms.append(f"{c:.4f} * {fn}")
            print(f"    rating = {' + '.join(terms)} + {model.intercept_:.4f}")
        elif "Ridge" in label:
            import re
            alpha_match = re.search(r'a=([\d.]+)', label)
            alpha = float(alpha_match.group(1)) if alpha_match else 1.0
            model = Ridge(alpha=alpha)
            sw = rd_weights if "WLS" in label else None
            model.fit(X_best, y, sample_weight=sw)
            y_pred_full = model.predict(X_best)
            print(f"\n  Training R²: {r2_score(y, y_pred_full):.4f}")
            print(f"  Training RMSE: {np.sqrt(mean_squared_error(y, y_pred_full)):.0f}")
            print("\n  Formula:")
            terms = []
            for fn, c in zip(feat_list, model.coef_):
                terms.append(f"{c:.4f} * {fn}")
            print(f"    rating = {' + '.join(terms)} + {model.intercept_:.4f}")
        else:
            model = LinearRegression()
            sw = rd_weights if "WLS" in label else None
            model.fit(X_best, y, sample_weight=sw)
            y_pred_full = model.predict(X_best)
            print(f"\n  Training R²: {r2_score(y, y_pred_full):.4f}")
            print(f"  Training RMSE: {np.sqrt(mean_squared_error(y, y_pred_full)):.0f}")
            print("\n  Formula:")
            terms = []
            for fn, c in zip(feat_list, model.coef_):
                terms.append(f"{c:.4f} * {fn}")
            print(f"    rating = {' + '.join(terms)} + {model.intercept_:.4f}")

        # Show LOO predictions for the best model
        is_weighted = "WLS" in label
        if "Poly" not in label:
            if "Huber" in label:
                y_pred_loo = loo_cv(X_best, y, model_class=HuberRegressor, epsilon=1.35)
            elif "Ridge" in label:
                import re
                alpha_match = re.search(r'a=([\d.]+)', label)
                alpha = float(alpha_match.group(1)) if alpha_match else 1.0
                if is_weighted:
                    y_pred_loo = loo_cv_weighted(X_best, y, rd_weights, model_class=Ridge, alpha=alpha)
                else:
                    y_pred_loo = loo_cv(X_best, y, model_class=Ridge, alpha=alpha)
            elif is_weighted:
                y_pred_loo = loo_cv_weighted(X_best, y, rd_weights)
            else:
                y_pred_loo = loo_cv(X_best, y)
        else:
            degree = 3 if "Poly3" in label else 2
            y_pred_loo = loo_cv_poly(X_best, y, degree=degree)

        print(f"\n  {'Model':<45} {'Actual':>8} {'LOO Pred':>8} {'Error':>8}")
        print("  " + "-" * 71)
        idx_sorted = np.argsort(-y)
        for i in idx_sorted:
            err = y[i] - y_pred_loo[i]
            marker = " ***" if abs(err) > 300 else ""
            print(f"  {names[i]:<45} {y[i]:>8.0f} {y_pred_loo[i]:>8.0f} {err:>+8.0f}{marker}")

    # =========================================================================
    # Also show top 3 from each category
    # =========================================================================
    print("\n" + "=" * 80)
    print("BEST BY CATEGORY")
    print("=" * 80)

    for cat_name, cat_results in [
        ("Single-feature", results),
        ("Two-feature", results_2f),
        ("Three-feature", results_3f),
        ("Polynomial", results_poly),
        ("Robust (Huber)", results_robust),
        ("Ridge", results_ridge),
        ("RD-Weighted", results_weighted),
    ]:
        cat_sorted = sorted(cat_results, key=lambda r: -r["r2"])
        print(f"\n  {cat_name}:")
        for r in cat_sorted[:3]:
            print(f"    {r['label']:<70} R²={r['r2']:.4f}  RMSE={r['rmse']:.0f}")


if __name__ == "__main__":
    main()
