"""
Regression analysis of position benchmark results.

Fits various regression models to understand relationships between:
- Position benchmark metrics (CPL, legal%, best%) and game-play ratings
- Legal move percentage and CPL
- Performance on blunder vs equal positions

Reads from unified results.json with per-type summary breakdowns.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')


# Ratings from the main benchmark (Glicko-2 ratings from actual games)
# Format: {model_name: (rating, games_rd)}
# games_rd tracks RD from game results only (not benchmark seeding)
# Updated Mar 2026 — models with games_rd >= 100 excluded from regression
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

# Filter to models with games_rd < MAX_RD
BENCHMARK_RATINGS = {
    name: rating for name, (rating, games_rd) in _ALL_BENCHMARK_RATINGS.items()
    if games_rd < MAX_RD
}


def load_results():
    """Load unified benchmark results file."""
    base_path = Path(__file__).parent

    with open(base_path / "results.json") as f:
        data = json.load(f)

    return data


def extract_model_features(data):
    """Extract features for each model from unified results with per-type breakdowns."""
    models = []

    for model_name, model_data in data.items():
        if model_name not in BENCHMARK_RATINGS:
            continue

        summary = model_data.get("summary", {})

        # Need both blunder and equal breakdowns
        blunder_stats = summary.get("blunder")
        equal_stats = summary.get("equal")
        if not blunder_stats or not equal_stats:
            continue

        # Detect engine vs LLM from results (engines always have 100% legal)
        is_llm = 0 if blunder_stats["legal_pct"] == 100.0 and equal_stats["legal_pct"] == 100.0 else 1

        models.append({
            "name": model_name,
            "is_llm": is_llm,
            "rating": BENCHMARK_RATINGS[model_name],
            # Blunder benchmark metrics
            "blunder_legal_pct": blunder_stats["legal_pct"],
            "blunder_best_pct": blunder_stats["best_pct"],
            "blunder_avg_cpl": blunder_stats["avg_cpl"],
            "blunder_avoided_pct": blunder_stats.get("avoided_pct", 100.0),
            "blunder_median_cpl": blunder_stats.get("median_cpl", blunder_stats["avg_cpl"]),
            # Equal position benchmark metrics
            "equal_legal_pct": equal_stats["legal_pct"],
            "equal_best_pct": equal_stats["best_pct"],
            "equal_avg_cpl": equal_stats["avg_cpl"],
            "equal_avoided_pct": equal_stats.get("avoided_pct", 100.0),
            "equal_median_cpl": equal_stats.get("median_cpl", equal_stats["avg_cpl"]),
        })

    return models


def fit_regression_models(models):
    """Fit regression models to predict rating from position benchmark metrics."""
    print("=" * 70)
    print("REGRESSION: POSITION BENCHMARK -> GAME-PLAY RATING")
    print("=" * 70)
    print(f"\nDataset: {len(models)} models with position metrics and Glicko-2 ratings\n")

    # Convert to numpy arrays
    names = [m["name"] for m in models]
    is_llm = np.array([m["is_llm"] for m in models])
    ratings = np.array([m["rating"] for m in models])

    blunder_legal = np.array([m["blunder_legal_pct"] for m in models])
    blunder_best = np.array([m["blunder_best_pct"] for m in models])
    blunder_cpl = np.array([m["blunder_avg_cpl"] for m in models])

    equal_legal = np.array([m["equal_legal_pct"] for m in models])
    equal_best = np.array([m["equal_best_pct"] for m in models])
    equal_cpl = np.array([m["equal_avg_cpl"] for m in models])

    # Log-transformed CPL (essential - linear CPL gives much lower R²)
    log_blunder_cpl = np.log(blunder_cpl + 1)
    log_equal_cpl = np.log(equal_cpl + 1)

    y = ratings

    # =========================================================================
    # Model 1: Simple log-linear (single feature)
    # =========================================================================
    print("-" * 70)
    print("MODEL 1: Rating from log(equal_cpl) (single feature)")
    print("-" * 70)

    X1 = log_equal_cpl.reshape(-1, 1)
    reg1 = LinearRegression()
    reg1.fit(X1, y)
    y_pred1 = reg1.predict(X1)

    print(f"\n  rating = {reg1.coef_[0]:.2f} * log(equal_cpl + 1) + {reg1.intercept_:.2f}")
    print(f"  R² Score: {r2_score(y, y_pred1):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred1)):.0f} rating points")
    print(f"  Correlation: {np.corrcoef(log_equal_cpl, ratings)[0,1]:.4f}")

    # =========================================================================
    # Model 2: Two-feature (log_cpl + best%)
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODEL 2: Rating from log(equal_cpl) + equal_best%")
    print("-" * 70)

    X2 = np.column_stack([log_equal_cpl, equal_best])
    reg2 = LinearRegression()
    reg2.fit(X2, y)
    y_pred2 = reg2.predict(X2)

    print(f"\n  rating = {reg2.coef_[0]:.2f}*log(e_cpl+1) + {reg2.coef_[1]:.2f}*e_best% + {reg2.intercept_:.2f}")
    print(f"  R² Score: {r2_score(y, y_pred2):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred2)):.0f} rating points")

    # =========================================================================
    # Model 3: Four-feature (log CPLs + best pcts)
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODEL 3: Rating from Both Benchmarks (log CPLs + best%)")
    print("-" * 70)

    X3 = np.column_stack([log_equal_cpl, log_blunder_cpl, equal_best, blunder_best])
    reg3 = LinearRegression()
    reg3.fit(X3, y)
    y_pred3 = reg3.predict(X3)

    print(f"\n  rating = {reg3.coef_[0]:.2f}*log(e_cpl+1) + {reg3.coef_[1]:.2f}*log(b_cpl+1) + {reg3.coef_[2]:.2f}*e_best% + {reg3.coef_[3]:.2f}*b_best% + {reg3.intercept_:.2f}")
    print(f"  R² Score: {r2_score(y, y_pred3):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred3)):.0f} rating points")

    # =========================================================================
    # Model 4: Polynomial degree 2 on log(e_cpl) + e_best
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODEL 4: Polynomial Degree 2 (log_cpl + best%)")
    print("-" * 70)

    X4_raw = np.column_stack([log_equal_cpl, equal_best])
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X4 = poly.fit_transform(X4_raw)
    reg4 = LinearRegression()
    reg4.fit(X4, y)
    y_pred4 = reg4.predict(X4)

    feature_names = poly.get_feature_names_out(['log_cpl', 'best_pct'])
    print(f"\n  Features and coefficients:")
    for fname, coef in zip(feature_names, reg4.coef_):
        print(f"    {fname}: {coef:.2f}")
    print(f"    intercept: {reg4.intercept_:.2f}")
    print(f"  R² Score: {r2_score(y, y_pred4):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred4)):.0f} rating points")

    # =========================================================================
    # Model 5: Combined linear (legal% + best% from both benchmarks)
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODEL 5: Combined Linear (legal% + best% from both benchmarks)")
    print("-" * 70)

    X5 = np.column_stack([blunder_legal, blunder_best, equal_legal, equal_best])
    reg5 = LinearRegression()
    reg5.fit(X5, y)
    y_pred5 = reg5.predict(X5)

    print(f"\n  rating = {reg5.coef_[0]:.2f}*b_legal + {reg5.coef_[1]:.2f}*b_best + {reg5.coef_[2]:.2f}*e_legal + {reg5.coef_[3]:.2f}*e_best + {reg5.intercept_:.2f}")
    print(f"  R² Score: {r2_score(y, y_pred5):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred5)):.0f} rating points")

    # =========================================================================
    # Model 6: Ridge regression (regularized, all features)
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODEL 6: Ridge Regression (regularized, all features)")
    print("-" * 70)

    X6 = np.column_stack([log_equal_cpl, log_blunder_cpl, equal_best, blunder_best, equal_legal, blunder_legal])
    reg6 = Ridge(alpha=1.0)
    reg6.fit(X6, y)
    y_pred6 = reg6.predict(X6)

    print(f"\n  Features: log(e_cpl), log(b_cpl), e_best, b_best, e_legal, b_legal")
    print(f"  Coefficients: {[f'{c:.2f}' for c in reg6.coef_]}")
    print(f"  R² Score: {r2_score(y, y_pred6):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred6)):.0f} rating points")

    # =========================================================================
    # Correlation Matrix
    # =========================================================================
    print("\n" + "=" * 70)
    print("CORRELATION MATRIX WITH RATING")
    print("=" * 70)

    metrics = np.column_stack([ratings, blunder_legal, blunder_best, log_blunder_cpl, equal_legal, equal_best, log_equal_cpl])
    metric_names = ["Rating", "B_Leg%", "B_Best%", "logB_CPL", "E_Leg%", "E_Best%", "logE_CPL"]

    corr_matrix = np.corrcoef(metrics.T)

    print("\n          ", end="")
    for name in metric_names:
        print(f"{name:>9}", end="")
    print()

    for i, name in enumerate(metric_names):
        print(f"  {name:<8}", end="")
        for j in range(len(metric_names)):
            print(f"{corr_matrix[i,j]:>9.3f}", end="")
        print()

    # =========================================================================
    # Predictions vs Actual (all models)
    # =========================================================================
    # Find best model by R²
    all_models = {
        "M1 log(e_cpl)": (y_pred1, r2_score(y, y_pred1)),
        "M2 2-feature": (y_pred2, r2_score(y, y_pred2)),
        "M3 4-feature": (y_pred3, r2_score(y, y_pred3)),
        "M4 poly-2": (y_pred4, r2_score(y, y_pred4)),
        "M5 linear": (y_pred5, r2_score(y, y_pred5)),
        "M6 ridge": (y_pred6, r2_score(y, y_pred6)),
    }
    best_model_name = max(all_models, key=lambda k: all_models[k][1])
    best_pred = all_models[best_model_name][0]

    print("\n" + "=" * 70)
    print(f"PREDICTIONS vs ACTUAL (Best Model: {best_model_name})")
    print("=" * 70)

    print(f"\n  {'Model':<42} {'Actual':>8} {'Pred':>8} {'Error':>8}")
    print("  " + "-" * 68)

    # Sort by actual rating
    results = [(names[i], ratings[i], best_pred[i], ratings[i] - best_pred[i])
               for i in range(len(names))]
    results.sort(key=lambda x: -x[1])

    for name, actual, pred, err in results:
        print(f"  {name:<42} {actual:>8.0f} {pred:>8.0f} {err:>+8.0f}")

    # =========================================================================
    # Model Comparison Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n  {'Model':<40} {'R²':>8} {'RMSE':>8}")
    print("  " + "-" * 58)
    for mname, (mpred, mr2) in all_models.items():
        mrmse = np.sqrt(mean_squared_error(y, mpred))
        marker = " <-- best" if mname == best_model_name else ""
        print(f"  {mname:<40} {mr2:>8.4f} {mrmse:>8.0f}{marker}")

    # =========================================================================
    # Key Findings
    # =========================================================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    corr_rating_log_e_cpl = np.corrcoef(ratings, log_equal_cpl)[0,1]
    corr_rating_e_best = np.corrcoef(ratings, equal_best)[0,1]
    corr_rating_b_best = np.corrcoef(ratings, blunder_best)[0,1]

    print(f"""
  1. BEST SINGLE PREDICTOR: log(equal_cpl)
     - log(e_cpl) R² = {r2_score(y, y_pred1):.4f} (correlation: {corr_rating_log_e_cpl:.3f})
     - e_best%    R² = {r2_score(y, LinearRegression().fit(equal_best.reshape(-1,1), y).predict(equal_best.reshape(-1,1))):.4f} (correlation: {corr_rating_e_best:.3f})
     - b_best%    R² = {r2_score(y, LinearRegression().fit(blunder_best.reshape(-1,1), y).predict(blunder_best.reshape(-1,1))):.4f} (correlation: {corr_rating_b_best:.3f})

  2. FOUR-FEATURE LOG MODEL (log CPLs + best%)
     - R² = {r2_score(y, y_pred3):.4f}
     - RMSE = {np.sqrt(mean_squared_error(y, y_pred3)):.0f} rating points

  3. POLYNOMIAL DEGREE 2
     - R² = {r2_score(y, y_pred4):.4f}
     - RMSE = {np.sqrt(mean_squared_error(y, y_pred4)):.0f} rating points

  4. INTERPRETATION
     - Position benchmark explains {r2_score(y, best_pred)*100:.1f}% of rating variance
     - RMSE of {np.sqrt(mean_squared_error(y, best_pred)):.0f} rating points
     - {len(names)} models in training set
""")

    return {
        "best_r2": r2_score(y, best_pred),
        "best_rmse": np.sqrt(mean_squared_error(y, best_pred)),
    }


def main():
    data = load_results()
    models = extract_model_features(data)

    if len(models) < 3:
        print(f"Error: Only {len(models)} models found in both datasets. Need at least 3.")
        return

    results = fit_regression_models(models)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
