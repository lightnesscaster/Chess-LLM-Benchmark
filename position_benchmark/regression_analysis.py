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
BENCHMARK_RATINGS = {
    "eubos": 2211,
    "maia-1900": 1816,
    "survival-bot": 1649,
    "maia-1100": 1628,
    "random-bot": 400,
    "gemini-2.5-flash (no thinking)": 355,
    "gemini-2.0-flash-001": 264,
    "kimi-k2": -144,
    "deepseek-v3.2 (no thinking)": -173,
    "llama-4-maverick": -175,
    "deepseek-chat-v3-0324": -189,
    "kimi-k2-0905": -203,
    "deepseek-v3.1-terminus (no thinking)": -239,
    "gpt-3.5-turbo-0613": -243,
    "deepseek-r1-distill-qwen-32b": -254,
    "mistral-medium-3": -255,
    "glm-4.6 (no thinking)": -325,
    "deepseek-chat-v3.1 (no thinking)": -341,
    "qwen3-235b-a22b-2507": -377,
    "gpt-3.5-turbo": -500,
    "llama-3.3-70b-instruct": -500,
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

    # =========================================================================
    # Model 1: Predict Rating from Blunder CPL
    # =========================================================================
    print("-" * 70)
    print("MODEL 1: Rating from Blunder CPL (single feature)")
    print("-" * 70)

    X = blunder_cpl.reshape(-1, 1)
    y = ratings

    reg1 = LinearRegression()
    reg1.fit(X, y)
    y_pred1 = reg1.predict(X)

    print(f"\n  rating = {reg1.coef_[0]:.4f} * blunder_cpl + {reg1.intercept_:.2f}")
    print(f"  R² Score: {r2_score(y, y_pred1):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred1)):.0f} rating points")
    print(f"  Correlation: {np.corrcoef(blunder_cpl, ratings)[0,1]:.4f}")

    # =========================================================================
    # Model 2: Predict Rating from Legal%
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODEL 2: Rating from Legal Move % (single feature)")
    print("-" * 70)

    X2 = blunder_legal.reshape(-1, 1)
    reg2 = LinearRegression()
    reg2.fit(X2, y)
    y_pred2 = reg2.predict(X2)

    print(f"\n  rating = {reg2.coef_[0]:.2f} * legal_pct + {reg2.intercept_:.2f}")
    print(f"  R² Score: {r2_score(y, y_pred2):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred2)):.0f} rating points")
    print(f"  Interpretation: Each 1% increase in legal moves = +{reg2.coef_[0]:.1f} rating points")

    # =========================================================================
    # Model 3: Predict Rating from Best%
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODEL 3: Rating from Best Move % (single feature)")
    print("-" * 70)

    X3 = blunder_best.reshape(-1, 1)
    reg3 = LinearRegression()
    reg3.fit(X3, y)
    y_pred3 = reg3.predict(X3)

    print(f"\n  rating = {reg3.coef_[0]:.2f} * best_pct + {reg3.intercept_:.2f}")
    print(f"  R² Score: {r2_score(y, y_pred3):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred3)):.0f} rating points")
    print(f"  Interpretation: Each 1% increase in best moves = +{reg3.coef_[0]:.1f} rating points")

    # =========================================================================
    # Model 4: Multi-feature model (Legal% + Best% + CPL)
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODEL 4: Rating from Multiple Features")
    print("-" * 70)

    X4 = np.column_stack([blunder_legal, blunder_best, blunder_cpl])
    reg4 = LinearRegression()
    reg4.fit(X4, y)
    y_pred4 = reg4.predict(X4)

    print(f"\n  rating = {reg4.coef_[0]:.2f}*legal% + {reg4.coef_[1]:.2f}*best% + {reg4.coef_[2]:.4f}*cpl + {reg4.intercept_:.2f}")
    print(f"  R² Score: {r2_score(y, y_pred4):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred4)):.0f} rating points")

    # =========================================================================
    # Model 5: Combined benchmarks (blunder + equal)
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODEL 5: Rating from Both Benchmarks (best model)")
    print("-" * 70)

    X5 = np.column_stack([blunder_legal, blunder_best, equal_legal, equal_best])
    reg5 = LinearRegression()
    reg5.fit(X5, y)
    y_pred5 = reg5.predict(X5)

    print(f"\n  rating = {reg5.coef_[0]:.2f}*b_legal + {reg5.coef_[1]:.2f}*b_best + {reg5.coef_[2]:.2f}*e_legal + {reg5.coef_[3]:.2f}*e_best + {reg5.intercept_:.2f}")
    print(f"  R² Score: {r2_score(y, y_pred5):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred5)):.0f} rating points")

    # =========================================================================
    # Model 6: Ridge regression (regularized)
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODEL 6: Ridge Regression (regularized)")
    print("-" * 70)

    X6 = np.column_stack([blunder_legal, blunder_best, blunder_cpl, equal_legal, equal_best, equal_cpl])
    reg6 = Ridge(alpha=1.0)
    reg6.fit(X6, y)
    y_pred6 = reg6.predict(X6)

    print(f"\n  Features: b_legal, b_best, b_cpl, e_legal, e_best, e_cpl")
    print(f"  Coefficients: {[f'{c:.2f}' for c in reg6.coef_]}")
    print(f"  R² Score: {r2_score(y, y_pred6):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred6)):.0f} rating points")

    # =========================================================================
    # Correlation Matrix
    # =========================================================================
    print("\n" + "=" * 70)
    print("CORRELATION MATRIX WITH RATING")
    print("=" * 70)

    metrics = np.column_stack([ratings, blunder_legal, blunder_best, blunder_cpl, equal_legal, equal_best, equal_cpl])
    metric_names = ["Rating", "B_Leg%", "B_Best%", "B_CPL", "E_Leg%", "E_Best%", "E_CPL"]

    corr_matrix = np.corrcoef(metrics.T)

    print("\n          ", end="")
    for name in metric_names:
        print(f"{name:>8}", end="")
    print()

    for i, name in enumerate(metric_names):
        print(f"  {name:<7}", end="")
        for j in range(len(metric_names)):
            print(f"{corr_matrix[i,j]:>8.3f}", end="")
        print()

    # =========================================================================
    # Predictions vs Actual
    # =========================================================================
    print("\n" + "=" * 70)
    print("PREDICTIONS vs ACTUAL (Best Model: #5)")
    print("=" * 70)

    print(f"\n  {'Model':<35} {'Actual':>8} {'Pred':>8} {'Error':>8}")
    print("  " + "-" * 60)

    # Sort by actual rating
    results = [(names[i], ratings[i], y_pred5[i], ratings[i] - y_pred5[i])
               for i in range(len(names))]
    results.sort(key=lambda x: -x[1])

    for name, actual, pred, err in results:
        print(f"  {name:<35} {actual:>8.0f} {pred:>8.0f} {err:>+8.0f}")

    # =========================================================================
    # Key Findings
    # =========================================================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    corr_rating_legal = np.corrcoef(ratings, blunder_legal)[0,1]
    corr_rating_best = np.corrcoef(ratings, blunder_best)[0,1]
    corr_rating_cpl = np.corrcoef(ratings, blunder_cpl)[0,1]

    # Find best single predictor
    r2_scores = {
        "Legal%": r2_score(y, y_pred2),
        "Best%": r2_score(y, y_pred3),
        "CPL": r2_score(y, y_pred1),
    }
    best_single = max(r2_scores, key=r2_scores.get)

    print(f"""
  1. BEST SINGLE PREDICTOR: {best_single}
     - Legal% R² = {r2_scores['Legal%']:.4f} (correlation: {corr_rating_legal:.3f})
     - Best%  R² = {r2_scores['Best%']:.4f} (correlation: {corr_rating_best:.3f})
     - CPL    R² = {r2_scores['CPL']:.4f} (correlation: {corr_rating_cpl:.3f})

  2. MULTI-FEATURE MODEL (Legal + Best + CPL)
     - R² = {r2_score(y, y_pred4):.4f}
     - RMSE = {np.sqrt(mean_squared_error(y, y_pred4)):.0f} rating points

  3. COMBINED BENCHMARKS MODEL (Blunder + Equal metrics)
     - R² = {r2_score(y, y_pred5):.4f}
     - RMSE = {np.sqrt(mean_squared_error(y, y_pred5)):.0f} rating points
     - Adding equal positions improves R² by {(r2_score(y, y_pred5) - r2_score(y, y_pred4))*100:.1f}%

  4. INTERPRETATION
     - Each 1% increase in legal moves ≈ +{reg2.coef_[0]:.0f} rating points
     - Each 1% increase in best moves ≈ +{reg3.coef_[0]:.0f} rating points
     - Position benchmark explains {r2_score(y, y_pred5)*100:.1f}% of rating variance
""")

    return {
        "best_r2": r2_score(y, y_pred5),
        "best_rmse": np.sqrt(mean_squared_error(y, y_pred5)),
        "legal_coef": reg2.coef_[0],
        "best_coef": reg3.coef_[0],
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
