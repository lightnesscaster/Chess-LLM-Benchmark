"""
Regression analysis of position benchmark results.

Fits various regression models to understand relationships between:
- Legal move percentage and CPL
- Performance on blunder vs equal positions
- Model type (LLM vs engine) effects
"""

import json
import numpy as np
from pathlib import Path
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


def load_results():
    """Load both benchmark result files."""
    base_path = Path(__file__).parent

    # Load blunder results
    with open(base_path / "results_combined.json") as f:
        blunder_data = json.load(f)

    # Load equal position results
    with open(base_path / "equal_results.json") as f:
        equal_data = json.load(f)

    return blunder_data, equal_data


def extract_model_features(blunder_data, equal_data):
    """Extract features for each model from both benchmarks."""
    models = []

    # Get blunder results
    blunder_results = blunder_data.get("results", blunder_data)

    for model_name in blunder_results:
        blunder_stats = blunder_results[model_name]

        # Check if model exists in equal results
        if model_name not in equal_data:
            continue

        equal_stats = equal_data[model_name].get("summary", equal_data[model_name])

        model_type = blunder_stats.get("type", "llm")
        is_llm = 1 if model_type == "llm" else 0

        models.append({
            "name": model_name,
            "is_llm": is_llm,
            # Blunder benchmark metrics
            "blunder_legal_pct": blunder_stats["legal_pct"],
            "blunder_best_pct": blunder_stats["best_pct"],
            "blunder_avg_cpl": blunder_stats["avg_cpl"],
            "blunder_avoided_pct": blunder_stats["avoided_pct"],
            "blunder_median_cpl": blunder_stats["median_cpl"],
            # Equal position benchmark metrics
            "equal_legal_pct": equal_stats["legal_pct"],
            "equal_best_pct": equal_stats["best_pct"],
            "equal_avg_cpl": equal_stats["avg_cpl"],
            "equal_avoided_pct": equal_stats.get("avoided_pct", 100.0),
            "equal_median_cpl": equal_stats.get("median_cpl", equal_stats["avg_cpl"]),
        })

    return models


def fit_regression_models(models):
    """Fit various regression models and report results."""
    print("=" * 70)
    print("REGRESSION ANALYSIS OF BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\nDataset: {len(models)} models with metrics from blunder and equal position benchmarks\n")

    # Convert to numpy arrays
    names = [m["name"] for m in models]
    is_llm = np.array([m["is_llm"] for m in models]).reshape(-1, 1)

    blunder_legal = np.array([m["blunder_legal_pct"] for m in models])
    blunder_best = np.array([m["blunder_best_pct"] for m in models])
    blunder_cpl = np.array([m["blunder_avg_cpl"] for m in models])
    blunder_avoided = np.array([m["blunder_avoided_pct"] for m in models])

    equal_legal = np.array([m["equal_legal_pct"] for m in models])
    equal_best = np.array([m["equal_best_pct"] for m in models])
    equal_cpl = np.array([m["equal_avg_cpl"] for m in models])

    # =========================================================================
    # Model 1: Predict equal_cpl from blunder_cpl (cross-benchmark correlation)
    # =========================================================================
    print("-" * 70)
    print("MODEL 1: Predict Equal Position CPL from Blunder CPL")
    print("-" * 70)

    X = blunder_cpl.reshape(-1, 1)
    y = equal_cpl

    reg1 = LinearRegression()
    reg1.fit(X, y)
    y_pred = reg1.predict(X)

    print(f"  Linear Regression: equal_cpl = {reg1.coef_[0]:.4f} * blunder_cpl + {reg1.intercept_:.2f}")
    print(f"  R² Score: {r2_score(y, y_pred):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.2f}")
    print(f"  Correlation coefficient: {np.corrcoef(blunder_cpl, equal_cpl)[0,1]:.4f}")

    # =========================================================================
    # Model 2: Predict CPL from legal_pct (both benchmarks)
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODEL 2: Predict CPL from Legal Move Percentage")
    print("-" * 70)

    # Blunder benchmark
    X_b = blunder_legal.reshape(-1, 1)
    y_b = blunder_cpl
    reg2a = LinearRegression()
    reg2a.fit(X_b, y_b)
    y_pred_b = reg2a.predict(X_b)

    print(f"\n  Blunder Benchmark:")
    print(f"    blunder_cpl = {reg2a.coef_[0]:.2f} * legal_pct + {reg2a.intercept_:.2f}")
    print(f"    R² Score: {r2_score(y_b, y_pred_b):.4f}")
    print(f"    Interpretation: Each 1% increase in legal moves reduces CPL by {-reg2a.coef_[0]:.1f}")

    # Equal benchmark
    X_e = equal_legal.reshape(-1, 1)
    y_e = equal_cpl
    reg2b = LinearRegression()
    reg2b.fit(X_e, y_e)
    y_pred_e = reg2b.predict(X_e)

    print(f"\n  Equal Positions Benchmark:")
    print(f"    equal_cpl = {reg2b.coef_[0]:.2f} * legal_pct + {reg2b.intercept_:.2f}")
    print(f"    R² Score: {r2_score(y_e, y_pred_e):.4f}")
    print(f"    Interpretation: Each 1% increase in legal moves reduces CPL by {-reg2b.coef_[0]:.1f}")

    # =========================================================================
    # Model 3: Multi-feature model for blunder CPL
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODEL 3: Multi-Feature Model for Blunder CPL")
    print("-" * 70)

    X_multi = np.column_stack([blunder_legal, blunder_best, is_llm])
    y_multi = blunder_cpl

    reg3 = LinearRegression()
    reg3.fit(X_multi, y_multi)
    y_pred_multi = reg3.predict(X_multi)

    print(f"\n  blunder_cpl = {reg3.coef_[0]:.2f}*legal_pct + {reg3.coef_[1]:.2f}*best_pct + {reg3.coef_[2]:.2f}*is_llm + {reg3.intercept_:.2f}")
    print(f"  R² Score: {r2_score(y_multi, y_pred_multi):.4f}")
    print(f"\n  Feature Importance:")
    print(f"    - legal_pct coefficient: {reg3.coef_[0]:.2f} (higher legal% -> lower CPL)")
    print(f"    - best_pct coefficient: {reg3.coef_[1]:.2f} (higher best% -> {'lower' if reg3.coef_[1] < 0 else 'higher'} CPL)")
    print(f"    - is_llm coefficient: {reg3.coef_[2]:.2f} (LLMs have {'higher' if reg3.coef_[2] > 0 else 'lower'} CPL than engines)")

    # =========================================================================
    # Model 4: Polynomial regression (legal_pct vs CPL)
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODEL 4: Polynomial Regression (Legal% -> CPL)")
    print("-" * 70)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(blunder_legal.reshape(-1, 1))

    reg4 = LinearRegression()
    reg4.fit(X_poly, blunder_cpl)
    y_pred_poly = reg4.predict(X_poly)

    print(f"\n  blunder_cpl = {reg4.coef_[1]:.4f}*legal²  + {reg4.coef_[0]:.2f}*legal + {reg4.intercept_:.2f}")
    print(f"  R² Score: {r2_score(blunder_cpl, y_pred_poly):.4f}")
    print(f"  (Compared to linear R² = {r2_score(y_b, y_pred_b):.4f})")

    # =========================================================================
    # Model 5: LLM-only analysis
    # =========================================================================
    print("\n" + "-" * 70)
    print("MODEL 5: LLM-Only Analysis (excluding engines)")
    print("-" * 70)

    llm_mask = np.array([m["is_llm"] == 1 for m in models])
    llm_models = [m for m in models if m["is_llm"] == 1]

    if len(llm_models) >= 3:
        llm_blunder_legal = np.array([m["blunder_legal_pct"] for m in llm_models])
        llm_blunder_cpl = np.array([m["blunder_avg_cpl"] for m in llm_models])
        llm_equal_cpl = np.array([m["equal_avg_cpl"] for m in llm_models])

        # Legal -> CPL for LLMs only
        reg5 = LinearRegression()
        reg5.fit(llm_blunder_legal.reshape(-1, 1), llm_blunder_cpl)
        y_pred_llm = reg5.predict(llm_blunder_legal.reshape(-1, 1))

        print(f"\n  {len(llm_models)} LLMs in dataset")
        print(f"  blunder_cpl = {reg5.coef_[0]:.2f} * legal_pct + {reg5.intercept_:.2f}")
        print(f"  R² Score: {r2_score(llm_blunder_cpl, y_pred_llm):.4f}")

        # Cross-benchmark correlation for LLMs
        corr_llm = np.corrcoef(llm_blunder_cpl, llm_equal_cpl)[0,1]
        print(f"  Blunder vs Equal CPL correlation (LLMs only): {corr_llm:.4f}")

    # =========================================================================
    # Summary Statistics
    # =========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    print("\n  Blunder Benchmark:")
    print(f"    Mean CPL: {np.mean(blunder_cpl):.1f}")
    print(f"    Std CPL: {np.std(blunder_cpl):.1f}")
    print(f"    Mean Legal%: {np.mean(blunder_legal):.1f}%")
    print(f"    Mean Best%: {np.mean(blunder_best):.1f}%")

    print("\n  Equal Positions Benchmark:")
    print(f"    Mean CPL: {np.mean(equal_cpl):.1f}")
    print(f"    Std CPL: {np.std(equal_cpl):.1f}")
    print(f"    Mean Legal%: {np.mean(equal_legal):.1f}%")
    print(f"    Mean Best%: {np.mean(equal_best):.1f}%")

    # Correlation matrix
    print("\n" + "-" * 70)
    print("CORRELATION MATRIX")
    print("-" * 70)

    metrics = np.column_stack([
        blunder_legal, blunder_cpl, equal_legal, equal_cpl, is_llm.flatten()
    ])
    metric_names = ["B_Legal%", "B_CPL", "E_Legal%", "E_CPL", "is_LLM"]

    corr_matrix = np.corrcoef(metrics.T)

    print("\n           ", end="")
    for name in metric_names:
        print(f"{name:>10}", end="")
    print()

    for i, name in enumerate(metric_names):
        print(f"  {name:<8}", end="")
        for j in range(len(metric_names)):
            print(f"{corr_matrix[i,j]:>10.3f}", end="")
        print()

    # =========================================================================
    # Model Predictions vs Actual
    # =========================================================================
    print("\n" + "=" * 70)
    print("MODEL PREDICTIONS vs ACTUAL (Model 1: Blunder CPL -> Equal CPL)")
    print("=" * 70)

    predictions = reg1.predict(blunder_cpl.reshape(-1, 1))

    print(f"\n  {'Model':<35} {'Actual':>10} {'Predicted':>10} {'Error':>10}")
    print("  " + "-" * 65)

    # Sort by error
    errors = [(names[i], equal_cpl[i], predictions[i], equal_cpl[i] - predictions[i])
              for i in range(len(names))]
    errors.sort(key=lambda x: abs(x[3]))

    for name, actual, pred, err in errors:
        print(f"  {name:<35} {actual:>10.1f} {pred:>10.1f} {err:>+10.1f}")

    # =========================================================================
    # Key Findings
    # =========================================================================
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    corr_b_e = np.corrcoef(blunder_cpl, equal_cpl)[0,1]
    corr_legal_cpl = np.corrcoef(blunder_legal, blunder_cpl)[0,1]

    print(f"""
  1. CROSS-BENCHMARK CONSISTENCY
     - Blunder CPL strongly correlates with Equal CPL (r = {corr_b_e:.3f})
     - Models that perform well on blunders also perform well on equal positions
     - The linear model explains {r2_score(equal_cpl, reg1.predict(blunder_cpl.reshape(-1,1)))*100:.1f}% of variance

  2. LEGAL MOVES ARE THE KEY PREDICTOR
     - Legal% vs CPL correlation: r = {corr_legal_cpl:.3f}
     - Each 1% increase in legal moves reduces CPL by ~{-reg2a.coef_[0]:.0f} centipawns
     - Legal move ability explains {r2_score(blunder_cpl, reg2a.predict(blunder_legal.reshape(-1,1)))*100:.1f}% of CPL variance

  3. LLM vs ENGINE GAP
     - Being an LLM adds ~{reg3.coef_[2]:.0f} CPL on average (after controlling for legal%)
     - This represents the "chess understanding" gap beyond just move legality

  4. POLYNOMIAL FIT
     - Quadratic term coefficient: {reg4.coef_[1]:.4f}
     - {'CPL drops faster as legal% approaches 100%' if reg4.coef_[1] > 0 else 'CPL drops more slowly as legal% approaches 100%'}
     - Polynomial R² = {r2_score(blunder_cpl, y_pred_poly):.4f} vs Linear R² = {r2_score(blunder_cpl, reg2a.predict(blunder_legal.reshape(-1,1))):.4f}
""")

    return {
        "model1_r2": r2_score(equal_cpl, reg1.predict(blunder_cpl.reshape(-1,1))),
        "model2_r2": r2_score(blunder_cpl, reg2a.predict(blunder_legal.reshape(-1,1))),
        "model3_r2": r2_score(y_multi, y_pred_multi),
        "correlation_blunder_equal": corr_b_e,
        "correlation_legal_cpl": corr_legal_cpl,
    }


def main():
    blunder_data, equal_data = load_results()
    models = extract_model_features(blunder_data, equal_data)

    if len(models) < 3:
        print(f"Error: Only {len(models)} models found in both datasets. Need at least 3.")
        return

    results = fit_regression_models(models)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
