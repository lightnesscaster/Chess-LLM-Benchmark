"""
Rating prediction formulas from position benchmark data.

These formulas predict Glicko-2 rating from position benchmark metrics.
Derived from 22 models tested on 200 equal positions and 100 blunder positions.
Data lives in unified results.json with per-type summary breakdowns.

Updated: Now includes gpt-5-chat in training data.
"""

import numpy as np


# =============================================================================
# FORMULA 1: Simple Log-Linear (Best single feature)
# =============================================================================
# R² = 0.9808, RMSE = 111 rating points
#
# rating = -501.26 * log(equal_cpl + 1) + 3655.39

def predict_rating_simple(equal_cpl: float) -> float:
    """
    Predict rating from equal position CPL using log-linear model.

    Args:
        equal_cpl: Average centipawn loss on equal positions

    Returns:
        Predicted Glicko-2 rating
    """
    return -501.26 * np.log(equal_cpl + 1) + 3655.39


# =============================================================================
# FORMULA 2: Two-Feature Linear (Best practical)
# =============================================================================
# R² = 0.9844, RMSE = 100 rating points
#
# rating = -418.67 * log(equal_cpl + 1) + 8.53 * equal_best_pct + 2961.70

def predict_rating_two_feature(equal_cpl: float, equal_best_pct: float) -> float:
    """
    Predict rating from equal position CPL and best move percentage.

    Args:
        equal_cpl: Average centipawn loss on equal positions
        equal_best_pct: Percentage of best moves found (0-100)

    Returns:
        Predicted Glicko-2 rating
    """
    return -418.67 * np.log(equal_cpl + 1) + 8.53 * equal_best_pct + 2961.70


# =============================================================================
# FORMULA 3: Four-Feature Linear (Both benchmarks)
# =============================================================================
# R² = 0.9860, RMSE = 95 rating points
#
# rating = -366.34 * log(equal_cpl + 1)
#          - 66.03 * log(blunder_cpl + 1)
#          + 7.83 * equal_best_pct
#          + 4.36 * blunder_best_pct
#          + 3148.48

def predict_rating_four_feature(
    equal_cpl: float,
    blunder_cpl: float,
    equal_best_pct: float,
    blunder_best_pct: float
) -> float:
    """
    Predict rating using metrics from both equal and blunder benchmarks.

    Args:
        equal_cpl: Average CPL on equal positions
        blunder_cpl: Average CPL on blunder positions
        equal_best_pct: Best move % on equal positions (0-100)
        blunder_best_pct: Best move % on blunder positions (0-100)

    Returns:
        Predicted Glicko-2 rating
    """
    return (
        -366.34 * np.log(equal_cpl + 1)
        - 66.03 * np.log(blunder_cpl + 1)
        + 7.83 * equal_best_pct
        + 4.36 * blunder_best_pct
        + 3148.48
    )


# =============================================================================
# FORMULA 4: Polynomial Degree 2 (Best accuracy)
# =============================================================================
# R² = 0.9897, RMSE = 82 rating points
#
# Features: log(e_cpl), e_best, log(e_cpl)², log(e_cpl)*e_best, e_best²

POLY2_COEFFICIENTS = {
    'log_cpl': -2629.84,
    'best_pct': -147.08,
    'log_cpl_squared': 145.81,
    'log_cpl_times_best': 19.58,
    'best_pct_squared': 0.77,
    'intercept': 11318.16,
}

def predict_rating_poly2(equal_cpl: float, equal_best_pct: float) -> float:
    """
    Predict rating using polynomial (degree 2) model.

    Args:
        equal_cpl: Average centipawn loss on equal positions
        equal_best_pct: Percentage of best moves found (0-100)

    Returns:
        Predicted Glicko-2 rating
    """
    log_cpl = np.log(equal_cpl + 1)

    return (
        POLY2_COEFFICIENTS['log_cpl'] * log_cpl
        + POLY2_COEFFICIENTS['best_pct'] * equal_best_pct
        + POLY2_COEFFICIENTS['log_cpl_squared'] * log_cpl ** 2
        + POLY2_COEFFICIENTS['log_cpl_times_best'] * log_cpl * equal_best_pct
        + POLY2_COEFFICIENTS['best_pct_squared'] * equal_best_pct ** 2
        + POLY2_COEFFICIENTS['intercept']
    )


# =============================================================================
# SUMMARY TABLE
# =============================================================================
"""
| Model                    | R²     | RMSE | Formula Complexity |
|--------------------------|--------|------|-------------------|
| Simple log(CPL)          | 0.9808 | 111  | 1 feature         |
| Two-feature              | 0.9844 | 100  | 2 features        |
| Four-feature             | 0.9860 | 95   | 4 features        |
| Polynomial degree 2      | 0.9897 | 82   | 5 terms           |

Key insights:
- log(CPL) is essential - linear CPL gives R² = 0.68, log gives R² = 0.98
- Equal positions are better predictors than blunder positions
- Each 1% increase in best_move% ≈ +48 rating points (simple regression)
- Polynomial helps capture non-linear effects at extremes
- Models trained on 22 data points (21 original + gpt-5-chat)

Data source: unified results.json with per-type breakdowns (summary.blunder, summary.equal).
Run regression_analysis.py to refit models from results.json.
"""


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Example: A model with 500 CPL and 20% best moves on equal positions
    equal_cpl = 500
    equal_best = 20
    blunder_cpl = 3000
    blunder_best = 5

    print("Example predictions for CPL=500, Best%=20:")
    print(f"  Simple:      {predict_rating_simple(equal_cpl):.0f}")
    print(f"  Two-feature: {predict_rating_two_feature(equal_cpl, equal_best):.0f}")
    print(f"  Four-feature:{predict_rating_four_feature(equal_cpl, blunder_cpl, equal_best, blunder_best):.0f}")
    print(f"  Polynomial:  {predict_rating_poly2(equal_cpl, equal_best):.0f}")

    # Test on gpt-5-chat (should be close to 276)
    print("\nVerification on gpt-5-chat (actual rating: 276):")
    gpt5_e_cpl = 904.2
    gpt5_e_best = 21.0
    gpt5_b_cpl = 5889.2
    gpt5_b_best = 12.0
    print(f"  Simple:      {predict_rating_simple(gpt5_e_cpl):.0f}")
    print(f"  Two-feature: {predict_rating_two_feature(gpt5_e_cpl, gpt5_e_best):.0f}")
    print(f"  Four-feature:{predict_rating_four_feature(gpt5_e_cpl, gpt5_b_cpl, gpt5_e_best, gpt5_b_best):.0f}")
    print(f"  Polynomial:  {predict_rating_poly2(gpt5_e_cpl, gpt5_e_best):.0f}")
