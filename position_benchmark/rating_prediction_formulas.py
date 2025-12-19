"""
Rating prediction formulas from position benchmark data.

These formulas predict Glicko-2 rating from position benchmark metrics.
Derived from 21 models tested on 100 equal positions and 100 blunder positions.
"""

import numpy as np


# =============================================================================
# FORMULA 1: Simple Log-Linear (Best single feature)
# =============================================================================
# R² = 0.9809, RMSE = 114 rating points
#
# rating = -501.23 * log(equal_cpl + 1) + 3653.55

def predict_rating_simple(equal_cpl: float) -> float:
    """
    Predict rating from equal position CPL using log-linear model.

    Args:
        equal_cpl: Average centipawn loss on equal positions

    Returns:
        Predicted Glicko-2 rating
    """
    return -501.23 * np.log(equal_cpl + 1) + 3653.55


# =============================================================================
# FORMULA 2: Two-Feature Linear (Best practical)
# =============================================================================
# R² = 0.9845, RMSE = 103 rating points
#
# rating = -417.5 * log(equal_cpl + 1) + 8.7 * equal_best_pct + 2952.3

def predict_rating_two_feature(equal_cpl: float, equal_best_pct: float) -> float:
    """
    Predict rating from equal position CPL and best move percentage.

    Args:
        equal_cpl: Average centipawn loss on equal positions
        equal_best_pct: Percentage of best moves found (0-100)

    Returns:
        Predicted Glicko-2 rating
    """
    return -417.5 * np.log(equal_cpl + 1) + 8.7 * equal_best_pct + 2952.3


# =============================================================================
# FORMULA 3: Four-Feature Linear (Both benchmarks)
# =============================================================================
# R² = 0.9860, RMSE = 97 rating points
#
# rating = -363.9 * log(equal_cpl + 1)
#          - 69.1 * log(blunder_cpl + 1)
#          + 8.0 * equal_best_pct
#          + 4.3 * blunder_best_pct
#          + 3157.0

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
        -363.9 * np.log(equal_cpl + 1)
        - 69.1 * np.log(blunder_cpl + 1)
        + 8.0 * equal_best_pct
        + 4.3 * blunder_best_pct
        + 3157.0
    )


# =============================================================================
# FORMULA 4: Polynomial Degree 2 (Best accuracy without overfitting)
# =============================================================================
# R² = 0.9900, RMSE = 82 rating points
#
# Features: log(e_cpl), e_best, log(e_cpl)², log(e_cpl)*e_best, e_best²
# Coefficients fitted from sklearn PolynomialFeatures(degree=2)

POLY2_COEFFICIENTS = {
    'log_cpl': -2745.28,
    'best_pct': -156.06,
    'log_cpl_squared': 153.21,
    'log_cpl_times_best': 20.51,
    'best_pct_squared': 0.83,
    'intercept': 11772.84,
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
| Model                    | R²     | RMSE | Max Error | Formula Complexity |
|--------------------------|--------|------|-----------|-------------------|
| Simple log(CPL)          | 0.9809 | 114  | 323       | 1 feature         |
| Two-feature              | 0.9845 | 103  | ~250      | 2 features        |
| Four-feature             | 0.9860 | 97   | ~200      | 4 features        |
| Polynomial degree 2      | 0.9900 | 82   | 244       | 5 terms           |

Key insights:
- log(CPL) is essential - linear CPL gives R² = 0.68, log gives R² = 0.98
- Equal positions are better predictors than blunder positions
- Each 1% increase in best_move% ≈ +48 rating points
- Polynomial helps capture non-linear effects at extremes
- qwen3-235b remains biggest outlier (-244 to -323 error across all models)
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
