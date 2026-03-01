"""
Rating prediction formulas from position benchmark data.

These formulas predict Glicko-2 rating from position benchmark metrics.
Derived from 30 models (RD < 100) tested on 50 equal positions at
Stockfish depth 30.

The recommended approach uses three features, all from the 50 equal positions:
- log(mean CPL) captures overall move quality
- pct_lt10 (% of positions with CPL < 10) captures consistency
- surv_40 (probability of surviving a 40-move game) captures illegal move impact

The survival feature models the 2-strikes-forfeit rule: a model that plays
illegal moves at rate p will survive N moves with probability
P(0 or 1 illegal in N) = (1-p)^N + N*p*(1-p)^(N-1). This is highly nonlinear:
2% illegal -> 81% survive, 4% -> 42%, 10% -> 8%.

Updated: Mar 2026
- All evaluations at depth 30 (was 16)
- Models with RD >= 100 excluded (unreliable game-play ratings)
- All features derived from equal positions only (50 positions)
- Survival probability uses is_legal field, not CPL heuristic
- LOO CV: R² = 0.9652, RMSE = 149 (30 models)
"""

import numpy as np
from scipy.stats import binom


# =============================================================================
# RECOMMENDED: log(mean CPL) + consistency + survival (Best validated accuracy)
# =============================================================================
# LOO CV:       R² = 0.9652, RMSE = 149 rating points  (30 models, RD < 100)
# Training fit:  R² = 0.9799, RMSE = 114 rating points
#
# rating = 1794.14 - 260.55 * log(mean_eq_cpl + 1) + 21.48 * pct_lt10 + 2.09 * surv_40
#
# Where:
#   mean_eq_cpl = average centipawn loss on the 50 equal positions
#   pct_lt10    = percentage of equal positions with CPL < 10 (0-100)
#   surv_40     = P(0 or 1 illegal moves in 40 moves) as percentage (0-100)
#                 computed from the is_legal field on the 50 equal positions

def survival_probability(illegal_rate: float, game_length: int = 40) -> float:
    """
    Compute probability of surviving a game under 2-strikes-forfeit rule.

    Uses binomial model: P(0 or 1 illegal moves in N moves).

    Args:
        illegal_rate: Per-move illegal move rate (0.0 to 1.0)
        game_length: Expected number of moves in a game

    Returns:
        Survival probability as percentage (0-100)
    """
    if illegal_rate <= 0:
        return 100.0
    if illegal_rate >= 1:
        return 0.0
    p = illegal_rate
    n = game_length
    return 100.0 * (binom.pmf(0, n, p) + binom.pmf(1, n, p))


def predict_rating(
    equal_cpl: float, pct_lt10: float, surv_40: float
) -> float:
    """
    Predict rating using equal position CPL, consistency, and survival.

    This is the recommended formula. It combines:
    - Overall move quality (log of mean CPL)
    - Consistency (% near-perfect moves with CPL < 10)
    - Game viability (survival probability under 2-strikes-forfeit)

    Args:
        equal_cpl: Average centipawn loss on the 50 equal positions
        pct_lt10: Percentage of equal positions with CPL < 10 (0-100)
        surv_40: Survival probability for 40-move game (0-100),
                 from survival_probability(illegal_rate, 40)

    Returns:
        Predicted Glicko-2 rating
    """
    return (
        1794.14
        - 260.55 * np.log(equal_cpl + 1)
        + 21.48 * pct_lt10
        + 2.09 * surv_40
    )


# =============================================================================
# 2-FEATURE: log(mean CPL) + consistency (No illegal move data needed)
# =============================================================================
# LOO CV:       R² = 0.9615, RMSE = 157 rating points  (30 models, RD < 100)
# Training fit:  R² = 0.9765, RMSE = 123 rating points
#
# rating = 2139.13 - 302.30 * log(mean_eq_cpl + 1) + 22.94 * pct_lt10
#
# Note: Use this when the is_legal field is not available.
#
# Where:
#   mean_eq_cpl = average centipawn loss on the 50 equal positions
#   pct_lt10    = percentage of equal positions with CPL < 10 (0-100)

def predict_rating_2feat(equal_cpl: float, pct_lt10: float) -> float:
    """
    Predict rating using equal position CPL and consistency only.

    Use when illegal move data (is_legal field) is not available.

    Args:
        equal_cpl: Average centipawn loss on the 50 equal positions
        pct_lt10: Percentage of equal positions with CPL < 10 (0-100)

    Returns:
        Predicted Glicko-2 rating
    """
    return 2139.13 - 302.30 * np.log(equal_cpl + 1) + 22.94 * pct_lt10


# =============================================================================
# SIMPLE: log(capped mean CPL) — Single feature fallback
# =============================================================================
# LOO CV:       R² = 0.9352, RMSE = 204 rating points  (30 models, RD < 100)
# Training fit:  R² = 0.9483, RMSE = 182 rating points
#
# rating = 4549.27 - 661.27 * log(capped_mean_eq_cpl + 1)
#
# Where:
#   capped_mean_eq_cpl = mean of min(cpl, 2000) for each equal position

def predict_rating_simple(capped_equal_cpl: float) -> float:
    """
    Predict rating from capped equal position CPL only.

    Use this when only CPL values are available (no consistency or
    illegal move data). CPL values should be capped at 2000 before
    averaging to reduce the impact of illegal moves.

    Args:
        capped_equal_cpl: Average of min(cpl, 2000) on the 50 equal positions

    Returns:
        Predicted Glicko-2 rating
    """
    return 4549.27 - 661.27 * np.log(capped_equal_cpl + 1)


# =============================================================================
# SUMMARY
# =============================================================================
"""
| Model                          | LOO R² | LOO RMSE | Inputs                                   |
|--------------------------------|--------|----------|------------------------------------------|
| Recommended (3-feat + surv)    | 0.9652 |      149 | log(eq CPL) + pct_lt10 + surv_40         |
| 2-feature                      | 0.9615 |      157 | log(eq CPL) + pct_lt10                   |
| Simple (1-feat)                | 0.9352 |      204 | log(capped eq CPL)                       |

Fitted on 30 models with RD < 100 (excluded gemini-3.1-pro-preview high/medium).
All three features derived from the 50 equal positions only.

Key insights (depth 30):
- log(mean CPL) captures overall move quality
- pct_lt10 (consistency) is the key 2nd feature — separates reliably decent
  models from those that oscillate between brilliant and catastrophic
- surv_40 (survival probability) captures the highly nonlinear impact of
  illegal moves under the 2-strikes-forfeit rule:
    0% illegal -> 100% survive, 2% -> 81%, 4% -> 42%, 10% -> 8%
- Illegal moves MUST be detected via the is_legal field, NOT via CPL > 5000
  (legal moves on blunder positions can exceed CPL 5000 when missing forced mate)
- Blunder positions add noise to CPL features at depth 30 and are excluded
- CPL capping at 2000 helps single-feature models handle illegal move outliers
- Gemini 3-pro reasoning models remain the largest outliers (~450 rating pts)

All evaluations use Stockfish depth 30.
Data source: results.json — 50 equal positions, 30 models with RD < 100.
"""


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Example: A model with 500 CPL, 10% near-perfect, 5% illegal rate
    equal_cpl = 500
    pct10 = 10
    illegal_rate = 0.05
    surv40 = survival_probability(illegal_rate, 40)

    print(f"Example predictions for eq_CPL=500, pct_lt10={pct10}%, illegal_rate={illegal_rate}:")
    print(f"  Recommended: {predict_rating(equal_cpl, pct10, surv40):.0f}")
    print(f"  2-feature:   {predict_rating_2feat(equal_cpl, pct10):.0f}")
    print(f"  Simple:      {predict_rating_simple(min(equal_cpl, 2000)):.0f}")

    # Survival probability examples
    print("\nSurvival probability (40-move game):")
    for rate in [0, 0.01, 0.02, 0.04, 0.05, 0.10, 0.20]:
        print(f"  {rate:5.1%} illegal -> {survival_probability(rate, 40):5.1f}% survive")

    # Verification on known models
    print("\nVerification on eubos (actual: 2211, 0% illegal):")
    s = survival_probability(0.0, 40)
    print(f"  Recommended: {predict_rating(19.8, 46.0, s):.0f}")
    print(f"  2-feature:   {predict_rating_2feat(19.8, 46.0):.0f}")

    print("\nVerification on gemini-3-pro-preview high (actual: 1998, 0% illegal):")
    s = survival_probability(0.0, 40)
    print(f"  Recommended: {predict_rating(487.4, 66.0, s):.0f}")
    print(f"  2-feature:   {predict_rating_2feat(487.4, 66.0):.0f}")

    print("\nVerification on gemini-3-pro-preview medium (actual: 537, 4.8% illegal):")
    s = survival_probability(0.048, 40)
    print(f"  Recommended: {predict_rating(487.4, 54.0, s):.0f}  (surv_40={s:.1f}%)")
    print(f"  2-feature:   {predict_rating_2feat(487.4, 54.0):.0f}")
