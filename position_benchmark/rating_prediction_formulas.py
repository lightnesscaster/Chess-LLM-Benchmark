"""
Rating prediction formulas from position benchmark data.

These formulas predict Glicko-2 rating from position benchmark metrics.
Derived from 37 models (RD < 100) tested on 50 equal positions at
Stockfish depth 30.

The recommended approach uses three features, all from the 50 equal positions:
- log(mean CPL) captures overall move quality
- best_pct (% of positions where model played best move) captures skill ceiling
- surv_40 (probability of surviving a 40-move game) captures illegal move impact

The survival feature models the 2-strikes-forfeit rule: a model that plays
illegal moves at rate p will survive N moves with probability
P(0 or 1 illegal in N) = (1-p)^N + N*p*(1-p)^(N-1). This is highly nonlinear:
2% illegal -> 81% survive, 4% -> 42%, 10% -> 8%.

Updated: Mar 2026
- 37 models with RD < 100 (up from 30)
- All evaluations at depth 30
- All features derived from equal positions only (50 positions)
- Survival probability uses is_legal field, not CPL heuristic
- best_pct replaces pct_lt10 as 2nd feature (better LOO CV)
- LOO CV: R² = 0.9184, RMSE = 237 (37 models)

Limitations:
- Position benchmark illegal rate (50 positions) is a noisy estimate of game
  illegal rate. Models like gemini-3.1-flash-lite-preview show 94% legal on
  the benchmark but only 88% legal in games, causing large prediction errors.
- Reasoning models (gemini-3-pro, gpt-5.1 high) consistently outperform their
  position benchmark scores by 400-700 rating points.
- Using actual game legal rates instead of position benchmark rates improves
  LOO R² from 0.92 to 0.94 (3-feature) or 0.95 (4-feature).
"""

import numpy as np
from scipy.stats import binom


# =============================================================================
# RECOMMENDED: log(mean CPL) + best% + survival (Best validated accuracy)
# =============================================================================
# LOO CV:       R² = 0.9184, RMSE = 237 rating points  (37 models, RD < 100)
# Training fit:  R² = 0.9392, RMSE = 204 rating points
#
# rating = 1298.57 - 200.43 * log(mean_eq_cpl + 1) + 15.39 * best_pct + 5.85 * surv_40
#
# Where:
#   mean_eq_cpl = average centipawn loss on the 50 equal positions
#   best_pct    = percentage of equal positions where model played the best move (0-100)
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
    equal_cpl: float, best_pct: float, surv_40: float
) -> float:
    """
    Predict rating using equal position CPL, best move %, and survival.

    This is the recommended formula. It combines:
    - Overall move quality (log of mean CPL)
    - Skill ceiling (% positions where best move was played)
    - Game viability (survival probability under 2-strikes-forfeit)

    Args:
        equal_cpl: Average centipawn loss on the 50 equal positions
        best_pct: Percentage of equal positions where best move played (0-100)
        surv_40: Survival probability for 40-move game (0-100),
                 from survival_probability(illegal_rate, 40)

    Returns:
        Predicted Glicko-2 rating
    """
    return (
        1298.57
        - 200.43 * np.log(equal_cpl + 1)
        + 15.39 * best_pct
        + 5.85 * surv_40
    )


# =============================================================================
# 2-FEATURE: best% + survival (No CPL needed, simpler)
# =============================================================================
# LOO CV:       R² = 0.9033, RMSE = 258 rating points  (37 models, RD < 100)
# Training fit:  R² = 0.9226, RMSE = 230 rating points
#
# rating = 22.44 * best_pct + 8.46 * surv_40 - 357.18
#
# Note: Surprisingly competitive with the 3-feature model. Use when CPL
# data is unreliable or unavailable.
#
# Where:
#   best_pct = percentage of equal positions where best move played (0-100)
#   surv_40  = P(0 or 1 illegal moves in 40 moves) as percentage (0-100)

def predict_rating_2feat(best_pct: float, surv_40: float) -> float:
    """
    Predict rating using best move % and survival only.

    Surprisingly close to the 3-feature model. Use when CPL data is
    unreliable or unavailable.

    Args:
        best_pct: Percentage of equal positions where best move played (0-100)
        surv_40: Survival probability for 40-move game (0-100),
                 from survival_probability(illegal_rate, 40)

    Returns:
        Predicted Glicko-2 rating
    """
    return 22.44 * best_pct + 8.46 * surv_40 - 357.18


# =============================================================================
# 2-FEATURE ALT: log(mean CPL) + best% (No illegal move data needed)
# =============================================================================
# LOO CV:       R² = 0.8833, RMSE = 283 rating points  (37 models, RD < 100)
# Training fit:  R² = 0.9074, RMSE = 252 rating points
#
# rating = 2732.43 - 370.90 * log(mean_eq_cpl + 1) + 14.65 * best_pct
#
# Note: Use this when the is_legal field is not available.

def predict_rating_2feat_cpl(equal_cpl: float, best_pct: float) -> float:
    """
    Predict rating using equal position CPL and best move %.

    Use when illegal move data (is_legal field) is not available.

    Args:
        equal_cpl: Average centipawn loss on the 50 equal positions
        best_pct: Percentage of equal positions where best move played (0-100)

    Returns:
        Predicted Glicko-2 rating
    """
    return 2732.43 - 370.90 * np.log(equal_cpl + 1) + 14.65 * best_pct


# =============================================================================
# SIMPLE: log(mean CPL) — Single feature fallback
# =============================================================================
# LOO CV:       R² = 0.8507, RMSE = 320 rating points  (37 models, RD < 100)
# Training fit:  R² = 0.8706, RMSE = 298 rating points
#
# rating = 4475.64 - 574.19 * log(mean_eq_cpl + 1)

def predict_rating_simple(equal_cpl: float) -> float:
    """
    Predict rating from equal position CPL only.

    Single feature fallback. Use when only CPL values are available.

    Args:
        equal_cpl: Average centipawn loss on the 50 equal positions

    Returns:
        Predicted Glicko-2 rating
    """
    return 4475.64 - 574.19 * np.log(equal_cpl + 1)


# =============================================================================
# SUMMARY
# =============================================================================
"""
| Model                          | LOO R² | LOO RMSE | Inputs                                   |
|--------------------------------|--------|----------|------------------------------------------|
| Recommended (3-feat)           | 0.9184 |      237 | log(eq CPL) + best_pct + surv_40         |
| 2-feature (best + surv)       | 0.9033 |      258 | best_pct + surv_40                       |
| 2-feature alt (CPL + best)    | 0.8833 |      283 | log(eq CPL) + best_pct                   |
| Simple (1-feat)                | 0.8507 |      320 | log(eq CPL)                              |

Fitted on 37 models with RD < 100.
All features derived from the 50 equal positions only.

Key insights (depth 30):
- log(mean CPL) captures overall move quality
- best_pct (% best moves) replaces pct_lt10 as the key 2nd feature
- surv_40 (survival probability) captures illegal move impact, but position
  benchmark illegal rates are noisy (50 positions). Models with ~5-10%
  illegal rates have wide confidence intervals.
- Position benchmark illegal rates can diverge significantly from game rates:
  e.g. gemini-3.1-flash-lite-preview: 6% pos illegal vs 12% game illegal
- Using actual game legal rates improves 3-feature LOO R² from 0.92 to 0.94
- Reasoning models consistently outperform position benchmark predictions
  (gemini-3-pro high: +414, gpt-5.1 high: +464)
- Blunder positions add noise and are excluded from all features

All evaluations use Stockfish depth 30.
Data source: results.json — 50 equal positions, 37 models with RD < 100.
"""


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == "__main__":
    # Example: A model with 500 CPL, 30% best, 5% illegal rate
    equal_cpl = 500
    best = 30
    illegal_rate = 0.05
    surv40 = survival_probability(illegal_rate, 40)

    print(f"Example predictions for eq_CPL=500, best_pct={best}%, illegal_rate={illegal_rate}:")
    print(f"  Recommended: {predict_rating(equal_cpl, best, surv40):.0f}")
    print(f"  2-feat:      {predict_rating_2feat(best, surv40):.0f}")
    print(f"  2-feat alt:  {predict_rating_2feat_cpl(equal_cpl, best):.0f}")
    print(f"  Simple:      {predict_rating_simple(equal_cpl):.0f}")

    # Survival probability examples
    print("\nSurvival probability (40-move game):")
    for rate in [0, 0.01, 0.02, 0.04, 0.05, 0.10, 0.20]:
        print(f"  {rate:5.1%} illegal -> {survival_probability(rate, 40):5.1f}% survive")

    # Verification on known models
    print("\nVerification on eubos (actual: 2211, 0% illegal, best=60%):")
    s = survival_probability(0.0, 40)
    print(f"  Recommended: {predict_rating(20.0, 60.0, s):.0f}")
    print(f"  2-feat:      {predict_rating_2feat(60.0, s):.0f}")

    print("\nVerification on maia-1100 (actual: 1628, 0% illegal, best=56%):")
    s = survival_probability(0.0, 40)
    print(f"  Recommended: {predict_rating(264.0, 56.0, s):.0f}")
    print(f"  2-feat:      {predict_rating_2feat(56.0, s):.0f}")

    print("\nVerification on gemini-3-pro-preview high (actual: 1884, 0% illegal, best=62%):")
    s = survival_probability(0.0, 40)
    print(f"  Recommended: {predict_rating(487.9, 62.0, s):.0f}")
    print(f"  2-feat:      {predict_rating_2feat(62.0, s):.0f}")

    print("\nVerification on gemini-3.1-flash-lite-preview (actual: 301, 6% pos-illegal/12% game-illegal, best=46%):")
    s_pos = survival_probability(0.06, 40)
    s_game = survival_probability(0.12, 40)
    print(f"  Recommended (pos illegal):  {predict_rating(560.2, 46.0, s_pos):.0f}  (surv={s_pos:.1f}%)")
    print(f"  Recommended (game illegal): {predict_rating(560.2, 46.0, s_game):.0f}  (surv={s_game:.1f}%)")
    print(f"  Note: position benchmark overestimates legality -> large error with pos illegal rate")
