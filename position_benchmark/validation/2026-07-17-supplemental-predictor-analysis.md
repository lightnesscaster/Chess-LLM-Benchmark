# Supplemental predictor family-held-out analysis — 2026-07-17

The frozen decision is **retain-current-production-predictor**. No production formula or rating
initialization changed.

The primary all-panel cohort contains 12
configurations across 3 model-line families. It consists
entirely of the twelve GPT-5.6 variants and fails the predeclared 30-model,
eight-family evidence requirement. The broader game-like cohort contains
15 configurations across 6
families.

## Broader game-like comparison

| Predictor | MAE | RMSE | Mean error | Family-bootstrap P(MAE improves) |
| --- | ---: | ---: | ---: | ---: |
| game_like_hard_cap_fixed | 238.3 | 322.4 | -186.1 | 0.670 |
| production_fixed | 239.1 | 323.0 | -155.5 | 0.000 |
| lofo_game_like_aggregate | 248.4 | 295.8 | -7.0 | 0.419 |
| lofo_legality | 267.7 | 315.1 | +3.6 | 0.245 |
| continuous_deadband_fixed | 288.3 | 381.4 | -236.1 | 0.001 |
| lofo_game_like_categories | 341.8 | 414.6 | -81.9 | 0.021 |
| core_fixed | 354.8 | 476.7 | -302.6 | 0.001 |

The current downside system is materially better than the core alone on this
cohort: core MAE/RMSE are 354.8/476.7,
versus 239.1/323.0
for production. None of the learned replacements improves family-held-out MAE
over production.

## All-panel prediction comparison

| Predictor | MAE | RMSE | Mean error | Family-bootstrap P(MAE improves) |
| --- | ---: | ---: | ---: | ---: |
| game_like_hard_cap_fixed | 227.3 | 309.4 | -165.9 | 0.000 |
| production_fixed | 227.3 | 309.4 | -165.9 | 0.000 |
| lofo_game_like_aggregate | 230.3 | 296.9 | +0.7 | 0.421 |
| lofo_legality | 233.5 | 298.3 | +3.5 | 0.411 |
| lofo_continuation | 236.5 | 300.6 | +8.2 | 0.398 |
| lofo_game_like_categories | 243.2 | 302.9 | +7.5 | 0.363 |
| continuous_deadband_fixed | 277.3 | 361.6 | -215.9 | 0.000 |
| core_fixed | 324.8 | 405.2 | -263.4 | 0.000 |

The numerically best candidate is `game_like_hard_cap_fixed`. Failed gate checks:
configuration_count, family_count, mae_improvement, rmse_improvement, family_win_fraction, bootstrap_probability. Even a favorable point estimate is therefore a research lead,
not a production candidate.

## Feature relationship to the remaining production error

| Feature | Spearman correlation with actual − production prediction |
| --- | ---: |
| category_defense | -0.643 |
| pooled_illegal_rate | -0.456 |
| game_like_gap | 0.308 |
| category_advantage_conversion | -0.287 |
| category_tactical_equal | -0.245 |
| retry_failure_rate | 0.219 |
| continuation_log_cpl | -0.091 |
| category_quiet_equal | 0.028 |

Correlations describe this small, selected cohort. Continuation CPL is retained as
evidence as planned, while the report keeps its random-opponent confounding and
depth-10 scoring limitation explicit.

The family contrast matters:

- Defense CPL correlates -0.643
  with the remaining error inside GPT-5.6, but only
  -0.157
  in the broader six-family game-like cohort. The earlier defense result does not
  generalize well enough to weight.
- Pooled first-answer illegality is the strongest broader residual diagnostic
  (Spearman -0.617),
  but its leakage-controlled model still worsens MAE. Keep it as a reliability
  diagnostic, not a fitted rating coefficient.
- Continuation CPL has little relationship to the remaining error here
  (Spearman -0.091)
  and adding it does not improve held-out MAE.
- The fixed production and game-like-hard-cap predictions are identical on all
  twelve ready continuation configurations. Thus the continuation forfeit/
  catastrophe cap adds no further adjustment in this cohort, although
  continuation legality still participates in the game-like panel's conservative
  legality input.
- Retry-failure evidence is still too sparse to fit a reliable model-specific
  correction.

## Coverage

- Game-like cohort: 15 configurations,
  6 families.
- Continuation cohort: 12 configurations,
  3 families.
- All-panel cohort: 12 configurations,
  3 families.

The RD-300 game target comes from 6,276 production games, with five invalid games
skipped by the existing recalculation contract. Its prior mean still comes from
the position benchmark, so this analysis reduces circularity but cannot eliminate
it. The next useful evidence is automatic completion of these supplements on more
new or actively selected non-GPT-5.6 model families, followed by rerunning this
frozen harness. Existing frozen models remain excluded by the production
acquisition policy unless a separately authorized research run is justified.
