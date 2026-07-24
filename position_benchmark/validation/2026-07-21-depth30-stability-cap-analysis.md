# Depth-30 continuation-cap audit — 2026-07-21

This zero-call audit evaluates the corrected depth-30 continuation artifacts. It
did not write the production rating store. The accompanying predictor change only
deduplicates catastrophe events within each trajectory. The cohort has
30 configurations across
17 model-line families.

## Result

The acquisition gate now passes: it requires at least
30 configurations and
8 families, and this cohort contains
30 configurations across
17 families. Passing that gate makes a redesign eligible
for review; it is not by itself an acceptance criterion.

The current move-level catastrophe count is structurally wrong for an absorbing
loss: repeated losing moves in one continuation are correlated and must not be
treated as independent catastrophe events. `deduplicated_move_exposure_cap`
implements a narrow correction: retain at most the first catastrophe in each
trajectory, preserve the existing move-exposure denominator and coefficients,
and therefore never make a cap harsher. This correction fits no target data.
`trajectory_hazard_cap` additionally censors later exposures; that more ambitious
survival-style redesign remains diagnostic rather than validated rating evidence.

`repeated_forfeit_only` is the leading fixed redesign. Its MAE is lower than the current cap
on both targets, but its family-bootstrap improvement probabilities are only
0.644 (RD-300) and
0.796
(no-position), and both family-bootstrap intervals cross zero. Only
5 configurations receive a different
prediction. The apparent gain is highly dependent on `gpt-3.5-turbo-instruct`:
excluding that family changes the candidate-minus-current MAE delta to
+14.6 Elo
on RD-300 and
+1.0
Elo on no-position, so the candidate becomes slightly worse.

Lab-level dependence is stronger still. Excluding `openai` changes the
candidate-minus-current MAE delta to
+20.4 Elo on
RD-300 and
-8.9
Elo on no-position. The lab-bootstrap improvement probabilities are
0.653 and
0.856,
respectively.

Accordingly, the production decision is **hold**. Keep the already deployed
within-trajectory catastrophe deduplication, but do not replace the remaining CPL
cap with `repeated_forfeit_only` on this cohort. Existing cap constants, the 150-Elo deadband,
continuation legality, and forfeit evidence remain unchanged.

## Current depth-30 comparison

### Position-seeded validation ratings with RD 300

| Candidate | MAE | RMSE | Bias | IVW MAE | Family bootstrap P(improves) | Lab bootstrap P(improves) | RD simulation P(improves) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_move_cap` | 227.3 | 322.2 | +34.3 | 214.5 | 0.000 | 0.000 | 0.000 |
| `deduplicated_move_exposure_cap` | 236.2 | 325.7 | +49.9 | 231.7 | 0.000 | 0.000 | 0.135 |
| `trajectory_hazard_cap` | 235.7 | 325.9 | +49.1 | 230.9 | 0.082 | 0.079 | 0.137 |
| `two_affected_trajectory_gate` | 230.3 | 323.5 | +44.0 | 215.7 | 0.000 | 0.000 | 0.348 |
| `repeated_forfeit_only` | 207.2 | 265.6 | +107.1 | 171.0 | 0.644 | 0.653 | 0.945 |
| `no_hard_cap` | 215.0 | 277.4 | +114.8 | 176.9 | 0.592 | 0.588 | 0.826 |
### No-position-seed game ratings

| Candidate | MAE | RMSE | Bias | IVW MAE | Family bootstrap P(improves) | Lab bootstrap P(improves) | RD simulation P(improves) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_move_cap` | 282.6 | 376.5 | +3.9 | 220.6 | 0.000 | 0.000 | 0.000 |
| `deduplicated_move_exposure_cap` | 278.8 | 371.4 | +19.5 | 235.6 | 0.541 | 0.553 | 0.808 |
| `trajectory_hazard_cap` | 278.3 | 371.6 | +18.7 | 234.9 | 0.622 | 0.627 | 0.803 |
| `two_affected_trajectory_gate` | 272.9 | 369.5 | +13.6 | 219.5 | 0.644 | 0.657 | 0.888 |
| `repeated_forfeit_only` | 249.3 | 319.6 | +76.7 | 175.4 | 0.796 | 0.856 | 0.992 |
| `no_hard_cap` | 257.1 | 329.6 | +84.5 | 181.0 | 0.722 | 0.758 | 0.971 |

All bootstrap and uncertainty probabilities compare the candidate against
`current_move_cap`. Candidates are fixed structural alternatives; no coefficients
were fitted to these targets.

## Rows whose rating is currently changed by the hard cap

| Model | Catastrophic moves | Affected catastrophe starts | All affected starts | Current | Deduplicated | Hazard-style | No hard cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemini-2.5-flash (no thinking) | 4 | 2 | 3 | 461 | 637 | 637 | 637 |
| gpt-5.1 (no thinking) | 4 | 4 | 4 | 550 | 550 | 531 | 747 |
| gpt-5.6-luna (high) | 2 | 2 | 2 | 598 | 598 | 595 | 1090 |
| deepseek-v4-flash (max) | 3 | 1 | 1 | 575 | 867 | 867 | 867 |
| gemini-3.1-flash-lite-preview | 0 | 0 | 2 | 500 | 500 | 500 | 732 |
| gpt-3.5-turbo-instruct | 2 | 2 | 2 | 600 | 600 | 598 | 1627 |

## Historical sensitivity

The stale June depth-10/pre-stratification snapshot is not current validation, but
it explains why the cap existed. Its three-row MAE was
107.3 with the
old cap, 298.7
with repeated-forfeit-only protection, and
399.3 with no hard cap.
Thus repeated-forfeit protection retains much of the historical weak-model value,
while the present cohort does not validate a direct random-reply CPL cap.

## Decision boundary

- The acquisition coverage gate passes, but no statistical acceptance threshold
  was preregistered; do not invent one after observing these results.
- `repeated_forfeit_only` advances as the leading candidate for independent holdout review,
  not as an automatic production change.
- Do not fit new cap coefficients on 30 configurations.
- Do not describe RD-300 as independent; it still shares the benchmark prior.
- Keep production catastrophe deduplication within a trajectory because this
  cannot increase any penalty and did not fit the validation targets.
- Do not deploy hazard censoring, new coefficients, or cap removal on this cohort.
- Re-run this fixed audit automatically as current depth-30 supplements accumulate.
