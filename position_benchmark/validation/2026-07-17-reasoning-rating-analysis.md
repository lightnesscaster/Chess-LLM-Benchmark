# Final-rating reasoning-level prediction — 2026-07-17

This analysis predicts **final production game-benchmark ratings** from one or
two other final production ratings in the same model line. The ordinary
position-benchmark initialization is already reflected in every rating; no
position features or separate position blend are used.

The primary `lab_curve_release_holdout` result excludes the entire target
release cohort while learning the prior. In particular, Luna, Terra, and Sol
are hidden together when GPT-5.6 is treated as a new release.

## Data and leakage control

The production rating store contains 27 underlying model lines with at least
two effort-like suffixes. Nineteen use comparable ordinal labels from `no
thinking` through `xhigh` (`max` is normalized to `xhigh`). Eight additional
binary `(thinking)` curves were inventoried but not forced onto Medium or High,
because that label represents incompatible budgets across labs.

Scored targets and anchors require at least eight games and RD no greater than
220. Sparse observations remain available to the RD-weighted prior but are not
treated as validation targets. Every scored prediction excludes the target
model line, and the primary result also excludes all siblings in the target
release cohort.

## Results

| Policy | Lines | Carry MAE | Global MAE | Lab/release-held MAE | RMSE | Max error |
|---|---:|---:|---:|---:|---:|---:|
| Medium → High | 8 | 273 | 313 | 316 | 484 | 1,220 |
| Low + Medium → High | 5 | 171 | 211 | 214 | 266 | 485 |
| Medium → Xhigh | 4 | 405 | 336 | 341 | 411 | 670 |
| Low + Medium → Xhigh | 4 | 405 | 324 | 324 | 386 | 533 |
| High → Xhigh | 4 | 229 | 173 | 174 | 232 | 416 |

`Carry` means using the highest tested level unchanged. The hierarchical model
learns RD-weighted adjacent-effort increments, partially pooling lab estimates
toward the global curve. With two anchors, Xhigh uses the mean target rating
implied independently by both anchors; High uses the nearer Medium anchor.

Low plus Medium does not improve the High point-estimate MAE over simply using
Medium, although it reduces the error tail: RMSE falls from 279 for carry to 266
for the hierarchical estimate, and maximum error falls from 602 to 485. For
Xhigh, the hierarchy improves MAE from 405 to 324 but remains much too uncertain
to substitute for measurement. Once High is known, Xhigh is more predictable:
MAE is 174 and RMSE is 232.

## GPT-5.6 prospective simulation

The following predictions hide all three GPT-5.6 lines together and learn only
from earlier releases and other labs:

| Line | Low+Medium → High | Actual High | Low+Medium → Xhigh | Actual Xhigh | High → Xhigh |
|---|---:|---:|---:|---:|---:|
| Luna | 624 | 1,109 | 890 | 1,371 | 1,185 |
| Sol | 1,510 | 1,538 | 1,613 | 1,612 | 1,614 |
| Terra | 856 | 766 | 978 | 1,259 | 842 |

This is the decisive negative result. GPT-5.5's nearly flat final-rating curve
did not predict GPT-5.6's much larger Xhigh gains, especially for Luna and
Terra. Pooling across labs cannot recover a release-specific change that is not
visible in Low and Medium.

## Recommended use

The new predictor is useful as a deliberately uncertain prior, not as a
replacement for running a level:

- Low plus Medium can supply a provisional High estimate with an empirical
  error scale around 270 rating points.
- Low plus Medium is not sufficient for a dependable Xhigh estimate; its error
  scale is about 390 points.
- High is the best single anchor for Xhigh, with an error scale around 230
  points. The practical cost-saving policy is therefore to measure through High
  and predict only Xhigh.
- A predicted level should retain a much larger uncertainty than a directly
  measured final rating and should never be presented as equivalent evidence.

No production rating, position initialization, or seeding behavior is changed
by this research.
