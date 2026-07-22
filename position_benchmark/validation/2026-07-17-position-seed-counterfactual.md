# Position-seed counterfactual — 2026-07-17

All 6,276 production games were recalculated in an isolated rating store
with position-benchmark predictions disabled. Non-anchor models therefore
started from the ordinary legality/model-type fallback with RD 350; normal
later-pass lower-effort sibling seeding was retained. Production ratings
were not changed.

## High final rating + Xhigh-position prediction

Each High-only baseline uses a High→Xhigh increment learned only from
earlier releases, excluding the target release cohort. GPT-5.5 therefore
uses a zero increment: no earlier OpenAI line had an Xhigh game rating.

| Position weight | MAE | RMSE | Bias |
|---:|---:|---:|---:|
| 0% | 313 | 370 | -267 |
| 25% | 217 | 303 | -195 |
| 50% | 175 | 262 | -123 |
| 75% | 203 | 260 | -51 |
| 100% | 249 | 298 | +21 |

| Model | Independent Xhigh | RD | High+curve | Position | 50/50 |
|---|---:|---:|---:|---:|---:|
| GPT-5.5 | 532 | 62 | 388 | 936 | 662 |
| GPT-5.6 Sol | 1664 | 186 | 1779 | 1539 | 1659 |
| GPT-5.6 Terra | 976 | 199 | 615 | 1246 | 930 |
| GPT-5.6 Luna | 1288 | 205 | 1009 | 1280 | 1145 |
| DeepSeek V4 Flash | 988 | 218 | 323 | 550 | 437 |

Across all five exact cases, High-only MAE is 313,
position-only MAE is 249, and the fixed 50/50 blend
is best on the fixed grid at 175 MAE
(262 RMSE).

GPT-5.5 is a clean chronological holdout for the already-proposed 50/50
blend. Its absolute errors are 144
for High-only, 404 for position-only,
and 130 for the fixed blend. Thus it
confirms a modest incremental benefit over High alone while strongly
rejecting position-only prediction for this release.

DeepSeek V4 Flash is the first frozen cross-lab test, and it is a clear
miss: High-only error 665,
position-only error 438, and
50/50 error 551. The blend still
has the lowest aggregate MAE, but this case shows that the OpenAI result
does not transfer cleanly across labs.

The evidence now spans five configurations, two labs, and three model
releases. It is still too small to tune or deploy a production weight.
