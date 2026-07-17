# Cross-family failure-transfer held-out result — 2026-07-17

The frozen primary decision is **FAIL — retain as research-only**.

## Primary result

| Endpoint | Failure states | Matched controls |
| --- | ---: | ---: |
| Illegal first answers | 6/24 (25.0%) | 4/24 (16.7%) |
| Mean depth-30 CPL | 2022.9 | 1220.5 |
| Legal-only mean depth-30 CPL | 1240.7 | 661.6 |

There were 4 failure-only and
2 control-only discordant pairs. The frozen
one-sided exact McNemar probability is 0.34375.
The direction was positive in 2 families:
google, qwen.

## By family

| Family | Failure illegals | Control illegals |
| --- | ---: | ---: |
| anthropic | 0/6 | 0/6 |
| deepseek | 2/6 | 2/6 |
| google | 2/6 | 0/6 |
| qwen | 1/3 | 0/3 |
| z-ai | 1/3 | 2/3 |

## By configuration

| Configuration | Failure illegals | Control illegals |
| --- | ---: | ---: |
| gemini-3.5-flash (medium) | 0/3 | 0/3 |
| gemini-3-flash-preview (minimal) | 2/3 | 0/3 |
| deepseek-v4-flash (no thinking) | 2/3 | 2/3 |
| deepseek-v4-flash (high) | 0/3 | 0/3 |
| claude-opus-4.6 (no thinking) | 0/3 | 0/3 |
| claude-opus-4.6 (medium) | 0/3 | 0/3 |
| glm-4.6 (thinking) | 1/3 | 2/3 |
| qwen3-235b-a22b-2507 | 1/3 | 0/3 |

## By frozen candidate pair

| Failure candidate | Failure illegals | Control illegals |
| --- | ---: | ---: |
| failure-transfer-luna-003 | 4/8 | 1/8 |
| failure-transfer-sol-001 | 1/8 | 1/8 |
| failure-transfer-sol-003 | 1/8 | 2/8 |

All illegal first answers triggered the production conditional-retry protocol.
Recovery was 8/10.

The run made 58 model calls and reports an actual
OpenRouter cost of $0.3190. The $0.3252
preflight bound was an estimated-cost admission guard, not a runtime spend stop.

## Interpretation

This validates or rejects only the frozen three-position shortlist. Because the
shortlist was selected using GPT-5.6 outcomes, it does not estimate the prevalence
of transferable failure positions in general. The observed lift was concentrated
in `failure-transfer-luna-003` (4/8 versus 1/8); the two Sol-sourced pairs were
flat or reversed. Passing would permit a broader, pre-frozen calibration
experiment; this result did not pass and has no production rating effect.
