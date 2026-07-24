# Gemini 3.6 Flash accounting audit

The first `gemini-3.6-flash (medium)` acquisition and game cohort was run on
2026-07-24. Its chess moves, outcomes, legality metrics, CPL measurements, and
rating evidence are valid.

Its recorded cost metadata is not valid:

- the direct Gemini client recorded visible candidate tokens but omitted hidden
  thinking tokens, which Google bills at the output-token rate;
- the 15 concurrently scheduled games reused one mutable player object, so
  per-game token and timing counters reset and accumulated across games.

The canonical position records and the prospective shadow prediction remain
unchanged because the prediction was immutably locked before rated games. Their
stored token totals must therefore be treated as incomplete, not as billed
cost. The affected game-side token and timing fields were cleared in Firestore
and marked unavailable by `scripts/repair_gemini36_accounting.py`; game results
were preserved.

For dashboard display only, the 31 continuation calls whose complete
`total_tokens` values survived cost $0.23577 at published pricing, or
$0.007605 per call. Extrapolating that observed rate across 343 model turns in
15 games gives an explicitly marked estimate of approximately $0.174 per game.
This is not an authoritative reconstruction of the original bill.

Future runs clone the configured LLM player for each game and count Gemini
candidate plus thought tokens as billed completion tokens.
