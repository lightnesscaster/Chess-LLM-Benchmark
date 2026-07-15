# Supplemental benchmark review — 2026-07-14

This review followed the completed 96-game GPT-5.6 matchup set. It is a research
record, not a replacement for the frozen June validation snapshot.

## Findings

1. The continuation runner's default eight starts were panel indices 0 through 7.
   Because `game_like_48.json` is category-ordered, all eight were
   advantage-conversion positions. Those summaries were not representative.
2. The GPT-5.6 game-like estimates were optimistic against opponent-adjusted game
   performance. Raw standalone error across the 12 configurations was MAE 381 Elo
   with +348 Elo mean bias.
3. That optimism was not universal. Among eight reliable historical models with
   results on the exact same 48-position panel, raw standalone game-like error was
   MAE 142 Elo with +10 Elo mean bias. A global offset or affine rescaling would
   therefore overfit the GPT-5.6 family and was not adopted.
4. Legality is more predictive when contexts are kept visible. On GPT-5.6, using
   the lowest rate across core, game-like, and the available continuation probe
   reduced standalone game-like error to MAE 267 Elo and +217 Elo bias. The full
   predictor changed from MAE 242 to 235 Elo against the independent game estimate;
   log loss improved from 0.634 to 0.632 and Brier score was effectively unchanged
   (0.1533 to 0.1538). These figures use the old, unrepresentative continuation
   rows only as a diagnostic and do not make those rows production-eligible.

## Decisions

- Keep the game-like panel at 48 calls and the continuation probe at eight starts;
  do not expand the workload merely to fix the selection bug.
- Select continuation starts deterministically in category round-robin order,
  giving two starts to each of the four game-like categories.
- Require `stability_probe_version: stratified-v2`. Earlier continuation summaries
  remain stored but cannot affect current predictions.
- Report the pooled legality rate as a diagnostic. Score the game-like downside
  check with the lowest eligible panel legality rate so averaging cannot conceal a
  context-specific failure or improve the estimate.
- Do not apply a global game-like calibration offset. Re-evaluate calibration after
  representative continuation reruns and more cross-family game evidence.

## Higher-RD validation target

Production benchmark seeds remain RD 166. A guarded local counterfactual was added
for validation only. It starts at the same position-benchmark mean with a specified
higher RD and writes ratings to a non-production file.

For RD 300, the 12 GPT-5.6 ratings had MAE 66 Elo and +28 Elo mean bias against the
opponent-adjusted estimates derived from the same 96 games, compared with MAE 159
and +80 Elo for the current production ratings. This is not independent predictive
validation—the counterfactual and opponent-adjusted estimates share game outcomes.
It is a less prior-dominated target for evaluating the position predictor while
retaining the production rating system unchanged.

## Required follow-up

The 12 GPT-5.6 configurations were rerun under `stratified-v2` on July 14. Every
result contains the required panel indices `0, 12, 24, 36, 1, 13, 25, 37` and
passes production readiness. Historical models still require representative
reruns before their continuation evidence can become current.

The representative rerun materially changed the interpretation of the earlier
diagnostic. Only GPT-5.6 Luna low triggered the standalone stability cap, after one
of eight starts ended in an illegal-move forfeit. Luna high, Luna xhigh, and Terra
medium each had one 1000+ CPL move but no forfeits; their combined risk remained
below the five-percent activation threshold. The other eight configurations had
neither a forfeit nor a 1000+ CPL move.

Against the opponent-adjusted GPT-5.6 game estimates, the resulting full predictor
had MAE 277 Elo and +122 Elo mean bias. This is worse than the MAE 235 diagnostic
that reused the old, unrepresentative continuation rows. The old result was partly
fortuitous and must not be used to justify the former selector. The representative
panel fixes the measurement contract but does not, by itself, solve the remaining
Luna-medium and Terra-low overestimates.
