# GPT-5.6 live-game illegality forensics — 2026-07-15

This is a no-model-call, read-only analysis of all 112 saved GPT-5.6 games. It
reconstructs every GPT-5.6 turn from the saved PGN, joins stored illegal-response
details to the board by ply, and compares illegal turns with all legal first
attempts. The reproducible implementation is
`scripts/analyze_game_illegal_moves.py`; the complete aggregate tables and
sanitized event records are in
`validation/2026-07-15-gpt56-illegal-forensics.json`.

This report diagnoses measurement gaps. It does not alter the production rating
formula, ratings, benchmark panels, or saved games.

## Data integrity

- 112 of 112 selected games were reconstructed successfully.
- 4,000 GPT-5.6 first-attempt turns contained 93 illegals (2.325%).
- Two same-turn retries were also illegal.
- These totals independently reproduce the existing retry audit: 93 first
  illegals, 66 eligible retries, 64 recoveries, two failed retries, and 27 later
  second strikes after a prior recovery.
- Full prompts are not copied into the artifact. Event rows contain only the
  reconstructed chess state, saved response, classification, and control
  features needed to reproduce the analysis.

## What the illegal answers were

The dominant problem was chess legality or board-state reasoning, not failure to
follow the requested response format.

| Primary class | Events | Share of 93 |
| --- | ---: | ---: |
| Movement rule or blocked path | 41 | 44.1% |
| Empty source square | 18 | 19.4% |
| Invalid UCI form or coordinates | 13 | 14.0% |
| Pseudo-legal move exposing or leaving the king in check | 13 | 14.0% |
| Destination occupied by own piece | 7 | 7.5% |
| Opponent's piece selected | 1 | 1.1% |

Eighty-two of 93 illegal responses (88.2%) were syntactically well-formed UCI.
All 93 responses were either a bare move (67) or used the accepted `MOVE:` prefix
(26); none contained explanatory text. The 11 malformed strings were generally
explicitly incomplete answers such as `c?`, `d3d?`, or `MOVE: Nh?` rather than
verbosity. Formatting enforcement alone therefore cannot close the gap.

The most common detailed failures were empty source square (18), king safety
(13), blocked slider path (12), bishop geometry (11), knight geometry (11), and
invalid format (11). Eleven events had at least one stale-board signal: three
repeated the model's preceding move, seven would have been legal on its preceding
turn, and four would have been legal if the opponent's last move were ignored.
The signals overlap. Stale-board mistakes are real, but 11 of 93 events is too
small to explain most failures.

## Where the risk appeared

Pooled rates can be confounded because the 12 configurations played different
numbers and mixes of turns. The associations below therefore use a
player-stratified Mantel-Haenszel odds ratio and a deterministic 1,000-sample
game-cluster bootstrap.

| Context | Exposed rate | Control rate | Stratified OR | 95% game-cluster interval |
| --- | ---: | ---: | ---: | ---: |
| Ply 21–60 | 3.14% | 1.78% | 1.70 | 1.06–2.74 |
| History length 20–39 plies | 3.41% | 1.97% | 1.62 | 0.92–2.50 |
| Previous move was a capture | 3.31% | 2.05% | 1.57 | 0.94–2.45 |
| Side to move retains castling rights | 1.39% | 2.73% | 0.40 | 0.21–0.63 |
| GPT opponent rather than Maia/random | 2.54% | 2.06% | 1.04 | 0.68–1.62 |
| A prior illegal strike already occurred | 2.52% | 2.25% | 0.90 | 0.53–1.43 |

Only the ply-21–60 increase and the protective castling-rights association have
intervals excluding one. The latter is partly a phase/development proxy and must
not be interpreted causally. Capture context is suggestive, not conclusive.
Material phase and legal-move count were weak. Check, promotion availability, and
opponent family did not identify a useful high-risk slice.

The no-castling association remains visible within the main risk window: at plies
21–60, turns without castling rights were illegal on 48/1,303 attempts (3.68%),
versus 3/321 (0.93%) with rights. After-capture rates were also higher within both
the middlegame and late-game strata.

The configurations that exposed the benchmark miss most clearly were:

| Configuration | Live first-attempt rate | Main concentration |
| --- | ---: | --- |
| GPT-5.6 Luna medium | 24/381 = 6.30% | 15/159 at plies 21–60; 14/111 at history 20–39 |
| GPT-5.6 Terra low | 19/482 = 3.94% | Elevated through the middlegame and later histories |
| GPT-5.6 Sol high | 3/311 = 0.96% | No comparable concentration |

Luna medium's rate was similar against GPT, Maia, and random opponents, which
argues against a particular opponent type being the explanation. Its illegals
were mostly movement/blocked-path errors (14), empty-source errors (6), and king
safety errors (3), with only two stale-board signals.

## Why static rebalancing is insufficient

The existing panels already contain the visible high-risk contexts:

| Panel | Ply 21–60 | History 20–39 | No castling rights | Previous move captured |
| --- | ---: | ---: | ---: | ---: |
| Core equal 50 | 20/50 | 17/50 | 26/50 | 15/50 |
| Game-like 48 | 32/48 | 29/48 | 34/48 | 11/48 |
| Legality-stress 6 | 2/6 | 2/6 | 3/6 | 2/6 |

The game-like panel is already 66.7% middlegame by ply and 70.8% without castling
rights. Nevertheless, Luna medium was illegal on only 1/48 core rows and 1/48
game-like rows, versus 24/381 live turns. Selecting more independent positions by
these same static attributes would therefore overfit the observed cohort without
measuring the missing mechanism.

The most plausible missing variable is continuous, model-generated context. Each
static row replays a fixed human/game history and then asks for one move. Live
games repeatedly feed the model a history containing its own earlier choices and
the consequences of those choices. The current continuation probe begins to test
that mechanism, but resets after only four model moves and uses a random opponent,
whose moves make later CPL unsuitable as chess-quality evidence.

## Protocol-sequence-v1 proposal and subsequent decision

The next paid experiment should keep the current continuation call budget while
putting more of it into uninterrupted sequences:

1. Use four frozen starting positions from `game_like_48.json`: the first position
   in each of advantage conversion, defense, quiet/equal, and tactical/equal.
   This category-round-robin choice predates these GPT-5.6 outcomes.
2. Request eight model moves per start rather than four model moves on each of
   eight starts. This remains exactly 32 ordinary model calls per configuration,
   plus failure-dependent production retries.
3. Retain seeded random legal replies and persist every opponent move. Score both
   sides with Stockfish afterward so random-stimulus severity remains measurable.
4. Use the exact production prompt and global two-strike policy. Record first
   illegality `p`, conditional retry failure `q`, the turn of each strike,
   recovery, and production forfeit. Do not grant a fresh retry after a recovered
   first strike.
5. Preserve later-ply CPL as noisy chess-quality evidence alongside raw, capped,
   median, catastrophe, and opponent-CPL diagnostics. Do not alter production
   coefficients from this within-GPT-5.6 analysis alone.
6. Pilot only the discriminator configurations Luna medium, Terra low, and Sol
   high: 96 ordinary calls total, plus conditional retries. Freeze the runner and
   starts before those calls. Compare the sequence rates with each model's live
   game interval and static-panel rate; do not require individual model-position
   failures to repeat.
7. Require a later cross-family validation before using this feature in production.
   GPT-5.6 supplied the diagnosis and therefore cannot also be independent proof
   of the panel or a newly fitted coefficient.

Four longer sequences are preferable to eight short ones here because the planned
maximum call count is unchanged while the model receives twice as many consecutive
own moves before reset. Natural termination or a production forfeit can reduce the
actual count. The tradeoff is fewer independent starts, so uncertainty must be
clustered by sequence and reported alongside the raw turn rate.

The completed pilot separated Luna medium, Terra low, and Sol high and reproduced
their live aggregate illegality closely. Results and updated conclusions are in
`validation/2026-07-15-protocol-sequence-pilot.md`.

## Production decision

No production change is justified yet. Keep the 50-position core, the current
downside-only supplemental behavior, and the validated rating coefficients
unchanged. The sequence protocol and three-model pilot now exist in a separate
research result file with an explicit `production_effect: none`; broader
predictive analysis and held-out validation are the next gates.
