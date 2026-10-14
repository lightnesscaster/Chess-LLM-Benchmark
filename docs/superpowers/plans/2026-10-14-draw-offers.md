# Draw offers implementation plan

Goal: allow models and humans to agree to rated draws through an explicit offer/accept exchange.

Use the existing checkout and preserve generated results and running games. Existing CLI processes retain their imported game rules. No engine evaluation or automatic draw acceptance is introduced.

## Contract

- Models output a normal move, `resign`, `<uci> offer_draw`, or (only when facing a pending opponent offer) `accept_draw`. Offers require a legal move. No standalone offer and no unilateral acceptance without a pending offer.
- Shared `llm.protocol.parse_draw_action(text) -> str | None` recognizes only exact draw responses; `split_draw_offer(action) -> tuple[str, bool]` splits a canonical move plus offer. Existing move parsing remains unchanged otherwise.
- `request_llm_move` adds keyword defaults `allow_draw_offers=False`, `opponent_draw_offer=False`, restores these request attributes in `finally`, and leaves benchmark prompts unchanged. Games enable offers only when the opponent can respond (another model or a human).
- A pending offer lasts until the opponent accepts, explicitly declines, resigns, or plays a legal move. Illegal retries preserve it; failed offers do not survive an illegal move. Natural terminal positions take precedence over an attached offer.
- Persist server-owned human-game fields `draw_offer` (`white`, `black`, or null) and `draw_events` (list of `{action, by, ply}` dictionaries). Offers occur after a move, at its resulting ply; acceptance/decline annotate that same ply before the next legal move. CLI uses the same event schema in memory.
- `game.draw_offers.annotate_draw_events(game, events)` annotates PGN nodes with exact `[%draw_offer white]`, `[%draw_accept black]`, or `[%draw_decline black]` comments. It validates event shape, color, sequence, board ply, and opposing participants. No GameResult schema migration is needed.
- Accepted offers finish as winner `draw`, termination `draw_agreement`. Saved rated benchmark publication validates a nonterminal final board and matching final-node offer/accept annotations for opposing sides. Keep publication validation self-contained because existing background CLI games ship this worker source before a deployment.
- Human controls: offer with next move, accept a model offer, or decline/play on. Use existing account lease, exact game/FEN/ply checks, durable completion journal, CSRF, and idempotent rating publication. Client never supplies pending-offer state.
- Live games expose the current offering color and show draw-by-agreement results. Archived PGNs retain the agreement evidence.

## Tasks and ownership

1. Protocol/providers: `llm/protocol.py`, `llm/base_llm.py`, `llm/prompts.py`, provider clients, focused protocol tests. Add capability-gated prompts and strict parsing across every backend; benchmarks retain existing prompts. Write failing tests first.
2. Game orchestration (primary agent): `game/draw_offers.py`, `game/game_runner.py`, `game/publication.py`, `game/live_games.py`, live UI, runner/publication/live tests. Enforce negotiation, preserve retries/counts and PGN proof, and publish pending offers.
3. Human arena: `web/play_service.py`, `web/app.py`, `web/human_challenges.py`, play template/JS/CSS and focused tests. Add offer/accept/decline without weakening authoritative turn ownership or repeating paid calls.
4. Independently review each area and the integrated result; fix material findings. Run full Python suite, existing JS checks and local rendered-page verification. Commit only feature sources/docs/tests. Report deployment status accurately; no retroactive edits to completed games or interruption of running matches.

Validation must cover offer/accept, rejection by a move, counteroffers, stale/wrong-side acceptance, illegal-offer retry, engine opponents, terminal moves, promotion, probe isolation, token accounting, durable web acceptance/rating idempotency, and publication proof rejection.

## Verification and review record

- Protocol/providers complete: 48 focused tests; independent review also confirmed 32 disabled-draw prompt combinations remain byte-identical to the previous implementation.
- Runner/publication/live complete: explicit two-player agreement, legal-move offers, retry preservation, counteroffers and exact final-PGN evidence. Same-ply reoffers and malformed event shapes are rejected. No migration of result schemas was required.
- Human arena complete: authoritative actions use the existing leased move endpoint and durable completion/rating path. The offer and accept/decline controls appear inside the position card beside the board.
- Review found a recovery edge case where offers from replaced moves survived a correction. The fix retains only valid prefix history, consumes any retained pending offer when applying the replacement move, and clears all unverified offers from the replaced turn. Five failing regression cases were observed before fixing; bounded re-review passed.
- Final full Python suite: **649 tests and 21 subtests passed**. Live-viewer JS tests, syntax checks and compileall passed. Existing deprecation warnings remain unchanged.
- Official `cli.py manual --no-save` smoke with synthetic in-memory players exercised the real runner and ended in `draw_agreement` after one ply with zero illegal responses. No model calls or rating writes were made by validation.
- Desktop and mobile local previews checked; no production or human account state was modified by previewing.
