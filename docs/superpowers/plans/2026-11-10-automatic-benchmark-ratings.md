# Automatic benchmark rating updates

**Goal:** Publishing a ready production position panel automatically applies its
seed to saved game history, including configurations that already have games.

**Architecture:** A changed, core-ready panel increments a durable Firestore
revision. Reuse CLI recalculation in isolated validation mode with explicit game
and prediction snapshots. Commit ratings, an audit receipt, and the applied
revision together, guarded against concurrent rating and benchmark changes.
The web cache observes the applied revision. Retry the same saved panel after
failure; never repeat model calls or apply an individual game twice.

**Constraints:** Existing checkout; production seed RD stays 166; engine anchors,
human snapshots, ghosts, historical anchor ratings, and game counts retain the
existing recalculation rules. No rating resets or intermediate production tables.
Identical panel retries do not increment the revision. Failed calculations retain
the pending revision. No credentials or generated data enter commits.

1. Add regression tests for explicit recalculation inputs and a ready panel
   updating an existing player's rating without changing its saved games.
   Verify a duplicate panel is idempotent, a game publication racing the update
   causes recomputation, and a failed calculation leaves a retryable revision.
2. Add `rating/benchmark_recalculation.py` to compute and atomically commit the
   snapshot. Extend `cli.recalculate_ratings` with optional `results_snapshot`
   and `benchmark_predictions` keyword arguments. Reuse shared position-input
   loading with an explicit remote snapshot rather than stale CLI-local seeds.
3. Update `position_benchmark/sync.py` to request and complete recalculation
   after a core-ready panel is stored and verified. Keep local-Firebase and
   authenticated Render publication paths equivalent.
4. Make `web/app.py` observe the applied revision, clear the rating/prediction
   caches on change, and rebuild the leaderboard. Test that unchanged revisions
   keep the warm cache and transient read failures preserve the previous page.
5. Run focused tests and the full suite, request read-only review, commit and
   deploy. Verify Carbon medium's completed panel has an automatic rating-update
   receipt and that its public rating matches Firestore. Continue the three
   already-authorized rated games and verify their publication receipts.
