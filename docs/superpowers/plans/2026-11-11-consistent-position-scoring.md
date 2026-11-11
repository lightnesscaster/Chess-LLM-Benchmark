# Consistent position scoring implementation plan

> For agentic workers: use superpowers:subagent-driven-development to implement and review this plan.

**Goal:** Remove stale-baseline CPL errors, rescore saved responses without model calls, and replay production ratings from corrected benchmark inputs.

**Architecture:** Share a versioned Stockfish root scorer across static panels and continuation probes. Cache references and forced-move scores by engine profile, depth and complete board history. Rescore copies first; publish corrected inputs and ratings with concurrency guards and durable backups.

**Tech stack:** Python, python-chess, SQLite, Firestore, pytest.

**Spec:** User authorization in this chat. Keep depth 30, prompts, retry policy, model responses and rating formula unchanged. Existing checkout is explicitly required. No new games or model calls for rescoring.

## Interfaces and tasks

1. Create `position_benchmark/evaluation.py`: `SCORING_VERSION`, `ConsistentEvaluator(engine, depth=30, cache_path=None)` and `.score(board, move_or_none)` returning JSON fields `best_move`, `best_move_san`, `eval_before`, `eval_best`, `eval_model`, `cpl`, `is_best`, `scoring_version`, `scoring_profile`. Reference and selected move are searched at the same root/depth. Reset engine state before cache misses. Cache keys include full history and engine profile. Best reference move reuses its score, so CPL is zero. Illegal moves retain existing penalty policy. Tests cover stale baseline, repeated calls, history separation, black perspective and mate conversion.
2. Integrate static LLM/engine and continuation scorers. Preserve all response/usage/retry fields and original history. Add CPU-only rescoring of eligible saved records into isolated output files; recompute summaries using existing helpers. Tests prove no model calls and consistent repeated scores. Do not promote legacy prompt rows.
3. Add scoring provenance validation and document migration. Keep old data readable until corrected records are available; prevent mixed scoring from entering a corrected production table. Create backups and calculate old/new predictions and ratings from explicit snapshots. Validate unchanged game identities and counts, immutable response data, best-move zero CPL and consistent provenance.
4. Run focused and full tests, review changes, finish all already-running acquisition jobs, rescore the final snapshot, atomically publish corrected records and replayed ratings with revision/rating guards. Verify production records, applied revision and website values. Commit only source, documentation and canonical result files.

## Execution ledger

- Plan reviewed: tasks 1 and 2 share the evaluator interface above; task 3 consumes corrected records, not mutable live files. Task 4 waits for all raw-response writers. No overlapping ownership is required.
- Ruling: preserve the existing checkout and existing authorized publication scope; no additional workspace or deploy confirmation is needed.
- Ruling: retain historical scoring/calibration artifacts unchanged for audit; record the correction as a separate scoring version without changing predictor coefficients.
- Task 1: complete. Shared root evaluator and regression tests pass; real Stockfish cache smoke verified.
- Task 2: complete. Reviewed fixes include zero-score forfeit provenance, verified resume output digests, and full engine audit metadata retained once per summary. CPU rescoring is running against isolated copies while remaining raw Opus responses finish.
- Task 3: complete. Durable backups, source evidence, collection snapshots, byte budgets and idempotency are tested. Real production preflight identified legacy-cloud fallback and registry-label differences; both have guarded handling. Preserve the existing production token aggregate for Gemini 3.5 medium rather than changing usage accounting during this correction.
- Task 4: in progress. Full suite: 832 tests and 21 subtests pass with `GRPC_ENABLE_FORK_SUPPORT=0`; isolated web client tests also pass (34). The environment flag avoids an intermittent macOS gRPC/Node fork crash under concurrent engine load. Additional focused review fixes pass their regression tests. No corrected data has been published yet.
- Ruling: store a compact hash identity per score and the full engine profile once per model. Repeating the full engine option map on every move would exceed the atomic Firestore request budget. Cache keys retain the original complete profile, so ongoing CPU analysis is reusable.
